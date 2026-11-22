"""DBOS durability: queued composition, one review per head, and recovery after a real process crash."""

from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import subprocess
import sys
from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest
from dbos import DBOS

from autoreviewer.github import Repo
from autoreviewer.review import durable, job
from autoreviewer.review.checkpoints import TurnCheckpoints
from autoreviewer.review.memory import ReviewMemory, conversation_id
from autoreviewer.review.posting import Publisher
from autoreviewer.review.schemas import Finding
from autoreviewer.web.routing import PullRequestUpdated, ThreadReply
from tests.review_fakes import BOT, FakeGitHub, GitRepo, finding, git, job_deps, make_repo
from tests.test_conversation import deps, scripted
from tests.test_withdrawals import reviewed

REPO_ = Repo(1, "mpfaffenberger", "demo")

pytestmark = pytest.mark.anyio

HEAD = "a" * 40
PULL = PullRequestUpdated(installation_id=1, owner="mpfaffenberger", repo="demo", number=7, head_sha=HEAD)


def pull_for(repo: GitRepo) -> PullRequestUpdated:
    return PullRequestUpdated(installation_id=1, owner="mpfaffenberger", repo="demo", number=7, head_sha=repo.head)


CRASH_SCRIPT = Path(__file__).with_name("durable_crash_script.py")

Launch = Callable[[list[dict[str, Any]]], Awaitable[FakeGitHub]]


@pytest.fixture
async def launch(tmp_path: Path) -> AsyncIterator[Launch]:
    async def start(findings: list[dict[str, Any]]) -> FakeGitHub:
        github = FakeGitHub(make_repo(tmp_path / "origin"))
        durable.configure(tmp_path / "dbos.sqlite")
        await durable.launch(job_deps(github, tmp_path, findings), concurrency=2)
        return github

    yield start
    durable.shutdown()


async def test_each_reply_is_answered_once_on_the_answer_queue(tmp_path: Path) -> None:
    """From the bot's review of #37: duplicate deliveries join one answer; a second reply gets its own."""
    github = FakeGitHub(make_repo(tmp_path / "origin"))
    await Publisher(github, REPO_, 7, head_sha=github.repo.head, author=BOT, block_at=1)(
        Finding.model_validate(finding(5))
    )
    first, second = github.say("alice", 1, "why?"), github.say("alice", 1, "and?")
    conversation_deps = deps(github, github.repo, tmp_path, scripted("holds"))
    durable.configure(tmp_path / "dbos.sqlite")
    await durable.launch(job_deps(github, tmp_path, []), conversation_deps, concurrency=1)
    try:
        replies = [ThreadReply(1, "mpfaffenberger", "demo", 7, comment, 1) for comment in (first, first, second)]
        handles = [await durable.enqueue_answer(reply) for reply in replies]
        assert [await handle.get_result() for handle in handles] == ["answered", "answered", "answered"]
    finally:
        durable.shutdown()
    assert len({handle.workflow_id for handle in handles}) == 2
    assert [c["author"] for c in github.review_comments].count(BOT) == 3  # the finding and two answers


async def test_a_withdrawal_reconsiders_the_verdict(tmp_path: Path) -> None:
    """From the bot's review of #38: the answer workflow's second step, after a withdrawal."""
    github = await reviewed(make_repo(tmp_path / "origin"), Finding.model_validate(finding(5)))
    comment = github.say("alice", 1, "That can't happen: the caller checks.")
    conversation_deps = deps(github, github.repo, tmp_path, scripted("withdrawn"))  # the checker agrees
    durable.configure(tmp_path / "dbos.sqlite")
    await durable.launch(job_deps(github, tmp_path, []), conversation_deps, concurrency=1)
    try:
        reply = ThreadReply(1, "mpfaffenberger", "demo", 7, comment, 1)
        assert await (await durable.enqueue_answer(reply)).get_result() == "withdrawn"
    finally:
        durable.shutdown()
    assert [r["event"] for r in github.reviews] == ["REQUEST_CHANGES", "APPROVE"]


async def test_a_queued_review_runs_all_three_phases(launch: Launch) -> None:
    github = await launch([finding(5)])

    result = await (await durable.enqueue(pull_for(github.repo))).get_result()

    assert result.conclusion == "failure"
    assert [c["status"] for c in github.checks] == ["completed"]
    assert github.checks[0]["title"] == "1 blocking issue(s)"
    assert [r["event"] for r in github.reviews] == ["REQUEST_CHANGES"]


async def test_duplicate_events_for_one_head_produce_one_review(launch: Launch) -> None:
    github = await launch([finding(5)])

    handles = [await durable.enqueue(pull_for(github.repo)) for _ in range(3)]  # e.g. GitHub redeliveries
    for handle in handles:
        await handle.get_result()

    assert len({handle.workflow_id for handle in handles}) == 1
    assert len(github.checks) == 1 and len(github.reviews) == 1


async def test_enqueue_review_says_whether_the_head_was_already_reviewed(launch: Launch) -> None:
    github = await launch([])
    pull = pull_for(github.repo)
    assert await durable.enqueue_review(pull) is False  # queued: a new review runs
    await (await durable.enqueue(pull)).get_result()
    assert await durable.enqueue_review(pull) is True  # finished: asking again runs nothing
    assert len(github.reviews) == 1


async def test_only_an_explicit_request_retries_a_crashed_review(launch: Launch) -> None:
    github = await launch([])
    github.fail_start_check = True
    pull = pull_for(github.repo)
    handle = await durable.enqueue(pull)
    with pytest.raises(Exception, match="maximum of 3 retries"):
        await handle.get_result()
    attempts = github.check_attempts
    assert await durable.enqueue_review(pull) is False  # e.g. a redelivered push: left alone
    await asyncio.sleep(3)  # long enough for the queue (polled about once a second) to start a retry
    assert (await handle.get_status()).status == "ERROR" and github.check_attempts == attempts

    github.fail_start_check = False
    assert await durable.enqueue_review(pull, comment_id=99) is False  # "@bot review": runs again
    for _ in range(200):
        if github.reviews:
            break
        await asyncio.sleep(0.05)
    assert [r["event"] for r in github.reviews] == ["APPROVE"]


async def test_an_explicit_request_starts_fresh_after_cancellation(launch: Launch) -> None:
    github = await launch([])
    github.check_update_delay = 0
    pull = pull_for(github.repo)
    handle = await durable.enqueue(pull)
    await DBOS.cancel_workflow_async(handle.workflow_id)
    assert (await handle.get_status()).status == "CANCELLED"
    await durable.enqueue_review(pull, comment_id=99)
    retry = await DBOS.retrieve_workflow_async(f"{handle.workflow_id}:attempt:2")
    assert (await retry.get_status()).queue_name == durable.QUEUE
    await retry.get_result()
    assert (await handle.get_status()).status == "CANCELLED"
    assert [r["event"] for r in github.reviews] == ["APPROVE"] and len(github.checks) == 1


async def test_a_request_after_timeout_gets_a_new_check_and_clock(
    launch: Launch, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    github = await launch([])
    pull = pull_for(github.repo)
    clock = durable.time
    with monkeypatch.context() as patch:
        patch.setattr(durable, "time", lambda: clock() + 601)
        original = await durable.enqueue(pull)
        assert (await original.get_result()).title == "Review timed out"
    assert await durable.enqueue_review(pull) is True  # a redelivered push: finished, left alone

    # Make the persisted start genuinely old: inheriting step 1 would immediately time out again.
    with sqlite3.connect(tmp_path / "dbos.sqlite") as db:
        db.execute(
            "UPDATE operation_outputs SET started_at_epoch_ms = started_at_epoch_ms - 601000 "
            "WHERE workflow_uuid = ? AND function_id = 1",
            (original.workflow_id,),
        )
    assert not await durable.enqueue_review(pull, comment_id=99)
    retry = await DBOS.retrieve_workflow_async(f"{original.workflow_id}:attempt:2")
    await retry.get_result()
    assert [r["event"] for r in github.reviews] == ["APPROVE"]
    assert len(github.checks) == 2 and github.checks[0]["title"] == "Review timed out"


async def test_concurrent_comments_and_pushes_share_attempts_and_remember_deliveries(
    launch: Launch, monkeypatch: pytest.MonkeyPatch
) -> None:
    github = await launch([])
    pull = pull_for(github.repo)
    base = durable.workflow_id(pull)
    gate = asyncio.Event()
    start = FakeGitHub.start_check

    async def gated_start(self: FakeGitHub, repo: Repo, head_sha: str) -> int:
        await gate.wait()
        return await start(self, repo, head_sha)

    monkeypatch.setattr(FakeGitHub, "start_check", gated_start)
    for number, ids in enumerate(((10, 11, 10), (20, 21, 20), (30,)), start=1):
        gate.clear()
        requests = [durable.enqueue_review(pull, comment_id=i) for i in ids]
        if number == 1:
            requests.append(durable.enqueue_review(pull))
        try:
            assert not any(await asyncio.gather(*requests))
            assert not await durable.enqueue_review(pull)  # push joins the active retry, too
        finally:
            gate.set()
        attempt = base if number == 1 else f"{base}:attempt:{number}"
        await (await DBOS.retrieve_workflow_async(attempt)).get_result()
        for comment_id in ids:
            assert await durable.enqueue_review(pull, comment_id=comment_id)
        assert len(github.checks) == number
    moved = PullRequestUpdated(pull.installation_id, pull.owner, pull.repo, pull.number, "b" * 40)
    assert await durable.enqueue_review(moved, comment_id=10)
    assert await DBOS.get_workflow_status_async(durable.workflow_id(moved)) is None


@pytest.mark.parametrize(
    "read",
    [AsyncMock(side_effect=sqlite3.OperationalError("database is locked")), AsyncMock(return_value=[])],
    ids=["read-fails", "no-row"],
)
async def test_an_unreadable_start_time_does_not_strand_the_check(
    launch: Launch, monkeypatch: pytest.MonkeyPatch, read: AsyncMock
) -> None:
    """From the bot's review of #61: raising here skipped `_finish_check`, leaving the check in progress for good."""
    github = await launch([])
    monkeypatch.setattr(DBOS, "list_workflow_steps_async", read)
    result = await (await durable.enqueue(pull_for(github.repo))).get_result()
    assert result.conclusion == "success"
    assert [c["status"] for c in github.checks] == ["completed"]
    assert [r["event"] for r in github.reviews] == ["APPROVE"]


async def test_a_retry_forked_by_the_previous_version_is_joined(
    launch: Launch, monkeypatch: pytest.MonkeyPatch
) -> None:
    """From the bot's review of #61: at deploy time a fork from the old retry path may still be reviewing this
    head. A push or a request must join it, not start a second review beside it."""
    github = await launch([])
    pull = pull_for(github.repo)
    base = durable.workflow_id(pull)
    await (await durable.enqueue(pull)).get_result()
    entered, gate, review = asyncio.Event(), asyncio.Event(), job.review_and_post

    async def gated(
        deps: job.ReviewDeps,
        pull: PullRequestUpdated,
        check_id: int | None = None,
        *,
        remaining_s: float | None = None,
        attempt: str | None = None,
    ) -> job.CheckResult:
        entered.set()
        await gate.wait()
        return await review(deps, pull, check_id, remaining_s=remaining_s, attempt=attempt)

    monkeypatch.setattr(job, "review_and_post", gated)
    # Step 2 is the review itself: the old retry path forked there, keeping step 1's check run.
    fork = await DBOS.fork_workflow_async(base, 2, queue_name=durable.QUEUE)
    try:
        await asyncio.wait_for(entered.wait(), 30)
        assert await durable.enqueue_review(pull) is False  # running: joined
        assert await durable.enqueue_review(pull, comment_id=5) is False
        assert await DBOS.get_workflow_status_async(f"{base}:attempt:2") is None
    finally:
        gate.set()
    await fork.get_result()
    assert len(github.checks) == 1  # the fork kept the original check run


async def test_a_push_after_a_second_attempt_starts_no_third(launch: Launch) -> None:
    github = await launch([])
    pull = pull_for(github.repo)
    base = durable.workflow_id(pull)
    await (await durable.enqueue(pull)).get_result()
    assert await durable.enqueue_review(pull, comment_id=5) is False
    await (await DBOS.retrieve_workflow_async(f"{base}:attempt:2")).get_result()
    assert await durable.enqueue_review(pull) is True  # the latest attempt finished: nothing new runs
    assert await DBOS.get_workflow_status_async(f"{base}:attempt:3") is None and len(github.checks) == 2


def test_workflow_ids_are_per_head_and_case_insensitive() -> None:
    other_head = PullRequestUpdated(1, "mpfaffenberger", "demo", 7, "b" * 40)
    shouted = PullRequestUpdated(1, "MPFAFFENBERGER", "Demo", 7, HEAD)
    assert durable.workflow_id(PULL) != durable.workflow_id(other_head)
    assert durable.workflow_id(PULL) == durable.workflow_id(shouted)


def run_crash_script(directory: Path, mode: str) -> subprocess.CompletedProcess[str]:
    """One process of a crash scenario (see durable_crash_script.py); its state lives in `directory`."""
    return subprocess.run(
        [sys.executable, str(CRASH_SCRIPT), str(directory), mode],
        env={**os.environ, "PYTHONPATH": str(Path(__file__).parents[1])},
        capture_output=True,
        text=True,
        timeout=120,
    )


@pytest.mark.parametrize("expired", [False, True], ids=["within-budget", "after-deadline"])
def test_a_review_interrupted_by_a_crash_finishes_after_restart_exactly_once(tmp_path: Path, expired: bool) -> None:
    """Process 1 dies (os._exit) inside the review; process 2 recovers it. One check run, one review."""
    first = run_crash_script(tmp_path, "enqueue-and-crash")
    assert first.returncode == 9, first.stderr[-2000:]
    assert json.loads((tmp_path / "github.json").read_text())["reviews"] == []  # died before posting

    second = run_crash_script(tmp_path, "recover-after-deadline" if expired else "recover")
    assert second.returncode == 0, second.stderr[-2000:]

    state = json.loads((tmp_path / "github.json").read_text())
    assert [c["status"] for c in state["checks"]] == ["completed"], "the check run must be created once and finished"
    if expired:
        assert state["checks"][0]["title"] == "Review timed out"
        assert state["reviews"] == []
    else:
        assert [r["event"] for r in state["reviews"]] == ["REQUEST_CHANGES"]


def test_a_review_crashed_between_turns_resumes_at_its_last_completed_turn(tmp_path: Path) -> None:
    """Issue #60's acceptance test. Process 1's lens reports a finding (turn 1), reads the change (turn 2), and
    dies in turn 3 once that finding is verified and posted; process 2 recovers the review on the same DB and
    memory. No completed turn runs again, the finished verdict isn't redone, and one review is posted."""
    first = run_crash_script(tmp_path, "crash-between-turns")
    assert first.returncode == 9, first.stderr[-2000:]
    head = git(tmp_path / "origin", "rev-parse", "HEAD")
    memory = ReviewMemory(tmp_path / "memory").directory(conversation_id(REPO_, 7))
    checkpoints = TurnCheckpoints.attempt_directory(memory, durable.workflow_id(pull_for(GitRepo(tmp_path, "", head))))
    crashed = json.loads((tmp_path / "github.json").read_text())
    assert crashed["reviews"] == [] and len(crashed["review_comments"]) == 1  # posted the finding, not the verdict
    assert checkpoints.is_dir()

    second = run_crash_script(tmp_path, "recover")
    assert second.returncode == 0, second.stderr[-2000:]

    calls = (tmp_path / "calls.log").read_text().splitlines()
    # Turns 1 and 2 finished before the crash: never sent again. Turn 3 was in flight: sent once more.
    assert [call for call in calls if call.startswith("lens:")] == ["lens:1", "lens:2", "lens:3", "lens:3"]
    # Its verdict was checkpointed: neither the replayed report nor the final report verifies it again.
    assert [call for call in calls if call.startswith("verifier:")] == ["verifier:stats.py:5"]

    state = json.loads((tmp_path / "github.json").read_text())
    assert [(c["status"], c["conclusion"]) for c in state["checks"]] == [("completed", "failure")]
    assert [r["event"] for r in state["reviews"]] == ["REQUEST_CHANGES"]
    inline = [c["comment"] for c in state["review_comments"]] + [c for r in state["reviews"] for c in r["comments"]]
    assert [(c["path"], c["line"]) for c in inline if finding()["title"] in c["body"]] == [("stats.py", 5)]
    assert not checkpoints.exists()  # the attempt has its verdict: nothing resumes it again


def test_old_sqlite_is_refused_with_a_clear_message(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(durable.sqlite3, "sqlite_version_info", (3, 40, 1))
    monkeypatch.setattr(durable.sqlite3, "sqlite_version", "3.40.1")
    with pytest.raises(RuntimeError, match=r"SQLite 3\.40\.1 is too old for DBOS"):
        durable.configure(tmp_path / "dbos.sqlite")
