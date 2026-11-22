"""Exercise overflow through the durable queue, including a waiting local-only backlog."""

import asyncio
from collections import Counter
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import replace
from pathlib import Path

import pytest
from dbos import DBOS

from autoreviewer.review import durable, job
from autoreviewer.web.routing import PullRequestUpdated
from tests.review_fakes import FakeGitHub, job_deps, make_repo


@pytest.mark.anyio
async def test_overflow_with_local_backlog(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    github = FakeGitHub(make_repo(tmp_path / "origin"))
    primary = job_deps(github, tmp_path, [])
    overflow = replace(primary, hard_deadline_s=60)
    allowed = PullRequestUpdated(1, "Owner", "Allowed", 1, "a" * 40)
    local = replace(allowed, repo="allowed-fork")
    entered = {n: asyncio.Event() for n in range(1, 6)}
    release = {n: asyncio.Event() for n in entered}
    phases: dict[int, list[str]] = {n: [] for n in entered}
    active: Counter[str] = Counter()
    result = job.CheckResult("Done", "Reviewed", "success")

    async def start(deps: job.ReviewDeps, pull: PullRequestUpdated) -> int:
        assert deps is primary
        phases[pull.number].append("start")
        return 100 + pull.number

    budgets: dict[int, float] = {}
    waiting = asyncio.Event()
    claim = durable._ReviewSlots.claim

    @asynccontextmanager
    async def observed_claim(slots: durable._ReviewSlots, pull: PullRequestUpdated) -> AsyncIterator[job.ReviewDeps]:
        if pull.number == 2:
            waiting.set()
        async with claim(slots, pull) as deps:
            yield deps

    async def review(
        deps: job.ReviewDeps, pull: PullRequestUpdated, check_id: int, *, remaining_s: float, attempt: str | None
    ) -> job.CheckResult:
        assert attempt and attempt == DBOS.workflow_id  # each attempt checkpoints its turns under its own id
        budgets[pull.number] = remaining_s
        assert check_id == 100 + pull.number
        lane = "primary" if deps is primary else "overflow"
        phases[pull.number].append(lane)
        active[lane] += 1
        if lane == "primary":
            assert active[lane] == 1
        entered[pull.number].set()
        await release[pull.number].wait()
        active[lane] -= 1
        return result

    async def finish(deps: job.ReviewDeps, pull: PullRequestUpdated, check_id: int, outcome: job.CheckResult) -> None:
        assert deps is primary and check_id == 100 + pull.number and outcome == result
        phases[pull.number].append("finish")

    monkeypatch.setattr(durable._ReviewSlots, "claim", observed_claim)
    monkeypatch.setattr(job, "start_check", start)
    monkeypatch.setattr(job, "review_and_post", review)
    monkeypatch.setattr(job, "finish_check", finish)
    durable.configure(tmp_path / "dbos.sqlite")
    await durable.launch(primary, concurrency=1, overflow_deps=overflow, overflow_repos=frozenset({"owner/allowed"}))
    try:
        async with asyncio.timeout(30):
            first = await durable.enqueue(allowed)
            await entered[1].wait()  # eligible but primary is free
            local_head = replace(local, number=2)
            assert not await durable.enqueue_review(local_head)
            local_id = durable.workflow_id(local_head)
            local_handle = await DBOS.retrieve_workflow_async(local_id)
            assert (await local_handle.get_status()).queue_name == durable.LOCAL_QUEUE
            await waiting.wait()  # check started, but the only primary slot is occupied
            rest = [local_handle, await durable.enqueue(replace(local, number=3))]
            for _ in range(2):  # redelivery must join the same workflow
                rest.append(await durable.enqueue(replace(allowed, number=4)))
            await entered[4].wait()  # the two local-only heads did not monopolize admission
            rest.append(await durable.enqueue(replace(allowed, number=5)))
            await entered[5].wait()
            assert active == {"primary": 1, "overflow": 2}
            assert 0 < budgets[4] <= overflow.hard_deadline_s
            # Advance the deadline clock while #2 waits for primary, not while #3 waits for dequeue.
            clock = durable.time
            with monkeypatch.context() as patch:
                patch.setattr(durable, "time", lambda: clock() + job.MAX_REVIEW_S + 1)
                release[1].set()
                await entered[2].wait()
            assert budgets[2] == 0
            for event in release.values():
                event.set()
            for handle in (first, *rest):
                assert await handle.get_result() == result
            assert budgets[3] > 0
            assert not await durable.enqueue_review(local_head, comment_id=42)
            retry = await DBOS.retrieve_workflow_async(f"{local_id}:attempt:2")
            assert (await retry.get_status()).queue_name == durable.LOCAL_QUEUE
            assert await retry.get_result() == result
            assert budgets[2] > 0  # fresh attempt, fresh clock
    finally:
        for event in release.values():
            event.set()
        durable.shutdown()
    for n, steps in phases.items():
        assert steps == ["start", "primary" if n < 4 else "overflow", "finish"] * (2 if n == 2 else 1)


@pytest.mark.anyio
async def test_cancelled_overflow_releases_only_its_slot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    primary = job_deps(FakeGitHub(make_repo(tmp_path / "origin")), tmp_path, [])
    overflow = replace(primary)
    monkeypatch.setattr(durable, "_deps", primary)
    slots = durable._ReviewSlots(2, overflow, frozenset({"*"}))
    pull = PullRequestUpdated(1, "Owner", "Allowed", 1, "a" * 40)
    claimed = asyncio.Event()

    async def hold_overflow() -> None:
        async with slots.claim(pull) as deps:
            assert deps is overflow
            claimed.set()
            await asyncio.Event().wait()

    async with asyncio.timeout(5), slots.claim(pull) as first, slots.claim(pull) as second:
        assert first is second is primary
        task = asyncio.create_task(hold_overflow())
        await claimed.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        async with slots.claim(pull) as reused:
            assert reused is overflow
