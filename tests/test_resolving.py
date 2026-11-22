"""The bot resolves the threads of findings the code has moved past (Mike, Nov 19), and only those."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from pathlib import Path

import pytest
from pydantic_ai.models.function import FunctionModel

from autoreviewer.review import job
from autoreviewer.review.agents import correctness_agent, lens_pair, verifier_agent
from autoreviewer.review.job import Lens, review_and_post
from autoreviewer.review.lint import LintError
from autoreviewer.review.memory import ReviewMemory
from autoreviewer.review.pipeline import Outcome
from autoreviewer.review.posting import Publisher, settle_fixed
from autoreviewer.review.schemas import Finding
from autoreviewer.web.routing import PullRequestUpdated
from tests.review_fakes import (
    BOT,
    FakeGitHub,
    GitRepo,
    add_commit,
    finding,
    job_deps,
    lens_agents,
    lens_model,
    never_answers,
    verifier_model,
)
from tests.test_withdrawals import REPO, blocking, reviewed

pytestmark = pytest.mark.anyio

NEXT_HEAD = "c" * 40


async def pushed_past(repo: GitRepo, *findings: Finding) -> FakeGitHub:
    """A PR whose findings were posted on one head, and whose next push changed the code under every one."""
    github = await reviewed(repo, *findings)
    for comment in github.review_comments:
        comment["outdated"] = True
    return github


CHECKED = Outcome(covered={("correctness", "stats.py")})
"""A review that checked stats.py for correctness in full and reported nothing."""


async def resolve(github: FakeGitHub, outcome: Outcome = CHECKED) -> list[int]:
    return await settle_fixed(github, REPO, 7, head_sha=NEXT_HEAD, author=BOT, outcome=outcome)


async def test_a_finding_the_review_found_nothing_like_is_resolved_with_a_note(repo: GitRepo) -> None:
    github = await pushed_past(repo, blocking(5), blocking(9, title="Division by zero on an empty list"))
    assert await resolve(github) == [1, 2]
    assert github.settled_threads == {"T1", "T2"}
    notes = [c["comment"].body for c in github.review_comments if c.get("in_reply_to") is not None]
    note = (
        f"The code here changed, and the review of `{NEXT_HEAD[:7]}` found nothing like this in the file: "
        "marking it resolved."
    )
    assert notes == [note] * 2


@pytest.mark.parametrize(
    "outcome",
    [
        replace(CHECKED, inline=[blocking(7)]),  # found again, on a line that moved
        replace(CHECKED, inline=[blocking(7, title="Mean divides by the wrong count")]),  # reworded
        replace(CHECKED, general=[blocking(30, title="Division by zero on an empty list")]),  # another, same file
        replace(CHECKED, unverified=[(blocking(7), "checker timed out")]),  # not yet confirmed
        replace(CHECKED, rejected=[(blocking(7), "not a bug")]),  # re-reported, then rejected
        replace(CHECKED, low_confidence=[blocking(7)]),  # re-reported as a hunch (the bot's review of #42)
        Outcome(),  # correctness wasn't checked in full: failed, cut short (the bot's review of #42)
        Outcome(covered={("security", "stats.py")}),
        Outcome(covered={("correctness", "notes.py")}),  # a scoped lens that didn't re-read stats.py (#42's review)
        replace(CHECKED, merged=[blocking(7)]),  # folded into another lens's finding on its line (#42's review)
    ],
)
async def test_a_finding_that_may_still_be_there_stays_open(repo: GitRepo, outcome: Outcome) -> None:
    github = await pushed_past(repo, blocking(5))
    assert await resolve(github, outcome) == []
    assert github.settled_threads == set()


@pytest.mark.parametrize(
    "reported",
    [
        blocking(5, kind="security"),  # another kind, same place
        Finding.model_validate({**finding(5), "path": "other.py"}),  # same kind, another file
    ],
)
async def test_only_the_same_kind_in_the_same_file_keeps_it_open(repo: GitRepo, reported: Finding) -> None:
    github = await pushed_past(repo, blocking(5))
    assert await resolve(github, replace(CHECKED, inline=[reported])) == [1]


async def test_a_comment_still_on_the_heads_code_stays_open(repo: GitRepo) -> None:
    """Not outdated: the code under it didn't change, so the push didn't fix it, whatever this review found."""
    github = await reviewed(repo, blocking(5))
    assert await resolve(github) == []


@pytest.mark.parametrize("done", ["resolved_threads", "settled_threads", "hidden_threads"])
async def test_a_thread_already_done_gets_no_second_note(repo: GitRepo, done: str) -> None:
    """Resolved by a person, settled by an earlier review, or hidden by a moderator: left alone, no note."""
    github = await pushed_past(repo, blocking(5))
    getattr(github, done).add("T1")
    assert await resolve(github) == []
    assert len(github.review_comments) == 1


@pytest.mark.parametrize("threads_fail", [False, True])
async def test_the_bots_review_resolves_them_and_never_fails_for_it(
    repo: GitRepo, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, threads_fail: bool
) -> None:
    """The job resolves after its verdict; if GitHub's threads can't be read, the review stands all the same."""
    github = FakeGitHub(repo)
    await Publisher(github, REPO, 7, head_sha="a" * 40, author=BOT, block_at=1)(blocking(5))  # an earlier head
    github.review_comments[0]["outdated"] = True  # the push changed the code under it
    if threads_fail:

        async def unreadable(*_: object) -> None:
            raise RuntimeError("502 Bad Gateway")

        monkeypatch.setattr(github, "review_threads", unreadable)
    pull = PullRequestUpdated(1, "mpfaffenberger", "demo", 7, repo.head)
    result = await review_and_post(job_deps(github, tmp_path, []), pull)
    assert (result.conclusion, github.reviews[-1]["event"]) == ("success", "APPROVE")
    assert github.settled_threads == (set() if threads_fail else {"T1"})


@pytest.mark.parametrize("cut_off", ["a lens before its report", "an advisory candidate, dropped unjudged"])
async def test_a_review_the_time_limit_cut_off_resolves_nothing_it_did_not_finish(
    repo: GitRepo, tmp_path: Path, cut_off: str
) -> None:
    """From the bot's review of #65: a lens that never reported didn't read the file in full, and a candidate
    dropped unjudged may be the earlier finding found again. Neither makes that finding "fixed"."""
    kind = "correctness" if cut_off == "a lens before its report" else "dry"
    github = FakeGitHub(repo)
    await Publisher(github, REPO, 7, head_sha="a" * 40, author=BOT, block_at=1)(blocking(5, kind=kind))
    github.review_comments[0]["outdated"] = True  # an earlier head's finding; the push changed the code under it
    stuck, working = FunctionModel(never_answers), verifier_model(lambda f: True)
    deps = replace(job_deps(github, tmp_path, []), hard_deadline_s=1.0)  # checking stops waiting at 0.9s
    if kind == "correctness":
        deps = replace(deps, agents=lambda ws: (correctness_agent(stuck, ws), verifier_agent(working, ws)))
    else:
        dry = Lens("dry", lambda ws: lens_pair("dry", lens_model([finding(5)]), stuck, ws), whole_pr=False)
        deps = replace(deps, lenses=(dry,))
    async with asyncio.timeout(5):
        result = await review_and_post(deps, PullRequestUpdated(1, "mpfaffenberger", "demo", 7, repo.head))
    assert result.title == ("Partially reviewed" if kind == "correctness" else "No blocking issues")
    assert github.settled_threads == set() and len(github.review_comments) == 1  # no thread settled, no note


ELSEWHERE = Finding.model_validate({**finding(3), "kind": "dry", "path": "notes.py"}).model_dump()


@pytest.mark.parametrize(
    ("lens", "resolved"),
    [
        (Lens("dry", lens_agents([], kind="dry")), {"T1"}),  # finished, found nothing like it
        (Lens("dry", lens_agents([], kind="dry", crash=True)), set()),  # failed: it can't say (#42's review)
        (Lens("dry", lens_agents([ELSEWHERE], kind="dry"), max_posted=1), set()),  # at its cap: may have dropped it
        (Lens("dry", lens_agents([ELSEWHERE], kind="dry"), max_posted=2), {"T1"}),  # under its cap
    ],
)
async def test_the_review_resolves_only_what_it_checked_in_full(
    repo: GitRepo, tmp_path: Path, lens: Lens, resolved: set[str]
) -> None:
    github = FakeGitHub(repo)
    await Publisher(github, REPO, 7, head_sha="a" * 40, author=BOT, block_at=1)(blocking(5, kind="dry"))
    github.review_comments[0]["outdated"] = True
    deps = replace(job_deps(github, tmp_path), lenses=(lens,))
    await review_and_post(deps, PullRequestUpdated(1, "mpfaffenberger", "demo", 7, repo.head))
    assert github.settled_threads == resolved


@pytest.mark.parametrize("ruff_fails", [False, True])
async def test_a_linters_finding_is_resolved_only_when_the_linter_ran(
    repo: GitRepo, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ruff_fails: bool
) -> None:
    github = FakeGitHub(repo)
    await Publisher(github, REPO, 7, head_sha="a" * 40, author=BOT, block_at=1)(blocking(5, kind="types"))
    github.review_comments[0]["outdated"] = True
    if ruff_fails:

        async def broken(*_: object) -> list[Finding]:
            raise LintError("ruff: no such file")

        monkeypatch.setattr(job, "lint_findings", broken)
    await review_and_post(job_deps(github, tmp_path), PullRequestUpdated(1, "mpfaffenberger", "demo", 7, repo.head))
    assert github.settled_threads == (set() if ruff_fails else {"T1"})


async def test_a_failed_correctness_lens_resolves_none_of_its_threads(repo: GitRepo, tmp_path: Path) -> None:
    """A lens that failed comes back as an empty outcome: that's "couldn't say", not "found nothing"."""
    github = FakeGitHub(repo)
    await Publisher(github, REPO, 7, head_sha="a" * 40, author=BOT, block_at=1)(blocking(5))
    github.review_comments[0]["outdated"] = True
    deps = replace(job_deps(github, tmp_path), agents=lens_agents([], crash=True))
    await review_and_post(deps, PullRequestUpdated(1, "mpfaffenberger", "demo", 7, repo.head))
    assert github.settled_threads == set()


async def test_a_failed_resolve_leaves_no_note_behind(repo: GitRepo) -> None:
    """From the bot's review of #42: resolving comes first, so a note never claims what didn't happen."""
    github = await pushed_past(repo, blocking(5))
    github.fail_threads = True
    with pytest.raises(RuntimeError):
        await resolve(github)
    assert len(github.review_comments) == 1


@pytest.mark.parametrize("security_priority", [2, 0])  # less urgent: dropped; more urgent: replaces
def test_a_finding_folded_into_another_lens_is_still_reported(security_priority: int) -> None:
    """From the bot's review of #42: correctness and security on one line post once, but both were reported."""
    outcome = Outcome(inline=[blocking(5)])
    outcome.absorb(Outcome(inline=[blocking(5, kind="security", priority=security_priority)]))
    assert sorted((f.kind, f.path) for f in outcome.reported) == [("correctness", "stats.py"), ("security", "stats.py")]


async def test_a_review_cut_short_resolves_nothing_its_lens_didnt_finish(repo: GitRepo, tmp_path: Path) -> None:
    """From the bot's review of #42: the deadline takes the wrapped-up lens's kinds out of what was checked."""
    github = await outdated(repo)
    deps = replace(job_deps(github, tmp_path), soft_deadline_s=0)
    await review_and_post(deps, PullRequestUpdated(1, "mpfaffenberger", "demo", 7, repo.head))
    assert github.settled_threads == set()


@pytest.mark.parametrize(("lint_cap", "resolved"), [(30, {"T1"}), (0, set())])
async def test_the_linter_at_its_cap_resolves_nothing(
    repo: GitRepo, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, lint_cap: int, resolved: set[str]
) -> None:
    """From the bot's review of #42: like a capped lens, a capped linter may have dropped a finding."""
    github = await outdated(repo, "types")
    monkeypatch.setattr(job, "MAX_LINT_FINDINGS", lint_cap)
    await review_and_post(job_deps(github, tmp_path), PullRequestUpdated(1, "mpfaffenberger", "demo", 7, repo.head))
    assert github.settled_threads == resolved


async def outdated(repo: GitRepo, kind: str = "correctness") -> FakeGitHub:
    """A PR with one finding of `kind` posted on an earlier head, whose code the latest push changed."""
    github = FakeGitHub(repo)
    await Publisher(github, REPO, 7, head_sha="a" * 40, author=BOT, block_at=1)(blocking(5, kind=kind))
    github.review_comments[0]["outdated"] = True
    return github


async def test_a_failed_correctness_lens_resolves_none_of_the_linters_threads(repo: GitRepo, tmp_path: Path) -> None:
    """From the bot's review of #42: the lint findings reach the outcome through the correctness lens."""
    github = await outdated(repo, "types")
    deps = replace(job_deps(github, tmp_path), agents=lens_agents([], crash=True))
    await review_and_post(deps, PullRequestUpdated(1, "mpfaffenberger", "demo", 7, repo.head))
    assert github.settled_threads == set()


@pytest.mark.parametrize(("cap", "resolved"), [(25, {"T1"}), (1, set())])
async def test_the_correctness_lens_at_its_cap_resolves_nothing(
    repo: GitRepo, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cap: int, resolved: set[str]
) -> None:
    """From the bot's review of #42: correctness keeps at most MAX_FINDINGS too."""
    github = await outdated(repo)
    monkeypatch.setattr(job, "MAX_FINDINGS", cap)
    deps = job_deps(github, tmp_path, [finding(3, path="notes.py")])
    await review_and_post(deps, PullRequestUpdated(1, "mpfaffenberger", "demo", 7, repo.head))
    assert github.settled_threads == resolved


@pytest.mark.parametrize(("whole_pr", "resolved"), [(True, {"T1"}), (False, set())])
async def test_a_scoped_lens_resolves_only_in_the_files_it_reread(
    repo: GitRepo, tmp_path: Path, whole_pr: bool, resolved: set[str]
) -> None:
    """From the bot's review of #42: continuing, DRY reads only what changed since; stats.py didn't."""
    github = FakeGitHub(repo)
    deps = replace(
        job_deps(github, tmp_path),
        lenses=(Lens("dry", lens_agents([], kind="dry"), whole_pr=whole_pr),),
        memory=ReviewMemory(tmp_path / "steps"),
    )
    await review_and_post(deps, PullRequestUpdated(1, "mpfaffenberger", "demo", 7, repo.head))  # remembered
    await Publisher(github, REPO, 7, head_sha="a" * 40, author=BOT, block_at=1)(blocking(5, kind="dry"))
    github.review_comments[-1]["outdated"] = True
    head = add_commit(repo, {"notes.py": "# only this file changes\n"})
    github.head_shas = [head]
    await review_and_post(deps, PullRequestUpdated(1, "mpfaffenberger", "demo", 7, head))
    assert github.settled_threads == resolved


@pytest.mark.parametrize(("changed", "settled"), [("stats.py", {"T1"}), ("notes.py", set())])
async def test_a_continued_review_settles_correctness_only_where_it_reread(
    repo: GitRepo, tmp_path: Path, changed: str, settled: set[str]
) -> None:
    """From the bot's review of #44: continuing, correctness reads the files changed since; the rest it didn't."""
    github = FakeGitHub(repo)
    deps = replace(job_deps(github, tmp_path), memory=ReviewMemory(tmp_path / "steps"))
    await review_and_post(deps, PullRequestUpdated(1, "mpfaffenberger", "demo", 7, repo.head))  # remembered
    await Publisher(github, REPO, 7, head_sha="a" * 40, author=BOT, block_at=1)(blocking(5))
    github.review_comments[-1]["outdated"] = True
    head = add_commit(repo, {changed: "# changed\n"})
    github.head_shas = [head]
    await review_and_post(deps, PullRequestUpdated(1, "mpfaffenberger", "demo", 7, head))
    assert github.settled_threads == settled


@pytest.mark.parametrize(("kind", "settled"), [("lint", set()), ("types", {"T1"})])
async def test_a_file_whose_unused_imports_went_unchecked_settles_no_lint_finding(
    repo: GitRepo, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str, settled: set[str]
) -> None:
    """From the bot's review of #44: an unreadable merge-base skips that file's unused-import check, not its types."""

    async def base_unreadable(*_: object, unchecked: set[str] | None = None) -> list[Finding]:
        assert unchecked is not None
        unchecked.add("stats.py")
        return []

    github = await outdated(repo, kind)
    monkeypatch.setattr(job, "lint_findings", base_unreadable)
    await review_and_post(job_deps(github, tmp_path), PullRequestUpdated(1, "mpfaffenberger", "demo", 7, repo.head))
    assert github.settled_threads == settled
