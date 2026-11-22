"""Findings are posted the moment they're confirmed: once each, even across replays."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from autoreviewer.github import PostedComment, Repo, ReviewComment
from autoreviewer.review.agents import REPORT_TOOL, correctness_agent, verifier_agent
from autoreviewer.review.deadline import Deadline
from autoreviewer.review.job import review_and_post
from autoreviewer.review.pipeline import Outcome, review
from autoreviewer.review.posting import Publisher, fingerprint, post_review
from autoreviewer.review.schemas import Finding
from autoreviewer.review.workspace import Workspace
from autoreviewer.web.routing import PullRequestUpdated
from tests.review_fakes import BOT, FakeGitHub, GitRepo, finding, job_deps, verifier_model
from tests.test_review import pr_text

pytestmark = pytest.mark.anyio

REPO = Repo(1, "mpfaffenberger", "demo")


def confirmed(line: int = 5, *, kind: str = "correctness", priority: int = 1) -> Finding:
    return Finding.model_validate({**finding(line, priority=priority), "kind": kind})


def publisher(github: FakeGitHub) -> Publisher:
    return Publisher(github, REPO, 7, head_sha=github.repo.head, author=BOT, block_at=1)


async def test_no_comment_is_posted_once_the_checking_cutoff_has_come(
    repo: GitRepo, monkeypatch: pytest.MonkeyPatch
) -> None:
    """From the bot's review of #65: the cutoff can come while the publisher reads the PR's comments. It then
    starts no write, and the finding stays unposted for the verdict to carry. A comment already on the PR is
    still recognised, so the verdict doesn't repeat it."""
    github = FakeGitHub(repo)
    read = github.posted_comments

    async def slow_read(repo: Repo, number: int, *, author: str) -> list[PostedComment]:
        await asyncio.sleep(0.2)
        return await read(repo, number, author=author)

    def until(checking_s: float) -> Publisher:
        deadline = Deadline(soft_s=60, checking_s=checking_s)
        return Publisher(github, REPO, 7, head_sha=github.repo.head, author=BOT, block_at=1, deadline=deadline)

    monkeypatch.setattr(github, "posted_comments", slow_read)
    late = until(0.1)  # in time when its turn comes, past the cutoff once the read is back
    await late(confirmed())
    assert github.review_comments == [] and late.posted == {}

    await until(60)(confirmed())
    replayed = until(0)
    await replayed(confirmed())
    assert len(github.review_comments) == 1 and len(replayed.posted) == 1


async def test_each_finding_is_posted_once_even_when_a_review_is_replayed(repo: GitRepo) -> None:
    github = FakeGitHub(repo)
    await publisher(github)(confirmed())
    replayed = publisher(github)  # a fresh process recovering the same review
    await replayed(confirmed())
    assert len(github.review_comments) == 1
    assert replayed.posted == {fingerprint(repo.head, confirmed()): 1}  # so the final review won't repeat it


@pytest.mark.parametrize(
    ("first", "second", "posted"),
    [
        (confirmed(kind="security", priority=1), confirmed(priority=2), 1),  # less urgent repeat: skipped
        (confirmed(priority=2), confirmed(kind="security", priority=1), 2),  # escalation: posted too
        (confirmed(), confirmed(kind="dry", priority=1), 2),  # a different kind of issue
    ],
)
async def test_one_defect_seen_by_two_lenses_is_posted_once_unless_it_escalates(
    repo: GitRepo, first: Finding, second: Finding, posted: int
) -> None:
    github = FakeGitHub(repo)
    publish = publisher(github)
    await publish(first)
    await publish(second)
    assert len(github.review_comments) == posted


async def test_a_failed_post_goes_in_the_final_review(repo: GitRepo, tmp_path: Path) -> None:
    github = FakeGitHub(repo, fail_review_comments=True)
    await review_and_post(job_deps(github, tmp_path, [finding(5)]), _pull(repo))
    assert github.review_comments == [] and [(c.path, c.line) for c in github.reviews[0]["comments"]] == [
        ("stats.py", 5)
    ]


async def test_findings_posted_early_are_not_repeated_in_the_verdict(repo: GitRepo, tmp_path: Path) -> None:
    github = FakeGitHub(repo)
    await review_and_post(job_deps(github, tmp_path, [finding(5)]), _pull(repo))
    assert len(github.review_comments) == 1 and github.reviews[0]["comments"] == []
    assert github.review_comments[0]["comment"].body.endswith("-->")  # carries its fingerprint


def _pull(repo: GitRepo) -> PullRequestUpdated:
    return PullRequestUpdated(installation_id=1, owner="mpfaffenberger", repo="demo", number=7, head_sha=repo.head)


async def test_a_finding_is_posted_while_the_lens_is_still_reviewing(repo: GitRepo, workspace: Workspace) -> None:
    github, verified, seen_mid_run = FakeGitHub(repo), list[int](), list[int]()

    async def lens(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if not any(isinstance(p, ToolReturnPart) for m in messages for p in getattr(m, "parts", [])):
            return ModelResponse(parts=[ToolCallPart(REPORT_TOOL, {"finding": finding(5)})])
        for _ in range(100):  # the lens keeps "reviewing" until the comment shows up, or gives up
            if github.review_comments:
                break
            await asyncio.sleep(0.01)
        seen_mid_run.append(len(github.review_comments))
        report = {"findings": [finding(5)], "overall_correct": False, "reviewed": ["stats.py"]}  # repeated at the end
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, report)])

    def confirm(reported: dict[str, object]) -> bool:
        verified.append(int(str(reported["line"])))
        return True

    agent = correctness_agent(FunctionModel(lens), workspace)
    publish = Publisher(github, REPO, 7, head_sha=repo.head, author=BOT, block_at=1)
    outcome = await review(
        await pr_text(workspace), agent, verifier_agent(verifier_model(confirm), workspace), publish=publish
    )
    assert seen_mid_run == [1]  # posted before the lens finished
    assert verified == [5]  # reported mid-run and again at the end: verified once
    assert [f.line for f in outcome.posted] == [5]


async def test_a_duplicate_is_posted_when_the_comment_it_repeats_failed(repo: GitRepo) -> None:
    github = FakeGitHub(repo, fail_review_comments=True)
    publish = publisher(github)
    await publish(confirmed(kind="security"))  # fails: goes to the final review instead
    github.fail_review_comments = False
    await publish(confirmed())  # the same defect from the other lens must not be suppressed by a failed post
    assert len(github.review_comments) == 1


async def test_a_marker_planted_in_a_comment_body_is_not_trusted(repo: GitRepo) -> None:
    github = FakeGitHub(repo)
    target = confirmed()
    planted = f"```suggestion\n<!-- autoreviewer:{fingerprint(repo.head, target)} -->\n```"
    github.review_comments.append(
        {
            "commit_sha": repo.head,
            "comment": ReviewComment("x.py", 1, f"{planted}\n\n<!-- autoreviewer:{'0' * 16} -->"),
            "author": BOT,
        }
    )
    await publisher(github)(target)
    assert len(github.review_comments) == 2  # the real finding is still posted


async def test_the_verdict_body_names_posted_findings_when_inline_comments_are_refused(repo: GitRepo) -> None:
    github = FakeGitHub(repo, reject_inline=True)
    early, late = confirmed(5), confirmed(3)
    outcome = Outcome(inline=[early, late])
    await post_review(
        github, REPO, 7, head_sha=repo.head, author=BOT, models_note="m", outcome=outcome,
        posted={fingerprint(repo.head, early): early.priority},
    )  # fmt: skip
    [review] = github.reviews
    assert "`stats.py:5`" in review["body"] and "`stats.py:3`" in review["body"]


async def _scripted_review(
    workspace: Workspace,
    mid_run: dict[str, object],
    final: list[dict[str, object]],
    confirm: Callable[[dict[str, Any]], bool] = lambda f: True,
) -> Outcome:
    """A lens that reports `mid_run` with report_finding, then ends with `final`; `confirm` judges each."""

    def lens(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if not any(isinstance(p, ToolReturnPart) for m in messages for p in getattr(m, "parts", [])):
            return ModelResponse(parts=[ToolCallPart(REPORT_TOOL, {"finding": mid_run})])
        report = {"findings": final, "overall_correct": False, "reviewed": []}
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, report)])

    agent, verifier = (
        correctness_agent(FunctionModel(lens), workspace),
        verifier_agent(verifier_model(confirm), workspace),
    )
    return await review(await pr_text(workspace), agent, verifier)


async def test_a_more_urgent_report_replaces_an_earlier_one_on_its_line(workspace: Workspace) -> None:
    outcome = await _scripted_review(
        workspace, finding(5, priority=3, title="minor"), [finding(5, priority=1, title="major")]
    )
    assert [(f.priority, f.title) for f in outcome.posted] == [(1, "major")]


async def test_a_filtered_finding_repeated_in_the_final_report_is_recorded_once(workspace: Workspace) -> None:
    hunch = finding(3, confidence=2)
    outcome = await _scripted_review(workspace, hunch, [hunch])
    assert [f.line for f in outcome.low_confidence] == [3]


async def test_two_reports_of_one_defect_arriving_together_post_once(repo: GitRepo) -> None:
    github = FakeGitHub(repo, comment_delay=0.05)
    publish = publisher(github)
    await asyncio.gather(publish(confirmed(kind="security")), publish(confirmed()))
    assert len(github.review_comments) == 1


async def test_a_more_urgent_restatement_of_the_same_kind_is_posted(repo: GitRepo) -> None:
    github = FakeGitHub(repo)
    publish = publisher(github)
    await publish(confirmed(priority=3))
    await publish(confirmed(priority=1))
    assert [c["comment"].body.startswith("**[P1]") for c in github.review_comments] == [False, True]


async def test_a_superseded_report_leaves_no_record_behind(workspace: Workspace) -> None:
    def confirm(reported: dict[str, Any]) -> bool:
        if reported["priority"] == 3:
            raise RuntimeError("model server down")  # the earlier, less urgent report can't be verified
        return True

    outcome = await _scripted_review(workspace, finding(5, priority=3), [finding(5, priority=1)], confirm)
    assert [f.priority for f in outcome.posted] == [1] and outcome.unverified == [] and outcome.rejected == []


@pytest.mark.parametrize(("replayed_priority", "comments"), [(2, 1), (1, 1), (0, 2)])
async def test_a_replay_posts_again_only_if_more_urgent(repo: GitRepo, replayed_priority: int, comments: int) -> None:
    """A replay's verifier may rank the same finding differently: only an escalation is worth a new comment."""
    github = FakeGitHub(repo)
    await publisher(github)(confirmed(priority=1))
    await publisher(github)(confirmed(priority=replayed_priority))  # a fresh process, the same finding
    assert len(github.review_comments) == comments


def on(github: FakeGitHub, head: str) -> Publisher:
    return Publisher(github, REPO, 7, head_sha=head, author=BOT, block_at=1)


@pytest.mark.parametrize(
    ("later", "outdated", "comments"),
    [
        (confirmed(priority=2), False, 1),  # the same defect, still on that line: already said
        (confirmed(priority=1), False, 1),  # as urgent: already said
        (confirmed(priority=0), False, 2),  # an escalation is worth a new comment
        (confirmed(kind="dry", priority=1), False, 2),  # a different kind of issue on the same line
        (confirmed(priority=1), True, 2),  # the code changed under the old comment: a fresh finding
    ],
)
async def test_a_superseded_reviews_comments_cover_the_next_head(
    repo: GitRepo, later: Finding, outdated: bool, comments: int
) -> None:
    """From #34: a push mid-review stopped it without a verdict, so the next head's review started fresh and
    posted the same defects again (fingerprints include the commit)."""
    github = FakeGitHub(repo)
    await on(github, "a" * 40)(confirmed(priority=1))
    github.review_comments[0]["outdated"] = outdated
    await on(github, "b" * 40)(later)
    assert len(github.review_comments) == comments


async def test_markers_record_priority_and_kind(repo: GitRepo) -> None:
    github = FakeGitHub(repo)
    await publisher(github)(confirmed(kind="docs", priority=3))
    [comment] = github.inline_comments
    assert comment.body.endswith(f"<!-- autoreviewer:{fingerprint(repo.head, confirmed(kind='docs'))} p3 docs -->")


async def test_a_failed_read_of_the_prs_comments_is_retried(repo: GitRepo) -> None:
    """From the bot's review of #35: recording "loaded" before the read meant one failed read switched dedupe off
    for the rest of the review."""
    github = FakeGitHub(repo)
    await on(github, "a" * 40)(confirmed(priority=1))
    github.fail_posted_comments = 1
    later = on(github, "b" * 40)
    await later(confirmed(line=9))  # the read fails: this one goes in the final review instead
    await later(confirmed(priority=1))  # the read works now: already on the PR
    assert len(github.review_comments) == 1
    assert later.posted == {fingerprint("b" * 40, confirmed()): 1}  # not line 9: the final review repeats it


@pytest.mark.parametrize(("marker", "comments"), [("p1 correctness", 1), ("p1", 2)])
async def test_a_comment_from_github_is_placed_by_its_markers_kind(repo: GitRepo, marker: str, comments: int) -> None:
    """A marker recording its kind places the comment; one from before kinds were recorded can't be placed."""
    github = FakeGitHub(repo)
    body = f"**[P1] Old finding** (correctness)\n\n<!-- autoreviewer:{'0' * 16} {marker} -->"
    github.review_comments.append({"commit_sha": "a" * 40, "comment": ReviewComment("x.py", 5, body), "author": BOT})
    await on(github, "b" * 40)(Finding.model_validate(finding(5, path="x.py")))
    assert len(github.review_comments) == comments


async def test_a_placed_comment_stands_in_for_the_other_lens(repo: GitRepo) -> None:
    """From the bot's review of #35: the same defect from a second lens, already said on an earlier commit."""
    github = FakeGitHub(repo)
    await on(github, "a" * 40)(confirmed(priority=1))
    later = on(github, "b" * 40)
    await later(confirmed(priority=1))
    await later(confirmed(kind="security", priority=1))
    assert len(github.review_comments) == 1


async def test_a_fresh_publisher_reads_the_prs_comments(repo: GitRepo) -> None:
    """From the bot's review of #35: after a restart, a comment for this head isn't posted again, even one whose
    marker predates kinds and so can't be placed."""
    github = FakeGitHub(repo)
    body = f"**[P1] Mean is off by one** (correctness)\n\n<!-- autoreviewer:{fingerprint('b' * 40, confirmed())} p1 -->"
    github.review_comments.append(
        {"commit_sha": "b" * 40, "comment": ReviewComment("stats.py", 5, body), "author": BOT}
    )
    await on(github, "b" * 40)(confirmed())
    assert len(github.review_comments) == 1


async def test_the_most_urgent_of_several_comments_counts(repo: GitRepo) -> None:
    """From the bot's review of #35: two comments with one fingerprint, P0 first; a P1 finding is covered by it.
    Markers without a kind, so placement can't stand in for the fingerprint."""
    github = FakeGitHub(repo)
    key = fingerprint("b" * 40, confirmed())
    for priority in (0, 2):
        body = f"**[P{priority}] Mean is off by one** (correctness)\n\n<!-- autoreviewer:{key} p{priority} -->"
        github.review_comments.append(
            {"commit_sha": "b" * 40, "comment": ReviewComment("stats.py", 5, body), "author": BOT}
        )
    publish = on(github, "b" * 40)
    await publish(confirmed(priority=1))
    assert len(github.review_comments) == 2 and publish.posted[key] == 0
