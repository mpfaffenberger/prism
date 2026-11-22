"""Time limits: a lens past the soft deadline wraps up with a report, checking stops at its cutoff so a verdict
still goes out, and a cut-short review never approves."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, ToolCallPart, UserPromptPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.tools import ToolDefinition

from autoreviewer.github import PullRequestState
from autoreviewer.review import job
from autoreviewer.review.agents import REPORT_TOOL, correctness_agent, lens_pair, verifier_agent
from autoreviewer.review.deadline import (
    DEFAULT_SOFT_S,
    FIVE_MINUTES,
    FOCUS_S,
    MAX_REVIEW_S,
    ONE_MINUTE,
    POSTING_RESERVE_S,
    REPORT_ONLY_S,
    WRAP_UP,
    Deadline,
    before_cutoff,
    checking_cutoff,
    until_cutoff,
)
from autoreviewer.review.job import Lens, Reviewed, repo_of, review_and_post, review_change
from autoreviewer.review.memory import ReviewMemory, conversation_id
from autoreviewer.review.workspace import Workspace
from autoreviewer.web.routing import PullRequestUpdated
from tests.review_fakes import (
    FakeGitHub,
    GitRepo,
    LocalSource,
    agents,
    finding,
    job_deps,
    lens_agents,
    lens_model,
    never_answers,
    verifier_model,
)

pytestmark = pytest.mark.anyio


async def test_a_lens_past_the_deadline_returns_its_report(workspace: Workspace) -> None:
    """The model would explore forever; once time is up it sees no tools and is told to report."""
    tools_offered: list[int] = []

    def endless(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        tools_offered.append(len(info.function_tools))
        told = any(
            isinstance(p, UserPromptPart) and p.content == WRAP_UP
            for m in messages
            if isinstance(m, ModelRequest)
            for p in m.parts
        )
        if info.function_tools and not told:
            return ModelResponse(parts=[ToolCallPart("changed_files", {})])
        report = {"findings": [finding(5)], "overall_correct": False, "reviewed": ["stats.py"]}
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, report)])

    deadline = Deadline(soft_s=0)  # already passed
    run = await correctness_agent(FunctionModel(endless), workspace).run(
        "review", capabilities=[deadline.watch("reviewer")]
    )
    assert tools_offered == [0] and [f.line for f in run.output.findings] == [5]
    assert deadline.wrapped_up == {"reviewer"}


@pytest.mark.parametrize(
    ("budget", "focus", "stop"),
    [(DEFAULT_SOFT_S, FOCUS_S, REPORT_ONLY_S), (DEFAULT_SOFT_S / 10, FOCUS_S / 10, REPORT_ONLY_S / 10)],
    ids=["default-budget", "a-tenth-of-it"],
)
async def test_countdown_is_once_per_stage_and_final_minute_is_report_only(
    workspace: Workspace, budget: float, focus: float, stop: float
) -> None:
    deadline = Deadline(soft_s=budget)
    notices: list[list[str]] = []

    def lens(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        last = messages[-1]
        assert isinstance(last, ModelRequest)
        notices.append(
            [
                p.content
                for p in last.parts
                if isinstance(p, UserPromptPart)
                and isinstance(p.content, str)
                and p.content in (FIVE_MINUTES, ONE_MINUTE, WRAP_UP)
            ]
        )
        if len(notices) < 4:
            assert info.function_tools and not deadline.wrapped_up
            if len(notices) == 1:
                deadline.started -= budget - focus
            elif len(notices) == 3:
                deadline.started -= focus - stop
            return ModelResponse(parts=[ToolCallPart("changed_files", {})])
        assert not info.function_tools and deadline.wrapped_up == {"reviewer"}
        return ModelResponse(
            parts=[ToolCallPart(info.output_tools[0].name, {"findings": [], "overall_correct": True, "reviewed": []})]
        )

    await correctness_agent(FunctionModel(lens), workspace).run("review", capabilities=[deadline.watch("reviewer")])
    assert notices == [[], [FIVE_MINUTES], [], [ONE_MINUTE]]


async def test_tool_removal_alone_marks_the_lens_forced_to_report() -> None:
    deadline = Deadline(soft_s=DEFAULT_SOFT_S)
    capability = deadline.watch("reviewer")
    tools = [ToolDefinition(name="explore")]
    assert await capability.prepare_tools(Mock(), tools) == tools
    assert not deadline.wrapped_up
    deadline.started -= DEFAULT_SOFT_S - REPORT_ONLY_S + 1
    assert await capability.prepare_tools(Mock(), tools) == []
    assert deadline.wrapped_up == {"reviewer"}


async def test_before_the_deadline_nothing_changes(workspace: Workspace) -> None:
    offered: list[int] = []

    def lens(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        offered.append(len(info.function_tools))
        return ModelResponse(
            parts=[ToolCallPart(info.output_tools[0].name, {"findings": [], "overall_correct": True, "reviewed": []})]
        )

    deadline = Deadline(soft_s=3600)
    await correctness_agent(FunctionModel(lens), workspace).run("review", capabilities=[deadline.watch("reviewer")])
    assert offered[0] > 0 and deadline.wrapped_up == set()


def pull(repo: GitRepo) -> PullRequestUpdated:
    return PullRequestUpdated(installation_id=1, owner="mpfaffenberger", repo="demo", number=7, head_sha=repo.head)


CHECKING_S = MAX_REVIEW_S - DEFAULT_SOFT_S
"""What the default budget keeps after the lenses wrap up: a recovered job with this much left is at soft expiry."""


@pytest.mark.parametrize(
    "remaining, warning",
    [
        (CHECKING_S + FOCUS_S + 100, None),
        (CHECKING_S + FOCUS_S - 100, FIVE_MINUTES),
        (CHECKING_S + REPORT_ONLY_S - 20, ONE_MINUTE),
        (CHECKING_S - 50, WRAP_UP),
    ],
)
async def test_recovered_job_counts_elapsed_time_toward_soft_warnings(
    repo: GitRepo, tmp_path: Path, remaining: float, warning: str | None
) -> None:
    notices: list[str] = []

    def lens(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        last = messages[-1]
        assert isinstance(last, ModelRequest)
        notices.extend(
            p.content
            for p in last.parts
            if isinstance(p, UserPromptPart)
            and isinstance(p.content, str)
            and p.content in (FIVE_MINUTES, ONE_MINUTE, WRAP_UP)
        )
        return ModelResponse(
            parts=[ToolCallPart(info.output_tools[0].name, {"findings": [], "overall_correct": True, "reviewed": []})]
        )

    github = FakeGitHub(repo)
    deps = replace(
        job_deps(github, tmp_path),
        hard_deadline_s=MAX_REVIEW_S * 2,  # capped at MAX_REVIEW_S, which is what `remaining` counts down from
        soft_deadline_s=DEFAULT_SOFT_S,
        agents=lambda ws: (
            correctness_agent(FunctionModel(lens), ws),
            verifier_agent(verifier_model(lambda f: True), ws),
        ),
    )
    result = await review_and_post(deps, pull(repo), remaining_s=remaining)
    assert notices == ([warning] if warning else [])
    assert result.conclusion == ("neutral" if warning in (ONE_MINUTE, WRAP_UP) else "success")


async def test_expired_job_does_not_start(repo: GitRepo, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    github = FakeGitHub(repo)
    state = AsyncMock()
    monkeypatch.setattr(github, "get_pull", state)
    result = await review_and_post(job_deps(github, tmp_path), pull(repo), remaining_s=0)
    assert (result.title, result.conclusion) == ("Review timed out", "neutral")
    state.assert_not_awaited()


async def test_initial_state_and_progress_publisher_share_the_budget(
    repo: GitRepo, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    github = FakeGitHub(repo)

    async def hang(*_: object) -> None:
        await asyncio.Event().wait()

    state = AsyncMock(side_effect=hang)
    publisher = AsyncMock(side_effect=hang)
    monkeypatch.setattr(github, "get_pull", state)
    monkeypatch.setattr(job, "_publish", publisher)
    async with asyncio.timeout(2):
        result = await review_and_post(job_deps(github, tmp_path), pull(repo), 1, remaining_s=0.05)
    state.assert_awaited_once()
    publisher.assert_awaited_once()
    assert (result.title, result.conclusion) == ("Review timed out", "neutral")


async def test_external_cancellation_is_not_a_timeout(
    repo: GitRepo, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    github = FakeGitHub(repo)
    started = asyncio.Event()

    async def wait_for_cancel(*_: object) -> None:
        started.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(github, "get_pull", wait_for_cancel)
    async with asyncio.timeout(2):
        task = asyncio.create_task(review_and_post(job_deps(github, tmp_path), pull(repo), 1))
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert not github.reviews


async def test_posting_the_verdict_is_inside_the_cap(
    repo: GitRepo, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    github = FakeGitHub(repo)

    async def hung_post(*args: object, **kwargs: object) -> None:
        await asyncio.Event().wait()

    post = AsyncMock(side_effect=hung_post)
    monkeypatch.setattr(job, "post_review", post)
    async with asyncio.timeout(3):
        result = await review_and_post(replace(job_deps(github, tmp_path), hard_deadline_s=1), pull(repo))
    post.assert_awaited_once()
    assert (result.title, result.conclusion) == ("Review timed out", "neutral")


async def test_a_review_cut_short_never_approves(repo: GitRepo, tmp_path: Path) -> None:
    github = FakeGitHub(repo)
    deps = replace(job_deps(github, tmp_path), soft_deadline_s=0, memory=ReviewMemory(tmp_path / "steps"))
    result = await review_and_post(deps, pull(repo))
    assert [r["event"] for r in github.reviews] == ["COMMENT"] and "Partially reviewed" in github.reviews[0]["body"]
    assert (result.conclusion, result.title) == ("neutral", "Partially reviewed")
    why = "the review hit its time limit before covering the whole change"  # #43's review: the reason, said
    assert result.summary == f"Partially reviewed: {why}; not approved." and why in github.reviews[0]["body"]
    assert deps.memory is not None and deps.memory.last(conversation_id(repo_of(pull(repo)), 7)) is None


async def test_a_review_cut_short_still_requests_changes_for_what_it_found(repo: GitRepo, tmp_path: Path) -> None:
    github = FakeGitHub(repo)
    result = await review_and_post(replace(job_deps(github, tmp_path, [finding(5)]), soft_deadline_s=0), pull(repo))
    assert [r["event"] for r in github.reviews] == ["REQUEST_CHANGES"] and result.conclusion == "failure"


async def test_an_advisory_lens_running_out_of_time_doesnt_withhold_approval(repo: GitRepo, tmp_path: Path) -> None:
    deadline = Deadline(soft_s=3600)
    deadline.wrapped_up.add("dry")
    github = FakeGitHub(repo)
    state = PullRequestState(head_sha=repo.head, base_sha=repo.base, state="open", draft=False, title="t", body="")
    result = await review_change(
        reader=github, source=LocalSource(repo), agents=agents([]), work_dir=tmp_path, repo=repo_of(pull(repo)),
        number=7, state=state, head_sha=repo.head, deadline=deadline,
        lenses=(Lens("dry", lens_agents([], kind="dry"), whole_pr=False),),
    )  # fmt: skip
    assert isinstance(result, Reviewed) and not result.outcome.cut_short


async def stuck_lens(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    """Report one finding at once, then never answer again."""
    if len(messages) == 1:
        return ModelResponse(parts=[ToolCallPart(REPORT_TOOL, {"finding": finding(5)})])
    return await never_answers(messages, info)


def cut_off(
    github: FakeGitHub, tmp_path: Path, agents: job.AgentFactory | None = None, lenses: tuple[Lens, ...] = ()
) -> job.ReviewDeps:
    """Deps whose checking cutoff comes after 0.9s (and the hard stop after 1s): fakes answer at once, so only
    work that never answers is still running by then."""
    deps = job_deps(github, tmp_path)
    return replace(deps, agents=agents or deps.agents, hard_deadline_s=1.0, lenses=lenses)


async def test_a_lens_stuck_at_the_cutoff_still_gets_a_verdict_from_what_was_judged(
    repo: GitRepo, tmp_path: Path
) -> None:
    """Until Nov 21 2026 this ended as "Review timed out": 7 of 9 live reviews lost their verdict that way."""
    github = FakeGitHub(repo)
    deps = cut_off(
        github,
        tmp_path,
        agents=lambda ws: (
            correctness_agent(FunctionModel(stuck_lens), ws),
            verifier_agent(verifier_model(lambda f: True), ws),
        ),
    )
    async with asyncio.timeout(5):
        result = await review_and_post(deps, pull(repo))
    assert (result.conclusion, result.title) == ("failure", "1 blocking issue(s)")
    [review] = github.reviews
    assert review["event"] == "REQUEST_CHANGES" and "**Partially reviewed:**" in review["body"]
    assert [c.line for c in github.inline_comments] == [5]


async def test_a_blocking_candidate_unchecked_at_the_cutoff_is_not_posted_and_not_approved(
    repo: GitRepo, tmp_path: Path
) -> None:
    github = FakeGitHub(repo)
    deps = cut_off(
        github,
        tmp_path,
        agents=lambda ws: (
            correctness_agent(lens_model([finding(5)]), ws),
            verifier_agent(FunctionModel(never_answers), ws),
        ),
    )
    async with asyncio.timeout(5):
        result = await review_and_post(deps, pull(repo))
    why = "the review ran out of time before checking 1 candidate finding(s)"
    assert (result.conclusion, result.title) == ("neutral", "Partially reviewed")
    assert why in result.summary and "The time limit left 1 candidate finding(s) unchecked." in result.summary
    assert "will be checked again" not in result.summary  # nothing is saved to recheck it from
    [review] = github.reviews
    assert review["event"] == "COMMENT" and why in review["body"]
    # Where to look, but not the claim: an unverified finding's text is never posted.
    assert "Not checked before the time limit: `stats.py:5`." in review["body"]
    assert finding(5)["title"] not in review["body"] and github.inline_comments == []
    # The check run's trail says the same, by location (from the bot's review of #65).
    assert "not checked in time: stats.py:5" in result.details
    assert f"not checked in time: {finding(5)['title']}" not in result.details


async def test_what_was_confirmed_before_the_cutoff_still_blocks(repo: GitRepo, tmp_path: Path) -> None:
    github = FakeGitHub(repo)
    deps = cut_off(
        github,
        tmp_path,
        agents=lambda ws: (
            correctness_agent(lens_model([finding(5), finding(4, title="Total starts at the wrong value")]), ws),
            verifier_agent(verifier_model(lambda f: True, stuck=lambda f: f["line"] == 4), ws),
        ),
    )
    async with asyncio.timeout(5):
        result = await review_and_post(deps, pull(repo))
    assert (result.conclusion, result.title) == ("failure", "1 blocking issue(s)")
    [review] = github.reviews
    assert review["event"] == "REQUEST_CHANGES" and "`stats.py:4`" in review["body"]
    assert [c.line for c in github.inline_comments] == [5]


async def test_an_unchecked_security_candidate_prevents_approval(repo: GitRepo, tmp_path: Path) -> None:
    """From the bot's review of #65: every lens that can block counts, not correctness alone."""
    security = Lens(
        "security", lambda ws: lens_pair("security", lens_model([finding(5)]), FunctionModel(never_answers), ws)
    )
    github = FakeGitHub(repo)
    async with asyncio.timeout(5):
        result = await review_and_post(cut_off(github, tmp_path, lenses=(security,)), pull(repo))
    assert (result.conclusion, result.title) == ("neutral", "Partially reviewed")
    [review] = github.reviews
    assert review["event"] == "COMMENT" and "Not checked before the time limit: `stats.py:5`." in review["body"]
    assert github.inline_comments == []


async def test_a_candidate_from_a_lens_that_never_finished_is_still_counted_as_unchecked(
    repo: GitRepo, tmp_path: Path
) -> None:
    """From the bot's review of #65: the lens reported it along the way, then never returned its final report,
    and its verifier never answered either. The candidate must not vanish with the lens."""
    github = FakeGitHub(repo)
    deps = cut_off(
        github,
        tmp_path,
        agents=lambda ws: (
            correctness_agent(FunctionModel(stuck_lens), ws),
            verifier_agent(FunctionModel(never_answers), ws),
        ),
    )
    async with asyncio.timeout(5):
        result = await review_and_post(deps, pull(repo))
    assert (result.conclusion, result.title) == ("neutral", "Partially reviewed")
    [review] = github.reviews
    assert review["event"] == "COMMENT" and "Not checked before the time limit: `stats.py:5`." in review["body"]


async def test_a_confirmed_finding_being_posted_at_the_cutoff_is_posted_once_and_not_called_unchecked(
    repo: GitRepo, tmp_path: Path
) -> None:
    """From the bot's review of #65: its verdict was in and its comment was on the way to GitHub. Cutting the
    write off would leave it unrecorded, so the verdict would post the same comment again."""
    github = FakeGitHub(repo, comment_delay=3.7)  # still posting when checking stops waiting, at 3.6s
    deps = replace(job_deps(github, tmp_path, [finding(5)]), hard_deadline_s=4.0)
    async with asyncio.timeout(8):
        result = await review_and_post(deps, pull(repo))
    assert (result.conclusion, result.title) == ("failure", "1 blocking issue(s)")
    [review] = github.reviews
    assert review["event"] == "REQUEST_CHANGES" and "Not checked" not in review["body"]
    assert len(github.review_comments) == 1 and review["comments"] == []  # posted by itself, not again in the verdict


async def test_an_earlier_verdict_does_not_stand_in_for_its_unchecked_replacement(
    repo: GitRepo, tmp_path: Path
) -> None:
    """From the bot's review of #65: a more urgent report for a line replaces the judged one. If only the
    replacement is still unjudged at the cutoff, the review must not approve over it."""

    async def lens(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        answered = sum(isinstance(message, ModelResponse) for message in messages)
        if answered == 0:
            return ModelResponse(parts=[ToolCallPart(REPORT_TOOL, {"finding": finding(5, priority=3)})])
        if answered == 1:
            await asyncio.sleep(0.3)  # the first report's verdict is in before the second is made
            return ModelResponse(parts=[ToolCallPart(REPORT_TOOL, {"finding": finding(5, priority=1)})])
        report = {"findings": [], "overall_correct": False, "reviewed": ["stats.py"]}
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, report)])

    github = FakeGitHub(repo)
    deps = cut_off(
        github,
        tmp_path,
        agents=lambda ws: (
            correctness_agent(FunctionModel(lens), ws),
            verifier_agent(verifier_model(lambda f: True, stuck=lambda f: f["priority"] == 1), ws),
        ),
    )
    async with asyncio.timeout(5):
        result = await review_and_post(deps, pull(repo))
    assert (result.conclusion, result.title) == ("neutral", "Partially reviewed")
    [review] = github.reviews
    assert review["event"] == "COMMENT" and "Not checked before the time limit: `stats.py:5`." in review["body"]


async def test_a_replaced_report_still_being_verified_does_not_hold_the_cutoff_open(
    repo: GitRepo, tmp_path: Path
) -> None:
    """From the bot's review of #65: the replacement's verdict is in, and the report it replaced shares its
    line. Only a task with its own verdict is left to finish; the stale one is cancelled like any other."""

    async def lens(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        answered = sum(isinstance(message, ModelResponse) for message in messages)
        if answered < 2:  # first a minor report whose verifier never answers, then a blocking one for the line
            return ModelResponse(parts=[ToolCallPart(REPORT_TOOL, {"finding": finding(5, priority=3 - 2 * answered)})])
        report = {"findings": [], "overall_correct": False, "reviewed": ["stats.py"]}
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, report)])

    github = FakeGitHub(repo)
    deps = cut_off(
        github,
        tmp_path,
        agents=lambda ws: (
            correctness_agent(FunctionModel(lens), ws),
            verifier_agent(verifier_model(lambda f: True, stuck=lambda f: f["priority"] == 3), ws),
        ),
    )
    async with asyncio.timeout(5):
        result = await review_and_post(deps, pull(repo))
    assert (result.conclusion, result.title) == ("failure", "1 blocking issue(s)")
    [review] = github.reviews
    assert review["event"] == "REQUEST_CHANGES" and "Not checked" not in review["body"]


async def test_an_advisory_check_unfinished_at_the_cutoff_does_not_withhold_approval(
    repo: GitRepo, tmp_path: Path
) -> None:
    """Advisory lenses can never block a merge, so their unfinished checks must not hold the verdict either."""
    dry = Lens(
        "dry",
        lambda ws: lens_pair("dry", lens_model([finding(5)]), FunctionModel(never_answers), ws),
        whole_pr=False,
    )
    github = FakeGitHub(repo)
    async with asyncio.timeout(5):
        result = await review_and_post(cut_off(github, tmp_path, lenses=(dry,)), pull(repo))
    assert (result.conclusion, result.title) == ("success", "No blocking issues")
    [review] = github.reviews
    assert review["event"] == "APPROVE" and "Not checked" not in review["body"] and github.inline_comments == []
    assert "1 dry candidate(s) not checked in time: dropped" in result.details  # counted, not described


async def test_a_recovered_job_past_the_cutoff_starts_no_lens_and_does_not_approve(
    repo: GitRepo, tmp_path: Path
) -> None:
    """From the bot's review of #65: with less left than posting needs, no model is called. This lens would
    answer at once with a clean report, which must not become an approval of a change nobody had time to read."""
    asked: list[int] = []

    def clean_at_once(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        asked.append(len(messages))
        report = {"findings": [], "overall_correct": True, "reviewed": ["stats.py"]}
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, report)])

    github = FakeGitHub(repo)
    deps = replace(
        job_deps(github, tmp_path),
        agents=lambda ws: (
            correctness_agent(FunctionModel(clean_at_once), ws),
            verifier_agent(verifier_model(lambda f: True), ws),
        ),
    )
    result = await review_and_post(deps, pull(repo), remaining_s=POSTING_RESERVE_S / 2)
    assert asked == []
    assert (result.conclusion, result.title) == ("neutral", "Partially reviewed")
    assert [r["event"] for r in github.reviews] == ["COMMENT"] and github.inline_comments == []


def test_the_cutoff_leaves_time_to_post() -> None:
    assert checking_cutoff(MAX_REVIEW_S) == MAX_REVIEW_S - POSTING_RESERVE_S
    assert checking_cutoff(10) == 9  # a budget too short for the full reserve keeps a tenth


async def test_work_is_cancelled_at_the_cutoff_and_only_then() -> None:
    cancelled = asyncio.Event()

    async def forever() -> str:
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()
        return "never"

    async def quick() -> str:
        return "done"

    async def times_out_itself() -> str:
        raise TimeoutError("the model server's own timeout")

    assert await before_cutoff(Deadline(soft_s=60, checking_s=0.05), forever) is None and cancelled.is_set()
    assert await before_cutoff(Deadline(soft_s=60, checking_s=60), quick) == "done"
    assert await before_cutoff(None, quick) == "done" and await before_cutoff(Deadline(soft_s=60), quick) == "done"
    with pytest.raises(TimeoutError, match="own timeout"):  # not mistaken for the cutoff
        await before_cutoff(Deadline(soft_s=60, checking_s=60), times_out_itself)

    # Past the cutoff the work is not begun at all: even work that would finish at once returns nothing.
    passed, begun = Deadline(soft_s=60, checking_s=0), Mock(side_effect=quick)
    assert await before_cutoff(passed, begun) is None
    begun.assert_not_called()

    tasks = [asyncio.create_task(quick()), asyncio.create_task(forever())]
    assert await until_cutoff(Deadline(soft_s=60, checking_s=0.05), tasks) == [tasks[1]] and tasks[1].cancelled()
    assert await until_cutoff(None, [asyncio.create_task(quick())]) == [] and await until_cutoff(passed, []) == []


async def test_a_push_mid_review_stops_it_instead_of_finishing(
    repo: GitRepo, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    finished: list[bool] = []

    async def slow(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        await asyncio.sleep(5)
        finished.append(True)
        return ModelResponse(
            parts=[ToolCallPart(info.output_tools[0].name, {"findings": [], "overall_correct": True, "reviewed": []})]
        )

    monkeypatch.setattr(job, "SUPERSEDED_CHECK_S", 0.05)
    github = FakeGitHub(repo, head_shas=[repo.head, repo.head, "b" * 40])  # moves on during the review
    deps = replace(
        job_deps(github, tmp_path, []),
        agents=lambda ws: (
            correctness_agent(FunctionModel(slow), ws),
            verifier_agent(verifier_model(lambda f: True), ws),
        ),
    )
    started = asyncio.get_running_loop().time()
    result = await review_and_post(deps, pull(repo))
    assert result.title == "Superseded" and github.reviews == [] and finished == []
    assert asyncio.get_running_loop().time() - started < 2


async def test_a_retry_after_a_partial_review_posts_the_full_verdict(repo: GitRepo, tmp_path: Path) -> None:
    """Live on #22: the partial COMMENT's marker made the retry's real verdict look already posted."""
    github = FakeGitHub(repo)
    await review_and_post(replace(job_deps(github, tmp_path), soft_deadline_s=0), pull(repo))  # cut short
    await review_and_post(replace(job_deps(github, tmp_path), soft_deadline_s=0), pull(repo))  # cut short again
    await review_and_post(job_deps(github, tmp_path), pull(repo))  # the retry gets through
    assert [r["event"] for r in github.reviews] == ["COMMENT", "APPROVE"]  # one partial, then the verdict


@pytest.mark.parametrize("state", ["closed", "open"])
async def test_timeout_cleanup_only_for_closed_pr(repo: GitRepo, tmp_path: Path, state: str) -> None:
    memory = ReviewMemory(tmp_path / "history")
    deps = replace(job_deps(FakeGitHub(repo, pr_state=state), tmp_path), memory=memory)
    event = pull(repo)
    memory.persistence(conversation_id(repo_of(event), event.number))
    check = await job.start_check(deps, event)
    result = await review_and_post(deps, event, check, remaining_s=0)
    await job.finish_check(deps, event, check, result)
    assert any((tmp_path / "history").iterdir()) == (state == "open")
