"""Time limits: a lens past the soft deadline wraps up with a report; a cut-short review never approves."""

from __future__ import annotations

import asyncio
import time
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.tools import ToolDefinition

from autoreviewer.github import PullRequestState
from autoreviewer.review import job
from autoreviewer.review.agents import REPORT_TOOL, correctness_agent, verifier_agent
from autoreviewer.review.deadline import FIVE_MINUTES, ONE_MINUTE, WRAP_UP, Deadline
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
    ("given_by", "warned_again"),
    [pytest.param("this attempt", False), pytest.param("an earlier attempt", True)],
)
async def test_a_resumed_run_is_not_given_a_notice_twice(
    workspace: Workspace, given_by: str, warned_again: bool
) -> None:
    """Resumed after a restart, the run's history already holds the notices this attempt gave it. A notice from an
    earlier attempt, in a continued conversation, doesn't count: this attempt still warns."""
    deadline = Deadline(soft_s=540, started=time.monotonic() - 300)  # four minutes left: the five-minute stage
    given = datetime.now(UTC) if given_by == "this attempt" else deadline.began - timedelta(hours=1)
    history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart("review")]),
        ModelResponse(parts=[ToolCallPart("changed_files", {}, "first")]),
        ModelRequest(
            parts=[ToolReturnPart("changed_files", "stats.py", "first"), UserPromptPart(FIVE_MINUTES, timestamp=given)]
        ),
        ModelResponse(parts=[ToolCallPart("changed_files", {}, "second")]),
        ModelRequest(parts=[ToolReturnPart("changed_files", "stats.py", "second")]),
    ]
    notices: list[int] = []

    def lens(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        notices.append(
            sum(
                isinstance(p, UserPromptPart) and p.content == FIVE_MINUTES
                for m in messages
                if isinstance(m, ModelRequest)
                for p in m.parts
            )
        )
        report = {"findings": [], "overall_correct": True, "reviewed": ["stats.py"]}
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, report)])

    await correctness_agent(FunctionModel(lens), workspace).run(
        message_history=history, capabilities=[deadline.watch("reviewer")]
    )
    assert notices == [2 if warned_again else 1]


@pytest.mark.parametrize(("budget", "focus", "stop"), [(540, 300, 60), (54, 30, 6)])
async def test_countdown_is_once_per_stage_and_final_minute_is_report_only(
    workspace: Workspace, budget: int, focus: int, stop: int
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
    deadline = Deadline(soft_s=540)
    capability = deadline.watch("reviewer")
    tools = [ToolDefinition(name="explore")]
    assert await capability.prepare_tools(Mock(), tools) == tools
    assert not deadline.wrapped_up
    deadline.started -= 481
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


@pytest.mark.parametrize("remaining, warning", [(500, None), (320, FIVE_MINUTES), (100, ONE_MINUTE), (30, WRAP_UP)])
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
        hard_deadline_s=900,  # capped at 600: wrap-up is 60s before the effective hard limit
        soft_deadline_s=540,
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


async def test_the_hard_deadline_ends_the_review_without_a_verdict(repo: GitRepo, tmp_path: Path) -> None:
    async def stuck(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:  # report one finding at once, then hang
            return ModelResponse(parts=[ToolCallPart(REPORT_TOOL, {"finding": finding(5)})])
        await asyncio.sleep(60)
        raise AssertionError("cancelled long before this")

    github = FakeGitHub(repo)
    deps = replace(
        job_deps(github, tmp_path, []),
        agents=lambda ws: (
            correctness_agent(FunctionModel(stuck), ws),
            verifier_agent(verifier_model(lambda f: True), ws),
        ),
        hard_deadline_s=0.5,
    )
    result = await review_and_post(deps, pull(repo))
    assert result.title == "Review timed out" and github.reviews == []
    assert [c.line for c in github.inline_comments] == [5]  # confirmed before the deadline: still posted


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


def test_multimodal_prompts_are_never_taken_for_notices() -> None:
    """A user prompt with a list of contents (text plus an image, say) can't be a notice, and mustn't break the
    search for one."""
    from autoreviewer.review.deadline import _last_notice

    started = datetime.now(UTC) - timedelta(minutes=1)
    history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(FIVE_MINUTES)]),
        ModelRequest(parts=[UserPromptPart(["see the screenshot", "and the log"])]),
    ]
    assert _last_notice(history, since=started) == FIVE_MINUTES
