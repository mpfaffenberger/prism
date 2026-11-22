"""Turn checkpoints (#60): a restarted attempt resumes each agent run at its last settled turn.

A crash is simulated by a model that raises at a given turn: like a killed process, the run stops there with every
earlier tool cycle settled. tests/test_durable.py kills a real process between turns.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel

from autoreviewer.review.agents import REPORT_TOOL
from autoreviewer.review.checkpoints import Start, TurnCheckpoints
from autoreviewer.review.compaction import review_compaction
from autoreviewer.review.pipeline import Judge, Outcome, PullRequestText
from autoreviewer.review.progress import Progress
from autoreviewer.review.sessions import RoleSessions
from tests.review_fakes import finding

pytestmark = pytest.mark.anyio

ATTEMPT = "review:octo/demo#7@abc123"
LAST_TURN = 4
"""Turns 1-3 call `lookup`; turn 4 reports."""


class Report(BaseModel):
    turns: int


class Crash(Exception):
    """The process dying mid-run."""


def last_lookup(messages: list[ModelMessage]) -> int:
    """The newest `lookup` result's number: unlike counting responses, it survives compaction's rewrites."""
    for message in reversed(messages):
        if isinstance(message, ModelRequest):
            for part in reversed(message.parts):
                if isinstance(part, ToolReturnPart) and part.tool_name == "lookup":
                    return int(str(part.content).split()[1])
    return 0


def scripted(
    log: list[str], *, name: str = "model", crash_at: int | None = None, last: int = LAST_TURN
) -> FunctionModel:
    """Calls `lookup` once per turn, then reports; each request is logged as `name:turn`. Summary requests (the
    compactor's, without output tools) are logged as `name:summary`."""

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if not info.output_tools:
            log.append(f"{name}:summary")
            return ModelResponse(parts=[TextPart("SUMMARY of the lookups so far")])
        turn = last_lookup(messages) + 1
        log.append(f"{name}:{turn}")
        if turn == crash_at:
            raise Crash
        if turn < last:
            return ModelResponse(parts=[ToolCallPart("lookup", {"i": turn})])
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, {"turns": turn})])

    return FunctionModel(respond, model_name=name)


def lens(model: FunctionModel, *, compact_at: int | None = None, result_size: int = 0) -> Agent[None, Report]:
    capabilities = [review_compaction(compact_at)] if compact_at else []
    agent = Agent(model, output_type=Report, instructions="Review the change.", capabilities=capabilities)

    @agent.tool_plain
    def lookup(i: int) -> str:
        return f"result {i} " + "lorem ipsum " * result_size

    return agent


def roles(directory: Path, *, attempt: str = ATTEMPT, namespace: str = "1:primary") -> RoleSessions:
    return RoleSessions(
        directory, "head", "base", AsyncMock(return_value=False), namespace, turns=TurnCheckpoints(directory, attempt)
    )


def start(agent: Agent[None, Report]) -> Start[Report]:
    return lambda messages, resuming, recorders: agent.run(
        None if resuming else "Review it.", message_history=messages, capabilities=list(recorders)
    )


async def crash(directory: Path, *, at: int, name: str = "model", **kwargs: object) -> list[str]:
    log: list[str] = []
    agent = lens(scripted(log, name=name, crash_at=at), **kwargs)  # type: ignore[arg-type]
    with pytest.raises(Crash):
        await roles(directory).run("correctness", agent, start(agent))
    return log


async def test_a_restart_resumes_at_the_last_settled_turn(tmp_path: Path) -> None:
    assert await crash(tmp_path, at=3) == ["model:1", "model:2", "model:3"]

    log: list[str] = []
    resumed: list[list[ModelMessage]] = []
    agent = lens(scripted(log))
    ran = await roles(tmp_path).run("correctness", agent, start(agent), on_resume=resumed.append)

    # Turns 1 and 2 aren't sent again; turn 3 was in flight with no recorded response, so it is.
    assert log == ["model:3", "model:4"]
    assert ran.output == Report(turns=LAST_TURN)
    assert [last_lookup(history) for history in resumed] == [2]  # the hook saw the settled history first
    prompts = [
        p for m in ran.all_messages() if isinstance(m, ModelRequest) for p in m.parts if isinstance(p, UserPromptPart)
    ]
    assert [p.content for p in prompts] == ["Review it."]  # resuming adds no second prompt


async def test_a_finished_run_returns_its_output_without_a_model_call(tmp_path: Path) -> None:
    first_log: list[str] = []
    agent = lens(scripted(first_log))
    first = await roles(tmp_path).run("correctness", agent, start(agent))

    log: list[str] = []
    again = lens(scripted(log))
    ran = await roles(tmp_path).run("correctness", again, start(again))

    assert len(first_log) == LAST_TURN and log == []
    assert ran.output == first.output and ran.all_messages() == first.all_messages()
    assert ran.usage.requests == 0  # this process spent nothing on it


@pytest.mark.parametrize(
    ("name", "namespace", "attempt"),
    [
        pytest.param("other-model", "1:primary", ATTEMPT, id="another provider or model"),
        pytest.param("model", "1:overflow", ATTEMPT, id="the overflow namespace"),
        pytest.param("model", "1:primary", ATTEMPT + ":attempt:2", id="another attempt"),
    ],
)
async def test_checkpoints_are_never_shared(tmp_path: Path, name: str, namespace: str, attempt: str) -> None:
    await crash(tmp_path, at=3)

    log: list[str] = []
    agent = lens(scripted(log, name=name))
    await roles(tmp_path, attempt=attempt, namespace=namespace).run("correctness", agent, start(agent))
    assert log == [f"{name}:{turn}" for turn in range(1, LAST_TURN + 1)]  # a fresh run: nothing borrowed

    log.clear()
    original = lens(scripted(log))
    await roles(tmp_path).run("correctness", original, start(original))
    assert log == ["model:3", "model:4"]  # and the interrupted run's own checkpoint is untouched


@pytest.mark.parametrize(
    "turns_after_summary",
    [
        pytest.param(2, id="later turn: the summary is kept"),
        pytest.param(0, id="the summarized request: repeated with its summary"),
    ],
)
async def test_a_resume_repeats_only_the_request_in_flight(tmp_path: Path, turns_after_summary: int) -> None:
    """Snapshots hold compacted history, so turns and summaries before the crash aren't redone. The request in
    flight is repeated together with any compaction that prepared it (see checkpoints' docstring)."""
    compacting = {"compact_at": 600, "result_size": 20}  # one summary in the run, after which history fits
    baseline: list[str] = []
    agent = lens(scripted(baseline, last=22), **compacting)
    await roles(tmp_path, attempt="baseline").run("correctness", agent, start(agent))
    assert baseline.count("model:summary") == 1
    summary = baseline.index("model:summary")
    crash_at = int(baseline[summary + 1 + turns_after_summary].split(":")[1])
    in_flight = baseline.index(f"model:{crash_at}")

    log: list[str] = []
    crashing = lens(scripted(log, crash_at=crash_at, last=22), **compacting)
    with pytest.raises(Crash):
        await roles(tmp_path).run("correctness", crashing, start(crashing))
    resumed = lens(scripted(log, last=22), **compacting)
    await roles(tmp_path).run("correctness", resumed, start(resumed))

    repeated = baseline[summary : in_flight + 1] if turns_after_summary == 0 else [baseline[in_flight]]
    assert log == baseline[: in_flight + 1] + repeated + baseline[in_flight + 1 :]


async def test_discard_drops_one_attempts_checkpoints(tmp_path: Path) -> None:
    for attempt in ("first", "second"):
        agent = lens(scripted([]))
        await roles(tmp_path, attempt=attempt).run("correctness", agent, start(agent))

    TurnCheckpoints.discard(tmp_path, "first")

    first: list[str] = []
    second: list[str] = []
    for attempt, log in (("first", first), ("second", second)):
        agent = lens(scripted(log))
        await roles(tmp_path, attempt=attempt).run("correctness", agent, start(agent))
    assert len(first) == LAST_TURN and second == []


async def test_an_unreadable_result_is_run_again_rather_than_trusted(tmp_path: Path) -> None:
    agent = lens(scripted([]))
    await roles(tmp_path).run("correctness", agent, start(agent))
    [result] = TurnCheckpoints.attempt_directory(tmp_path, ATTEMPT).glob("*/result.json")
    result.write_text('{"output": {"turns": "not a number"}')

    log: list[str] = []
    again = lens(scripted(log))
    ran = await roles(tmp_path).run("correctness", again, start(again))
    assert ran.output == Report(turns=LAST_TURN) and log  # ran from its snapshot, not from the broken record


def report_call(call_id: str, line: int) -> ModelResponse:
    return ModelResponse(parts=[ToolCallPart(REPORT_TOOL, {"finding": finding(line)}, call_id)])


async def test_replay_resubmits_each_answered_report_in_order() -> None:
    judge = Judge(PullRequestText("title", "", [], [], {}), AsyncMock(), Progress(), Outcome(), label="verifier")
    submitted: list[int] = []
    judge.submit = lambda reported: submitted.append(reported.line) or ""  # type: ignore[method-assign]

    judge.replay(
        [
            ModelRequest(parts=[UserPromptPart("Review it.")]),
            report_call("a", 5),
            ModelRequest(parts=[ToolReturnPart(REPORT_TOOL, "Recorded", "a")]),
            report_call("b", 9),  # retried (bad arguments): never recorded
            ModelRequest(parts=[ToolReturnPart(REPORT_TOOL, "bad arguments", "b", outcome="retried")]),
            report_call("c", 3),
            ModelRequest(parts=[ToolReturnPart(REPORT_TOOL, "Recorded", "c")]),
            report_call("d", 7),  # closed out as interrupted when an earlier resume left it pending
            ModelRequest(parts=[ToolReturnPart(REPORT_TOOL, "interrupted", "d", outcome="interrupted")]),
            report_call("e", 8),  # in flight at the crash: the resumed run executes it, so it isn't replayed
        ]
    )

    assert submitted == [5, 3]
