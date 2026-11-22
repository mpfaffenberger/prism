"""Turn checkpoints: a restarted receiver resumes each review agent at its last settled turn.

DBOS checkpoints the whole review-and-post phase as one step, so a restart used to repeat every model call the
attempt had already made. Now each agent run in an attempt records its settled turns as it goes (the harness's
`StepPersistence`: one snapshot after every completed tool cycle, compaction included) and, once it ends, its
output. A recovered attempt repeats no finished turn: a finished run returns its stored output without a model
call, and an interrupted one continues from its newest settled snapshot. A request that was in flight when the
process died is sent again, so completed turns are never repeated but billing is not exactly-once.

Snapshots hold compacted history, so compaction done for an earlier turn isn't redone. The request in flight is
repeated together with the compaction that prepared it: a summary made for that request is paid for again, because
the harness's token estimate is anchored on the last response's reported usage, which predates the summary.

The checkpoint key includes the attempt's DBOS workflow ID (one try at one head), the role, the candidate a
checker verifies, the session namespace, the merge base, and the agent's identity (`sessions._key`: provider and
model, prompts, output schema). If any of these changes, for example when a recovered run uses another model or
runs in the overflow namespace, the run starts fresh instead of reusing the checkpoint. Checkpoints live in the
PR's review memory, which closing the PR deletes, and `discard` drops an attempt's checkpoints once it has its
verdict.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shutil
import uuid
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from pydantic import TypeAdapter
from pydantic_ai import Agent
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import ModelMessage, ModelMessagesTypeAdapter
from pydantic_ai.usage import RunUsage
from pydantic_ai_harness.step_persistence import FileStepStore, StepPersistence

RECORDER_ID = "turn_checkpoints"
"""Capability id of the recorder: distinct from the correctness lens's own `StepPersistence` (review memory)."""
_RESULT = "result.json"
_KEEP_SNAPSHOTS = 2
"""Per execution. The store always keeps the newest complete snapshot too, the one a resume reads."""


class Ran[T](Protocol):
    """What callers read from a run: an `AgentRunResult`, or a run finished before a restart."""

    @property
    def output(self) -> T: ...

    @property
    def usage(self) -> RunUsage: ...

    def all_messages(self) -> list[ModelMessage]: ...


@dataclass(frozen=True)
class Finished[T]:
    """A run that finished before the restart: its output and transcript, and no new usage."""

    output: T
    messages: list[ModelMessage]
    usage: RunUsage = field(default_factory=RunUsage)

    def all_messages(self) -> list[ModelMessage]:
        return list(self.messages)


type Start[T] = Callable[[list[ModelMessage] | None, bool, Sequence[AbstractCapability[None]]], Awaitable[Ran[T]]]
"""Starts the agent run: (message history, resuming, extra capabilities). Resuming, the history already holds
the run's prompt, so the caller passes none."""


class TurnCheckpoints:
    def __init__(self, directory: Path, attempt: str) -> None:
        """`directory`: the PR's review memory directory. `attempt`: the DBOS workflow id."""
        self._root = self.attempt_directory(directory, attempt)

    @staticmethod
    def attempt_directory(directory: Path, attempt: str) -> Path:
        return directory / "turns" / hashlib.sha256(attempt.encode()).hexdigest()[:32]

    @classmethod
    def discard(cls, directory: Path, attempt: str) -> None:
        """Drop an attempt's checkpoints: it has its verdict, so nothing resumes it again."""
        shutil.rmtree(cls.attempt_directory(directory, attempt), ignore_errors=True)

    async def run[T](
        self,
        key: str,
        agent: Agent[None, T],
        start: Start[T],
        *,
        history: list[ModelMessage] | None,
        on_resume: Callable[[list[ModelMessage]], None] | None = None,
    ) -> Ran[T]:
        """Run `agent` under checkpoint `key`, unless it already finished in this attempt.

        Interrupted, it continues from its newest settled turn; `on_resume` sees that history first, so state the
        run built through its tools (a lens's reported findings) can be rebuilt. Otherwise it starts on `history`.
        """
        if (outputs := _outputs(agent)) is None:
            return await start(history, False, ())  # no stored form for its output: it just runs
        directory = self._root / key
        if (finished := await asyncio.to_thread(_load_finished, directory, outputs)) is not None:
            return finished
        store = FileStepStore(directory, max_snapshots_per_run=_KEEP_SNAPSHOTS)
        settled = await _latest_settled(store)
        if settled is not None and on_resume is not None:
            on_resume(settled)
        recorder = StepPersistence(store=store, run_id=uuid.uuid4().hex, id=RECORDER_ID)
        ran = await start(settled if settled is not None else history, settled is not None, [recorder])
        await asyncio.to_thread(_save_finished, directory, outputs, ran)
        return ran


async def _latest_settled(store: FileStepStore) -> list[ModelMessage] | None:
    """The newest settled history across this run's executions: each resumed execution's snapshots include the
    history it resumed from, so the newest one with any snapshot is the furthest point reached."""
    runs = sorted(await store.list_runs(), key=lambda record: record.started_at, reverse=True)
    for record in runs:
        if (snapshot := await store.latest_snapshot(run_id=record.run_id)) is not None:
            return snapshot.messages
    return None


def _outputs[T](agent: Agent[None, T]) -> TypeAdapter[T] | None:
    """How a finished run's output is stored: every review agent returns one model class. Any other output spec
    (output functions, unions) isn't checkpointed."""
    spec = agent.output_type
    return TypeAdapter(spec) if isinstance(spec, type) else None


def _load_finished[T](directory: Path, outputs: TypeAdapter[T]) -> Finished[T] | None:
    try:
        data = json.loads((directory / _RESULT).read_bytes())
        output = outputs.validate_python(data["output"])
        return Finished(output, ModelMessagesTypeAdapter.validate_python(data["messages"]))
    except (FileNotFoundError, KeyError, TypeError, ValueError):
        return None  # unfinished, or unreadable: run it (again) rather than trust a partial record


def _save_finished[T](directory: Path, outputs: TypeAdapter[T], ran: Ran[T]) -> None:
    data = {
        "output": outputs.dump_python(ran.output, mode="json"),
        "messages": ModelMessagesTypeAdapter.dump_python(ran.all_messages(), mode="json"),
    }
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = directory / f".{_RESULT}.{uuid.uuid4().hex}"
    try:
        with open(temporary, "w", opener=lambda path, flags: os.open(path, flags, 0o600)) as file:
            json.dump(data, file)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, directory / _RESULT)  # atomic: a crash leaves no half-written result
    finally:
        temporary.unlink(missing_ok=True)
