"""Bounded, PR-local cache of successful role transcripts. Nothing is published until commit.

`run` is how every review role runs: it continues the role's last compatible session and, given this attempt's
`TurnCheckpoints`, resumes the run's settled turns after a restart."""

from __future__ import annotations

import asyncio
import fcntl
import hashlib
import json
import os
from collections.abc import Awaitable, Callable
from importlib.metadata import version
from pathlib import Path

from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelMessagesTypeAdapter
from pydantic_ai.models import Model

from autoreviewer.review.checkpoints import Ran, Start, TurnCheckpoints

# Bump for changes to the review runtime/tools that invalidate earlier conversations.
_RUNTIME = (1, version("pydantic-ai-slim"), version("pydantic-ai-harness"), version("pydantic"))
_MAX_ENTRY = 4 << 20
_MAX_BYTES = 32 << 20
_MAX_ENTRIES = 128


def _key[T](namespace: str, base: str, role: str, subject: str, agent: Agent[None, T]) -> str | None:
    model = agent.model
    # Unresolved models and run-dependent prompt functions have no stable identity here.
    if not isinstance(model, Model) or agent._system_prompt_functions:
        return None
    prompts = [item.instruction for item in (*agent._instructions, *agent._cap_instructions)]
    if not all(isinstance(prompt, str) for prompt in prompts):
        return None
    identity = (type(model).__module__, type(model).__qualname__, model.system, model.model_name, model.base_url)
    material = (
        _RUNTIME,
        namespace,
        base,
        role,
        subject,
        identity,
        model.settings,
        agent.model_settings,
        agent._system_prompts,
        prompts,
        agent.output_json_schema(),
    )
    try:
        return hashlib.sha256(json.dumps(material, sort_keys=True, allow_nan=False).encode()).hexdigest()
    except (TypeError, ValueError):
        return None  # e.g. dynamic model settings; never guess at compatibility


class RoleSessions:
    def __init__(
        self,
        directory: Path,
        head: str,
        base: str,
        compatible: Callable[[str], Awaitable[bool]],
        namespace: str,
        turns: TurnCheckpoints | None = None,
    ) -> None:
        """`turns`: this attempt's checkpoints, so a restart resumes each run instead of repeating it."""
        self._directory = directory / "role-sessions"
        self._path = self._directory / "snapshot"
        self._head = head
        self._base = base
        self._compatible = compatible
        self._namespace = namespace
        self._pending: dict[str, bytes] = {}
        self._snapshot: dict[str, bytes] | None = None
        self._read_lock = asyncio.Lock()
        self._stamp: tuple[int, int] | None = None
        self._turns = turns

    async def run[T](
        self,
        role: str,
        agent: Agent[None, T],
        start: Start[T],
        *,
        subject: str = "",
        history: list[ModelMessage] | None = None,
        on_resume: Callable[[list[ModelMessage]], None] | None = None,
    ) -> Ran[T]:
        """Run `role`: on `history` if given (a continuation), else on its last compatible session; resumed from
        this attempt's settled turns after a restart; its transcript recorded for the next review."""
        if history is None:
            history = await self.history(role, agent, subject=subject)
        key = _key(self._namespace, self._base, role, subject, agent) if self._turns else None
        if self._turns is None or key is None:
            ran = await start(history, False, ())
        else:
            ran = await self._turns.run(key, agent, start, history=history, on_resume=on_resume)
        self.record(role, agent, ran.all_messages(), subject=subject)
        return ran

    def _load(self) -> dict[str, bytes]:
        # One checksummed snapshot: a valid-looking edit to a head, key or message is a miss too.
        try:
            with self._path.open("rb") as file:
                raw = file.read(_MAX_BYTES + 1)
            checksum, body = raw.split(b"\n", 1)
            if len(raw) > _MAX_BYTES or hashlib.sha256(body).hexdigest().encode() != checksum:
                return {}
            rows = body.split(b"\n") if body else []
            if len(rows) > _MAX_ENTRIES:
                return {}
            entries: dict[str, bytes] = {}
            for row in rows:
                key, data = row.split(b" ", 1)
                if len(key) != 64 or len(data) > _MAX_ENTRY:
                    return {}
                entries[key.decode("ascii")] = data
            return entries
        except (FileNotFoundError, ValueError):
            return {}

    async def history[T](self, role: str, agent: Agent[None, T], *, subject: str = "") -> list[ModelMessage]:
        key = _key(self._namespace, self._base, role, subject, agent)
        if key is None:
            return []
        async with self._read_lock:
            try:
                stat = self._path.stat()
            except FileNotFoundError:
                self._snapshot = None
                return []
            stamp = stat.st_ino, stat.st_mtime_ns
            if self._snapshot is None or self._stamp != stamp:
                self._snapshot = await asyncio.to_thread(self._load)
                self._stamp = stamp
            data = self._snapshot.get(key)
        if data is None:
            return []
        try:
            head_json, message_json = data.split(b"\t", 1)
            saved_head = json.loads(head_json)
            if not isinstance(saved_head, str):
                return []
            messages = ModelMessagesTypeAdapter.validate_json(message_json)
        except ValueError:
            return []
        if saved_head == self._head or await self._compatible(saved_head):
            return messages
        return []

    def record[T](self, role: str, agent: Agent[None, T], messages: list[ModelMessage], *, subject: str = "") -> None:
        key = _key(self._namespace, self._base, role, subject, agent)
        if key is None:
            return
        data = json.dumps(self._head).encode() + b"\t" + ModelMessagesTypeAdapter.dump_json(messages)
        if len(data) <= _MAX_ENTRY:
            self._pending.pop(key, None)
            self._pending[key] = data
            self._trim(self._pending)

    @staticmethod
    def _trim(entries: dict[str, bytes]) -> None:
        size = 65 + sum(66 + len(data) for data in entries.values())
        while len(entries) > _MAX_ENTRIES or size > _MAX_BYTES:
            size -= 66 + len(entries.pop(next(iter(entries))))

    def commit(self) -> None:
        if not self._pending:
            return
        self._directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        # Serialize read/merge/replace, even when two workers finish different roles at once.
        lock = os.open(self._directory, os.O_DIRECTORY)
        try:
            fcntl.flock(lock, fcntl.LOCK_EX)
            entries = self._load()
            for key, data in self._pending.items():
                entries.pop(key, None)
                entries[key] = data
            self._trim(entries)
            body = b"\n".join(key.encode() + b" " + data for key, data in entries.items())
            temporary = self._directory / ".pending"
            try:
                with open(temporary, "wb", opener=lambda p, f: os.open(p, f, 0o600)) as file:
                    file.write(hashlib.sha256(body).hexdigest().encode() + b"\n")
                    file.write(body)
                    file.flush()
                    os.fsync(file.fileno())
                os.replace(temporary, self._path)
                os.fsync(lock)
            finally:
                temporary.unlink(missing_ok=True)
            self._pending.clear()
            self._snapshot = None
        finally:
            os.close(lock)


async def run_role[T](
    sessions: RoleSessions | None,
    role: str,
    agent: Agent[None, T],
    start: Start[T],
    *,
    subject: str = "",
    history: list[ModelMessage] | None = None,
    on_resume: Callable[[list[ModelMessage]], None] | None = None,
) -> Ran[T]:
    """`RoleSessions.run`, or just the run on `history` when there's no review memory (on-demand reviews)."""
    if sessions is None:
        return await start(history, False, ())
    return await sessions.run(role, agent, start, subject=subject, history=history, on_resume=on_resume)
