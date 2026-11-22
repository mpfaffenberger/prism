"""Review memory: the lens conversation per PR, so a new commit continues the last review.

Each PR gets one directory (named by a hash of its conversation id, so repository names can't shape
paths) holding a harness `FileStepStore` with the lens's message history, and `index.json` naming the
last *successful* review: which run, which head and merge-base, which prompt/model fingerprint, and the
findings it posted and dropped. A run that crashed is never continued by a later commit's review (its own
attempt resumes it after a restart: see `checkpoints`). Closing the PR deletes the directory, and with it
everything stored about the PR, checkpoints included.

Lives in the reviewer's own container, never in a sandbox: the history is trusted input to every later
review of the PR, so it must stay where untrusted code can't touch it.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

from pydantic_ai.messages import ModelMessage
from pydantic_ai_harness.step_persistence import FileStepStore, StepPersistence, continue_run

from autoreviewer.github import Repo

LENS_AGENT = "lens"


def conversation_id(repo: Repo, number: int) -> str:
    return f"review:{repo.owner}/{repo.name}#{number}".lower()


@dataclass(frozen=True)
class PriorReview:
    run_id: str
    head_sha: str
    base_sha: str
    version: str
    """Fingerprint of the lens prompt, model, and output schema: history from another version isn't reused."""
    findings: list[dict[str, object]] = field(default_factory=list[dict[str, object]])
    """Confirmed findings it posted (path, line, title)."""
    dropped: list[dict[str, object]] = field(default_factory=list[dict[str, object]])
    """Candidates the verifier judged not to be bugs (path, line, title, reason)."""
    unverified: list[dict[str, object]] = field(default_factory=list[dict[str, object]])
    """Candidates the verifier never judged (it failed): the next review checks them again."""


class ReviewMemory:
    def __init__(self, root: Path) -> None:
        self._root = root

    def directory(self, conversation: str) -> Path:
        return self._root / hashlib.sha256(conversation.encode()).hexdigest()[:32]

    def _store(self, conversation: str) -> FileStepStore:
        directory = self.directory(conversation) / "steps"
        directory.mkdir(parents=True, exist_ok=True)
        return FileStepStore(directory)

    def persistence(self, conversation: str) -> tuple[StepPersistence[None], str]:
        """A fresh capability that records one lens run under a new run id."""
        run_id = uuid.uuid4().hex
        return StepPersistence(store=self._store(conversation), agent_name=LENS_AGENT, run_id=run_id), run_id

    def last(self, conversation: str) -> PriorReview | None:
        try:
            data = json.loads((self.directory(conversation) / "index.json").read_text())
            return PriorReview(**data)
        except (FileNotFoundError, json.JSONDecodeError, TypeError):
            return None  # nothing usable: the next review starts fresh

    async def history(self, conversation: str, prior: PriorReview) -> list[ModelMessage]:
        return await continue_run(self._store(conversation), run_id=prior.run_id)

    def remember(self, conversation: str, review: PriorReview) -> None:
        directory = self.directory(conversation)
        directory.mkdir(parents=True, exist_ok=True)
        temporary = directory / f".index.{uuid.uuid4().hex}.json"
        temporary.write_text(json.dumps(asdict(review)))
        os.replace(temporary, directory / "index.json")  # atomic: a crash leaves the old index intact

    def forget(self, conversation: str) -> None:
        shutil.rmtree(self.directory(conversation), ignore_errors=True)
