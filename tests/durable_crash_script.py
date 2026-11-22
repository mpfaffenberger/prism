"""Helper for test_durable: one process that crashes mid-review, then one that recovers it.

    python durable_crash_script.py <dir> enqueue-and-crash    # exits 9 from inside the lens's first model call
    python durable_crash_script.py <dir> crash-between-turns  # exits 9 in the lens's third turn, once the
                                                              # finding it reported in turn 1 is verified and posted
    python durable_crash_script.py <dir> recover               # DBOS recovers the review; exits 0
    python durable_crash_script.py <dir> recover-after-deadline

GitHub is faked to <dir>/github.json, and the review memory (where turn checkpoints live) is <dir>/memory, so
their state survives the crash, like the real ones would. Every model call appends a line to <dir>/calls.log:
`lens:<turn>` or `verifier:<path>:<line>`.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path
from time import time
from typing import Any

from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from autoreviewer.github import Conclusion, Repo, ReviewComment, ReviewEvent
from autoreviewer.review import durable
from autoreviewer.review.agents import REPORT_TOOL, correctness_agent, verifier_agent
from autoreviewer.review.memory import ReviewMemory
from autoreviewer.web.routing import PullRequestUpdated
from tests.review_fakes import FakeGitHub, GitRepo, finding, job_deps, make_repo, verifier_model

DIR = Path(sys.argv[1])
MODE = sys.argv[2]
CALLS = DIR / "calls.log"
GITHUB = DIR / "github.json"
FINDING = finding(5)
POST_TIMEOUT_S = 60
NOT_POSTED = 3
"""Exit status when the verified finding never reached GitHub: the scenario couldn't be set up."""


def _comment(data: dict[str, Any]) -> ReviewComment:
    return ReviewComment(**data)


class PersistentGitHub(FakeGitHub):
    """FakeGitHub whose writes (checks, reviews, inline comments) are saved to disk after every call."""

    def _save(self) -> None:
        reviews = [{**r, "comments": [asdict(c) for c in r["comments"]]} for r in self.reviews]
        comments = [{**c, "comment": asdict(c["comment"])} for c in self.review_comments]
        GITHUB.write_text(json.dumps({"checks": self.checks, "reviews": reviews, "review_comments": comments}))

    @classmethod
    def load(cls, repo: GitRepo) -> PersistentGitHub:
        github = cls(repo)
        if GITHUB.exists():
            saved = json.loads(GITHUB.read_text())
            github.checks = saved["checks"]
            github.reviews = [{**r, "comments": [_comment(c) for c in r["comments"]]} for r in saved["reviews"]]
            github.review_comments = [{**c, "comment": _comment(c["comment"])} for c in saved["review_comments"]]
        return github

    async def start_check(self, repo: Repo, head_sha: str) -> int:
        check_id = await super().start_check(repo, head_sha)
        self._save()
        return check_id

    async def finish_check(
        self, repo: Repo, check_id: int, *, conclusion: Conclusion, title: str, summary: str, text: str = ""
    ) -> None:
        await super().finish_check(repo, check_id, conclusion=conclusion, title=title, summary=summary, text=text)
        self._save()

    async def create_review(
        self, repo: Repo, number: int, *, commit_sha: str, event: ReviewEvent, body: str, comments: list[ReviewComment]
    ) -> None:
        await super().create_review(repo, number, commit_sha=commit_sha, event=event, body=body, comments=comments)
        self._save()

    async def create_review_comment(self, repo: Repo, number: int, *, commit_sha: str, comment: ReviewComment) -> None:
        await super().create_review_comment(repo, number, commit_sha=commit_sha, comment=comment)
        self._save()


def log_call(name: str) -> None:
    with CALLS.open("a") as calls:
        calls.write(name + "\n")


def crash() -> None:
    os._exit(9)  # a hard crash in the middle of the review: no cleanup, no exception handling


async def wait_until_posted() -> None:
    """Until the finding's inline comment is on GitHub. It's posted only after its verdict was checkpointed, so
    crashing now loses no verifier turn: a recovery that runs the verifier again repeated a finished run."""
    for _ in range(POST_TIMEOUT_S * 20):
        if GITHUB.exists() and json.loads(GITHUB.read_text())["review_comments"]:
            return
        await asyncio.sleep(0.05)
    os._exit(NOT_POSTED)


async def lens(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    """The correctness lens: one turn per call, numbered from the responses already in its (restored) history."""
    turn = 1 + sum(isinstance(message, ModelResponse) for message in messages)
    log_call(f"lens:{turn}")
    if MODE == "enqueue-and-crash":
        crash()
    if MODE == "crash-between-turns":
        if turn == 1:
            return ModelResponse(parts=[ToolCallPart(tool_name=REPORT_TOOL, args={"finding": FINDING})])
        if turn == 2:
            return ModelResponse(parts=[ToolCallPart(tool_name="changed_files", args={})])
        await wait_until_posted()
        crash()
    final = {"findings": [FINDING], "overall_correct": False, "reviewed": []}  # every finding, reported or not
    return ModelResponse(parts=[ToolCallPart(tool_name=info.output_tools[0].name, args=final)])


def confirm(reported: dict[str, Any]) -> bool:
    log_call(f"verifier:{reported['path']}:{reported['line']}")
    return True


def origin() -> GitRepo:
    """The same repository in both processes: created by the first, reopened by the second."""
    path = DIR / "origin"
    if not path.exists():
        return make_repo(path)
    sha = lambda ref: subprocess.run(["git", "rev-parse", ref], cwd=path, capture_output=True, text=True).stdout.strip()  # noqa: E731
    return GitRepo(path, sha("HEAD~1"), sha("HEAD"))


async def main() -> None:
    repo = origin()
    pull = PullRequestUpdated(installation_id=1, owner="mpfaffenberger", repo="demo", number=7, head_sha=repo.head)
    deps = replace(
        job_deps(PersistentGitHub.load(repo), DIR),
        agents=lambda workspace: (
            correctness_agent(FunctionModel(lens), workspace),
            verifier_agent(verifier_model(confirm), workspace),
        ),
        memory=ReviewMemory(DIR / "memory"),
    )
    if MODE == "recover-after-deadline":
        # Advance only the review clock: simulate ten minutes of downtime without sleeping.
        durable.time = lambda: time() + 600
    durable.configure(DIR / "dbos.sqlite")
    await durable.launch(deps, concurrency=2)
    try:
        if MODE in {"enqueue-and-crash", "crash-between-turns"}:
            await (await durable.enqueue(pull)).get_result()  # never returns: the lens kills the process
        else:
            # launch() recovered the interrupted workflow; wait for it by its deterministic id.
            from dbos import DBOS

            handle = await DBOS.retrieve_workflow_async(durable.workflow_id(pull))
            await asyncio.wait_for(handle.get_result(), timeout=60)
    finally:
        durable.shutdown()


asyncio.run(main())
