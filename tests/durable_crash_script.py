"""Helper for test_durable: one process that crashes mid-review, then one that recovers it.

    python durable_crash_script.py <dir> enqueue-and-crash   # exits 9 from inside the lens model
    python durable_crash_script.py <dir> recover              # DBOS recovers the review; exits 0

GitHub is faked to <dir>/github.json so its state survives the crash, like the real one would.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from time import time

from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from autoreviewer.github import Conclusion, Repo, ReviewComment, ReviewEvent
from autoreviewer.review import durable
from autoreviewer.review.agents import correctness_agent, verifier_agent
from autoreviewer.review.deadline import MAX_REVIEW_S
from autoreviewer.web.routing import PullRequestUpdated
from tests.review_fakes import FakeGitHub, GitRepo, finding, job_deps, make_repo, verifier_model

DIR = Path(sys.argv[1])
MODE = sys.argv[2]


class PersistentGitHub(FakeGitHub):
    """FakeGitHub whose writes are saved to disk after every call."""

    def _save(self) -> None:
        reviews = [{**r, "comments": [asdict(c) for c in r["comments"]]} for r in self.reviews]
        (DIR / "github.json").write_text(json.dumps({"checks": self.checks, "reviews": reviews}))

    @classmethod
    def load(cls, repo: GitRepo) -> PersistentGitHub:
        github = cls(repo)
        if (state := DIR / "github.json").exists():
            saved = json.loads(state.read_text())
            github.checks, github.reviews = saved["checks"], saved["reviews"]
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


def crashing_lens(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    if MODE == "enqueue-and-crash":
        os._exit(9)  # a hard crash in the middle of the review: no cleanup, no exception handling
    return ModelResponse(
        parts=[
            ToolCallPart(
                tool_name=info.output_tools[0].name,
                args={"findings": [finding(5)], "overall_correct": True, "reviewed": []},
            )
        ]
    )


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
    github = PersistentGitHub.load(repo)
    deps = job_deps(github, DIR)
    deps = type(deps)(**{**deps.__dict__, "agents": lambda root: (
        correctness_agent(FunctionModel(crashing_lens), root), verifier_agent(verifier_model(lambda f: True), root)
    )})  # fmt: skip
    if MODE == "recover-after-deadline":
        # Advance only the review clock: simulate downtime as long as the whole budget, without sleeping.
        durable.time = lambda: time() + MAX_REVIEW_S
    durable.configure(DIR / "dbos.sqlite")
    await durable.launch(deps, concurrency=2)
    try:
        if MODE == "enqueue-and-crash":
            await (await durable.enqueue(pull)).get_result()  # never returns: the lens kills the process
        else:
            # launch() recovered the interrupted workflow; wait for it by its deterministic id.
            from dbos import DBOS

            handle = await DBOS.retrieve_workflow_async(durable.workflow_id(pull))
            await asyncio.wait_for(handle.get_result(), timeout=60)
    finally:
        durable.shutdown()


asyncio.run(main())
