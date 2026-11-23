"""Shared fakes for the review tests: a real local git repo, a scripted GitHub, scripted models."""

from __future__ import annotations

import asyncio
import json
import subprocess
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, ToolCallPart, UserPromptPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from autoreviewer.github import (
    CommitNotFound,
    Comparison,
    Conclusion,
    PostedComment,
    PostedReview,
    PullRequestState,
    Reaction,
    Repo,
    ReviewComment,
    ReviewEvent,
    ReviewRejected,
    ReviewState,
    ReviewThread,
    ThreadComment,
)
from autoreviewer.review.agents import (
    correctness_agent,
    lens_pair,
    simplify_agent,
    simplify_checker_agent,
    verifier_agent,
)
from autoreviewer.review.job import ReviewDeps
from autoreviewer.review.schemas import LensKind, LensReport, Verification
from autoreviewer.review.simplify import Simplifier, SimplifierFactory
from autoreviewer.review.workspace import Workspace, fetch

BOT = "example-reviewer[bot]"

BASE_SOURCE = "def average(xs):\n    total = 0\n    for x in xs:\n        total += x\n    return total / len(xs)\n"
HEAD_SOURCE = "def average(xs):\n    total = 0\n    for x in xs:\n        total += x\n    return total / len(xs) - 1\n"


_STATES: dict[str, ReviewState] = {
    "APPROVE": "APPROVED",
    "REQUEST_CHANGES": "CHANGES_REQUESTED",
    "COMMENT": "COMMENTED",
}


@dataclass(frozen=True)
class GitRepo:
    path: Path
    base: str
    head: str

    @property
    def url(self) -> str:
        return f"file://{self.path}"


def git(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@t", *args], cwd=cwd, check=True, capture_output=True, text=True
    ).stdout.strip()


def make_repo(path: Path) -> GitRepo:
    """base: stats.py + uv.lock + a symlink. head: the seeded bug, a lockfile bump, a new file."""
    path.mkdir(parents=True)
    git(path, "init", "-q", "-b", "main")
    git(path, "config", "uploadpack.allowAnySHA1InWant", "true")  # GitHub serves any SHA; mimic it
    (path / "stats.py").write_text(BASE_SOURCE)
    (path / "uv.lock").write_text("version = 1\n")
    (path / "secrets").symlink_to("/etc/passwd")
    git(path, "add", "-A")
    git(path, "commit", "-qm", "base")
    base = git(path, "rev-parse", "HEAD")
    (path / "stats.py").write_text(HEAD_SOURCE)
    (path / "uv.lock").write_text("version = 2\n")
    (path / "notes.py").write_text("GREETING = 'hi'\n")
    git(path, "add", "-A")
    git(path, "commit", "-qm", "head")
    return GitRepo(path, base, git(path, "rev-parse", "HEAD"))


def add_commit(repo: GitRepo, files: dict[str, str], *, branch_from: str | None = None) -> str:
    """Commit `files` on top of main (or on a new branch from `branch_from`, for a diverged history)."""
    if branch_from is not None:
        git(repo.path, "checkout", "-q", "-B", f"side-{branch_from[:7]}", branch_from)
    for name, content in files.items():
        (repo.path / name).write_text(content)
    git(repo.path, "add", "-A")
    git(repo.path, "commit", "-qm", "more")
    sha = git(repo.path, "rev-parse", "HEAD")
    git(repo.path, "checkout", "-q", "main")
    return sha


class LocalSource:
    """RepoSource over a local repository: the real fetch() path, no network."""

    def __init__(self, repo: GitRepo) -> None:
        self._repo = repo

    async def checkout(self, repo: Repo, *, head: str, base: str, dest: Path, previous: str | None = None) -> Workspace:
        return await fetch(dest, url=self._repo.url, head=head, base=base, previous=previous)


def finding(
    line: int = 5, *, path: str = "stats.py", title: str = "Mean is off by one", priority: int = 1, confidence: int = 9
) -> dict[str, Any]:
    return {
        "path": path,
        "line": line,
        "priority": priority,
        "confidence": confidence,
        "title": title,
        "explanation": "average([2, 4]) returns 2 instead of 3 because of the trailing `- 1`.",
        "suggestion": "Drop the `- 1`.",
    }


def _final(info: AgentInfo, args: dict[str, Any]) -> ModelResponse:
    return ModelResponse(parts=[ToolCallPart(tool_name=info.output_tools[0].name, args=args)])


def _prompt(messages: list[ModelMessage]) -> str:
    return "".join(
        part.content
        for message in messages
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, UserPromptPart) and isinstance(part.content, str)
    )


async def never_answers(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    """A model call that never returns: work the deadline has to cut off."""
    await asyncio.Event().wait()
    raise AssertionError("cancelled before this")


def lens_model(findings: list[dict[str, Any]]) -> FunctionModel:
    return FunctionModel(
        lambda messages, info: _final(info, {"findings": findings, "overall_correct": not findings, "reviewed": []})
    )


def verifier_model(
    confirm: Callable[[dict[str, Any]], bool],
    priority: Callable[[dict[str, Any]], int | None] = lambda f: None,
    explanation: Callable[[dict[str, Any]], str | None] = lambda f: None,
    *,
    stuck: Callable[[dict[str, Any]], bool] = lambda f: False,
) -> FunctionModel:
    """Confirms what `confirm` says; a finding `stuck` picks never gets its verdict."""

    async def decide(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        reported = json.loads(_prompt(messages).split("```json\n", 1)[1].split("\n```", 1)[0])
        if stuck(reported):
            return await never_answers(messages, info)
        ok = confirm(reported)
        verdict = {"confirmed": ok, "reason": "checked the code" if ok else "not a real bug"}
        if ok and (adjusted := priority(reported)) is not None:
            verdict["priority"] = adjusted
        if ok and (corrected := explanation(reported)) is not None:
            verdict["explanation"] = corrected
        return _final(info, verdict)

    return FunctionModel(decide)


def agents(
    findings: list[dict[str, Any]],
    confirm: Callable[[dict[str, Any]], bool] = lambda f: True,
    priority: Callable[[dict[str, Any]], int | None] = lambda f: None,
    explanation: Callable[[dict[str, Any]], str | None] = lambda f: None,
) -> Callable[[Workspace], tuple[Agent[None, LensReport], Agent[None, Verification]]]:
    return lambda ws: (
        correctness_agent(lens_model(findings), ws),
        verifier_agent(verifier_model(confirm, priority, explanation), ws),
    )


@dataclass
class FakeGitHub:
    """Implements PullRequests against a GitRepo; records every write."""

    repo: GitRepo
    head_shas: list[str] = field(default_factory=list[str])
    """Successive heads `get_pull` reports (default: the repo's head); the last one repeats."""
    pr_state: str = "open"
    reject_inline: bool = False
    """Make create_review raise ReviewRejected when it has comments, like GitHub's 422."""
    reviews: list[dict[str, Any]] = field(default_factory=list[dict[str, Any]])
    checks: list[dict[str, Any]] = field(default_factory=list[dict[str, Any]])
    reactions: list[tuple[int, str]] = field(default_factory=list[tuple[int, str]])
    review_comments: list[dict[str, Any]] = field(default_factory=list[dict[str, Any]])
    fail_review_comments: bool = False
    resolved_threads: set[str] = field(default_factory=set[str])
    """Threads a person resolved (the bot can't)."""
    settled_threads: set[str] = field(default_factory=set[str])
    """Threads whose first comment is minimized as resolved (by the bot)."""
    hidden_threads: set[str] = field(default_factory=set[str])
    """Threads whose first comment a moderator minimized for another reason (spam, off-topic)."""
    fail_threads: bool = False
    """Make resolving or reopening a thread fail, like GitHub's GraphQL API on a bad day."""
    fail_posted_comments: int = 0
    fail_dismissals: bool = False
    """Make dismissing a review fail, like a protected branch that limits who may dismiss."""
    """How many reads of the PR's comments fail (like a transient 5xx) before they work again."""
    comment_delay: float = 0.0
    """Inline comments posted on their own, as findings are confirmed."""
    check_updates: list[dict[str, Any]] = field(default_factory=list[dict[str, Any]])
    fail_check_updates: bool = False
    check_update_delay: float = 0.0
    fail_start_check: bool = False
    check_attempts: int = 0
    fail_get_pull: bool = False
    fail_reactions: bool = False

    async def get_pull(self, repo: Repo, number: int) -> PullRequestState:
        if self.fail_get_pull:
            raise RuntimeError("GitHub is down")
        heads = self.head_shas or [self.repo.head]
        head = heads.pop(0) if len(heads) > 1 else heads[0]
        return PullRequestState(
            head_sha=head, base_sha=self.repo.base, state=self.pr_state, draft=False, title="Add average()", body=""
        )

    async def compare(self, repo: Repo, base_sha: str, head_sha: str) -> Comparison:
        """Real git ancestry on the fixture repo, like GitHub's compare API (a 404 for unknown commits)."""
        try:
            merge_base = git(self.repo.path, "merge-base", base_sha, head_sha)
        except subprocess.CalledProcessError as error:
            raise CommitNotFound(f"{base_sha[:7]}...{head_sha[:7]}") from error
        if base_sha == head_sha:
            status = "identical"
        elif merge_base == base_sha:
            status = "ahead"
        elif merge_base == head_sha:
            status = "behind"
        else:
            status = "diverged"
        return Comparison(merge_base=merge_base, status=status)

    async def start_check(self, repo: Repo, head_sha: str) -> int:
        self.check_attempts += 1
        if self.fail_start_check:
            raise RuntimeError("GitHub is down")
        self.checks.append({"status": "in_progress", "head_sha": head_sha})
        return len(self.checks)

    async def update_check(self, repo: Repo, check_id: int, *, title: str, summary: str, text: str) -> None:
        await asyncio.sleep(self.check_update_delay)
        if self.fail_check_updates:
            raise RuntimeError("secondary rate limit")
        self.check_updates.append({"check_id": check_id, "title": title, "summary": summary, "text": text})

    async def finish_check(
        self, repo: Repo, check_id: int, *, conclusion: Conclusion, title: str, summary: str, text: str = ""
    ) -> None:
        self.checks[check_id - 1].update(
            status="completed", conclusion=conclusion, title=title, summary=summary, text=text
        )

    async def create_review(
        self, repo: Repo, number: int, *, commit_sha: str, event: ReviewEvent, body: str, comments: list[ReviewComment]
    ) -> None:
        if self.reject_inline and comments:
            raise ReviewRejected("Unprocessable Entity: line must be part of the diff")
        self.reviews.append(
            {"commit_sha": commit_sha, "event": event, "body": body, "comments": comments, "author": BOT}
        )

    async def react(self, repo: Repo, comment_id: int, content: Reaction) -> None:
        if self.fail_reactions:
            raise RuntimeError("GitHub is down")
        self.reactions.append((comment_id, content))

    async def create_review_comment(self, repo: Repo, number: int, *, commit_sha: str, comment: ReviewComment) -> None:
        await asyncio.sleep(self.comment_delay)
        if self.fail_review_comments:
            raise RuntimeError("secondary rate limit")
        if self.reject_inline:
            raise ReviewRejected("Unprocessable Entity: line must be part of the diff")
        self.review_comments.append({"commit_sha": commit_sha, "comment": comment, "author": BOT})

    @property
    def inline_comments(self) -> list[ReviewComment]:
        """Every inline comment, posted on its own or inside a review."""
        return [c["comment"] for c in self.review_comments] + [c for r in self.reviews for c in r["comments"]]

    async def thread(self, repo: Repo, number: int, root_id: int) -> list[ThreadComment]:
        """Comment ids are 1-based positions in `review_comments`; a reply records the comment it answers."""
        return [
            ThreadComment(i, c["author"], c["comment"].body, c["comment"].path)
            for i, c in enumerate(self.review_comments, start=1)
            if i == root_id or c.get("in_reply_to") == root_id
        ]

    async def reply(self, repo: Repo, number: int, root_id: int, body: str) -> None:
        self.say(BOT, root_id, body)

    def say(self, author: str, root_id: int, body: str) -> int:
        """A reply in the thread under comment `root_id`, by anyone; returns its id."""
        root = self.review_comments[root_id - 1]["comment"]
        self.review_comments.append(
            {"commit_sha": self.repo.head, "comment": ReviewComment(root.path, root.line, body), "author": author,
             "in_reply_to": root_id}
        )  # fmt: skip
        return len(self.review_comments)

    async def review_threads(self, repo: Repo, number: int) -> list[ReviewThread]:
        """One thread per comment that starts one, named after it."""
        return [
            ReviewThread(
                f"T{i}",
                i,
                f"T{i}" in self.resolved_threads,
                f"N{i}",
                f"T{i}" in self.settled_threads | self.hidden_threads,
                f"T{i}" in self.settled_threads,
            )
            for i, c in enumerate(self.review_comments, start=1)
            if c.get("in_reply_to") is None
        ]

    async def posted_reviews(self, repo: Repo, number: int, *, author: str) -> list[PostedReview]:
        """The verdicts, ids 1-based positions in `reviews` (the empty wrappers never decide anything)."""
        return [
            PostedReview(i, r["commit_sha"], "DISMISSED" if "dismissed" in r else _STATES[r["event"]], r["body"])
            for i, r in enumerate(self.reviews, start=1)
            if r["author"] == author
        ]

    async def dismiss_review(self, repo: Repo, number: int, review_id: int, *, message: str) -> None:
        if self.fail_dismissals:
            raise RuntimeError("403 Forbidden")
        self.reviews[review_id - 1]["dismissed"] = message

    async def set_settled(self, repo: Repo, thread: ReviewThread, *, settled: bool) -> None:
        if self.fail_threads:
            raise RuntimeError("502 Bad Gateway")
        if settled:
            self.settled_threads.add(thread.id)
        else:  # like GitHub: unminimizing shows the comment, whoever hid it and why
            self.settled_threads.discard(thread.id)
            self.hidden_threads.discard(thread.id)

    async def posted_comments(self, repo: Repo, number: int, *, author: str) -> list[PostedComment]:
        """Like GitHub: a comment stays on its line on later heads unless marked `outdated`."""
        if self.fail_posted_comments:
            self.fail_posted_comments -= 1
            raise RuntimeError("502 Bad Gateway")
        return [
            PostedComment(
                c["comment"].path,
                None if c.get("outdated") else c["comment"].line,
                c["comment"].body,
                i,
                c.get("in_reply_to"),
            )
            for i, c in enumerate(self.review_comments, start=1)
            if c["author"] == author
        ]

    async def review_bodies(self, repo: Repo, number: int, *, commit_sha: str, author: str) -> list[str]:
        """Like GitHub: each single inline comment comes wrapped in an empty COMMENTED review of its own."""
        verdicts = [
            r.body for r in await self.posted_reviews(repo, number, author=author) if r.commit_sha == commit_sha
        ]
        wrappers = ["" for c in self.review_comments if c["commit_sha"] == commit_sha and c["author"] == author]
        return verdicts + wrappers


def job_deps(
    github: FakeGitHub,
    work_dir: Path,
    findings: list[dict[str, Any]] | None = None,
    *,
    confirm: Callable[[dict[str, Any]], bool] = lambda f: True,
    priority: Callable[[dict[str, Any]], int | None] = lambda f: None,
) -> ReviewDeps:
    return ReviewDeps(
        github=github,
        source=LocalSource(github.repo),
        agents=agents([] if findings is None else findings, confirm=confirm, priority=priority),
        work_dir=work_dir,
        bot_login=BOT,
        models_note="test models",
    )


def simplifier(
    suggestions: list[dict[str, Any]],
    confirm: Callable[[str], bool] = lambda prompt: True,
    *,
    crash: bool = False,
    prompts: list[str] | None = None,
) -> SimplifierFactory:
    """Proposes `suggestions`; the checker confirms when `confirm(its prompt)` says so."""

    def propose(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if crash:
            raise RuntimeError("model server down")
        if prompts is not None:
            prompts.append(_prompt(messages))
        return _final(info, {"suggestions": suggestions})

    def check(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        ok = confirm(_prompt(messages))
        return _final(info, {"confirmed": ok, "reason": "same behaviour" if ok else "changes behaviour"})

    return lambda ws: Simplifier(
        simplify_agent(FunctionModel(propose), ws), simplify_checker_agent(FunctionModel(check), ws)
    )


def lens_agents(
    findings: list[dict[str, Any]],
    confirm: Callable[[dict[str, Any]], bool] = lambda f: True,
    *,
    kind: LensKind = "security",
    prompts: list[str] | None = None,
    crash: bool = False,
) -> Callable[[Workspace], tuple[Agent[None, LensReport], Agent[None, Verification]]]:
    """A fake extra lens: reports `findings`, records its prompts, or crashes."""

    def lens(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if crash:
            raise RuntimeError("model server down")
        if prompts is not None:
            prompts.append(_prompt(messages))
        return _final(info, {"findings": findings, "overall_correct": not findings, "reviewed": []})

    return lambda ws: lens_pair(kind, FunctionModel(lens), verifier_model(confirm), ws)
