"""The only module that talks to GitHub. Models never get here: they return typed output, code posts it."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal, Protocol, cast

from githubkit import AppAuthStrategy, GitHub, TokenAuthStrategy, UnauthAuthStrategy
from githubkit.exception import RequestFailed

CHECK_NAME = "autoreviewer"

Conclusion = Literal["success", "failure", "neutral"]
ReviewEvent = Literal["APPROVE", "REQUEST_CHANGES", "COMMENT"]
ReviewState = Literal["APPROVED", "CHANGES_REQUESTED", "COMMENTED", "DISMISSED", "PENDING"]


@dataclass(frozen=True)
class Repo:
    installation_id: int
    owner: str
    name: str

    @property
    def clone_url(self) -> str:
        return f"https://github.com/{self.owner}/{self.name}.git"


@dataclass(frozen=True)
class PullRequestState:
    head_sha: str
    base_sha: str
    state: str
    draft: bool
    title: str
    body: str


@dataclass(frozen=True)
class Comparison:
    merge_base: str
    status: str
    """GitHub's compare status of head against base: "ahead" (head adds commits on top of base),
    "identical", "behind", or "diverged" (e.g. after a force-push or rebase)."""


@dataclass(frozen=True)
class ReviewComment:
    path: str
    line: int
    body: str
    start_line: int | None = None
    """For a comment spanning lines start_line..line (e.g. a multi-line suggestion)."""


@dataclass(frozen=True)
class PostedComment:
    """An inline comment already on a PR, placed where GitHub shows it on the PR's current head."""

    path: str
    line: int | None
    """None once the code it was on has changed (GitHub shows it as outdated)."""
    body: str
    id: int
    in_reply_to: int | None = None
    """The thread's first comment, for a reply; None for a comment that starts a thread."""


@dataclass(frozen=True)
class PostedReview:
    """A review already on a PR, as GitHub lists it."""

    id: int
    commit_sha: str
    state: ReviewState
    body: str


@dataclass(frozen=True)
class ThreadComment:
    """One comment in an inline comment thread."""

    id: int
    author: str
    body: str
    path: str = ""


@dataclass(frozen=True)
class ReviewThread:
    """An inline comment thread, as GitHub's GraphQL API names it."""

    id: str
    root_comment_id: int
    resolved: bool
    """Resolved on GitHub, by a person: a GitHub App can't resolve threads without write access to the code."""
    root_node_id: str
    hidden: bool
    """Its first comment is minimized, for any reason, by the bot or by a moderator."""
    settled: bool
    """Minimized as resolved: how the bot marks a thread done, since it can't resolve it."""


_REVIEW_THREADS = """query($owner: String!, $name: String!, $number: Int!, $after: String) {
  repository(owner: $owner, name: $name) { pullRequest(number: $number) {
    reviewThreads(first: 100, after: $after) {
      pageInfo { hasNextPage endCursor }
      nodes { id isResolved comments(first: 1) { nodes { databaseId id isMinimized minimizedReason } } }
    }
  } }
}"""
_SET_SETTLED = {
    True: "mutation($id: ID!) { minimizeComment(input: {subjectId: $id, classifier: RESOLVED}) { clientMutationId } }",
    False: "mutation($id: ID!) { unminimizeComment(input: {subjectId: $id}) { clientMutationId } }",
}
"""Minimize a thread's first comment as resolved, or show it again. Resolving the thread itself needs
`contents: write` (Nov 19: FORBIDDEN with `pull_requests: write`), which the bot doesn't get."""


class CommitNotFound(Exception):
    """GitHub doesn't know one of the commits, e.g. an old head garbage-collected after a force-push."""


class ReviewRejected(Exception):
    """GitHub refused the review (422), typically because a comment's line isn't in its diff view."""


class PullRequestReader(Protocol):
    """What reviewing a PR needs from GitHub's API: read-only. Code itself comes from git (workspace.py)."""

    async def get_pull(self, repo: Repo, number: int) -> PullRequestState: ...

    async def compare(self, repo: Repo, base_sha: str, head_sha: str) -> Comparison: ...


Reaction = Literal["eyes", "+1"]


class Commands(PullRequestReader, Protocol):
    """What answering a comment command needs: read the PR, and acknowledge the comment."""

    async def react(self, repo: Repo, comment_id: int, content: Reaction) -> None: ...


class PullRequests(PullRequestReader, Protocol):
    """Everything the bot's review job needs: reading, plus check runs and posting reviews."""

    async def start_check(self, repo: Repo, head_sha: str) -> int: ...

    async def update_check(self, repo: Repo, check_id: int, *, title: str, summary: str, text: str) -> None:
        """Show progress on a check run that's still in progress."""
        ...

    async def finish_check(
        self, repo: Repo, check_id: int, *, conclusion: Conclusion, title: str, summary: str, text: str = ""
    ) -> None: ...

    async def create_review(
        self, repo: Repo, number: int, *, commit_sha: str, event: ReviewEvent, body: str, comments: list[ReviewComment]
    ) -> None: ...

    async def review_bodies(self, repo: Repo, number: int, *, commit_sha: str, author: str) -> list[str]:
        """The bodies of `author`'s reviews of this exact commit, including the empty COMMENTED reviews GitHub
        creates around single inline comments."""
        ...

    async def create_review_comment(self, repo: Repo, number: int, *, commit_sha: str, comment: ReviewComment) -> None:
        """One inline comment outside any review: a finding posted the moment it's confirmed."""
        ...

    async def posted_comments(self, repo: Repo, number: int, *, author: str) -> list[PostedComment]:
        """Every inline comment `author` left on the PR, where GitHub places it on the current head."""
        ...

    async def reply(self, repo: Repo, number: int, root_id: int, body: str) -> None:
        """A reply in the thread whose first comment is `root_id`."""
        ...

    async def review_threads(self, repo: Repo, number: int) -> list[ReviewThread]: ...

    async def set_settled(self, repo: Repo, thread: ReviewThread, *, settled: bool) -> None: ...


class Conversations(PullRequests, Protocol):
    """What answering a reply in one of the bot's threads needs: read the thread and reply in it, plus what a
    withdrawal needs to reconsider the verdict (reviews and check runs)."""

    async def thread(self, repo: Repo, number: int, root_id: int) -> list[ThreadComment]:
        """The thread's first comment, then its replies, oldest first."""
        ...

    async def posted_reviews(self, repo: Repo, number: int, *, author: str) -> list[PostedReview]:
        """`author`'s reviews of the PR, every commit, oldest first."""
        ...

    async def dismiss_review(self, repo: Repo, number: int, review_id: int, *, message: str) -> None: ...


async def _get_pull(github: GitHub, repo: Repo, number: int) -> PullRequestState:
    pull = (await github.rest.pulls.async_get(repo.owner, repo.name, number)).parsed_data
    return PullRequestState(
        head_sha=pull.head.sha,
        base_sha=pull.base.sha,
        state=pull.state,
        draft=bool(pull.draft),
        title=pull.title,
        body=pull.body or "",
    )


async def _compare(github: GitHub, repo: Repo, base_sha: str, head_sha: str) -> Comparison:
    """Merge base (what GitHub diffs a PR against) and how head relates to base, in one call."""
    # per_page=1 keeps the response small; only two fields are read.
    try:
        response = await github.rest.repos.async_compare_commits(
            repo.owner, repo.name, f"{base_sha}...{head_sha}", per_page=1
        )
    except RequestFailed as error:
        if error.response.status_code == 404:
            raise CommitNotFound(f"{repo.owner}/{repo.name}: {base_sha[:7]}...{head_sha[:7]}") from error
        raise
    data = response.json()
    return Comparison(merge_base=str(data["merge_base_commit"]["sha"]), status=str(data["status"]))


class PublicGitHub:
    """Read-only access to any repository you can see: anonymous for public repos, or with a token.

    For on-demand reviews of PRs where the App isn't installed; it can't post anything.
    """

    def __init__(self, token: str | None = None) -> None:
        self._github: GitHub = GitHub(TokenAuthStrategy(token) if token else UnauthAuthStrategy())

    async def get_pull(self, repo: Repo, number: int) -> PullRequestState:
        return await _get_pull(self._github, repo, number)

    async def compare(self, repo: Repo, base_sha: str, head_sha: str) -> Comparison:
        return await _compare(self._github, repo, base_sha, head_sha)


class GitHubApp:
    """Acts as one installation per call; githubkit mints and refreshes the installation tokens."""

    def __init__(self, github: GitHub[AppAuthStrategy]) -> None:
        self._github = github

    @classmethod
    def from_key(cls, app_id: int, private_key: str) -> GitHubApp:
        return cls(GitHub(AppAuthStrategy(app_id, private_key)))

    def _as_installation(self, installation_id: int) -> GitHub:
        # Callers must hold the result in a local: githubkit's `rest` namespace only keeps a weak
        # reference, so chaining off this temporary lets CPython collect it mid-call (M0 live test).
        return self._github.with_auth(self._github.auth.as_installation(installation_id))

    async def read_token(self, repo: Repo) -> str:
        """A short-lived token that can only read this one repository's contents (for `git fetch`)."""
        response = await self._github.rest.apps.async_create_installation_access_token(
            repo.installation_id, data={"repositories": [repo.name], "permissions": {"contents": "read"}}
        )
        return str(response.json()["token"])

    async def get_pull(self, repo: Repo, number: int) -> PullRequestState:
        github = self._as_installation(repo.installation_id)
        return await _get_pull(github, repo, number)

    async def compare(self, repo: Repo, base_sha: str, head_sha: str) -> Comparison:
        github = self._as_installation(repo.installation_id)
        return await _compare(github, repo, base_sha, head_sha)

    async def start_check(self, repo: Repo, head_sha: str) -> int:
        github = self._as_installation(repo.installation_id)
        response = await github.rest.checks.async_create(
            repo.owner,
            repo.name,
            data={"name": CHECK_NAME, "head_sha": head_sha, "status": "in_progress"},
        )
        # Only the id is needed; validating the whole check-run schema would only add brittleness.
        return int(response.json()["id"])

    async def update_check(self, repo: Repo, check_id: int, *, title: str, summary: str, text: str) -> None:
        github = self._as_installation(repo.installation_id)
        await github.rest.checks.async_update(
            repo.owner, repo.name, check_id, data={"output": {"title": title, "summary": summary, "text": text}}
        )

    async def finish_check(
        self, repo: Repo, check_id: int, *, conclusion: Conclusion, title: str, summary: str, text: str = ""
    ) -> None:
        github = self._as_installation(repo.installation_id)
        await github.rest.checks.async_update(
            repo.owner,
            repo.name,
            check_id,
            data={
                "status": "completed",
                "conclusion": conclusion,
                "output": {"title": title, "summary": summary, "text": text},
            },
        )

    async def create_review(
        self, repo: Repo, number: int, *, commit_sha: str, event: ReviewEvent, body: str, comments: list[ReviewComment]
    ) -> None:
        github = self._as_installation(repo.installation_id)
        try:
            await github.rest.pulls.async_create_review(
                repo.owner,
                repo.name,
                number,
                data={
                    "commit_id": commit_sha,
                    "event": event,
                    "body": body,
                    "comments": [
                        {
                            "path": c.path,
                            "line": c.line,
                            "side": "RIGHT",
                            "body": c.body,
                            "start_line": c.start_line,
                            "start_side": "RIGHT",
                        }
                        if c.start_line is not None and c.start_line < c.line
                        else {"path": c.path, "line": c.line, "side": "RIGHT", "body": c.body}
                        for c in comments
                    ],
                },
            )
        except RequestFailed as error:
            if error.response.status_code == 422:
                raise ReviewRejected(str(error)) from error
            raise

    async def react(self, repo: Repo, comment_id: int, content: Reaction) -> None:
        github = self._as_installation(repo.installation_id)
        await github.rest.reactions.async_create_for_issue_comment(
            repo.owner, repo.name, comment_id, data={"content": content}
        )

    async def create_review_comment(self, repo: Repo, number: int, *, commit_sha: str, comment: ReviewComment) -> None:
        github = self._as_installation(repo.installation_id)
        c = comment
        try:
            await github.rest.pulls.async_create_review_comment(
                repo.owner,
                repo.name,
                number,
                data={
                    "body": c.body,
                    "commit_id": commit_sha,
                    "path": c.path,
                    "line": c.line,
                    "side": "RIGHT",
                    "start_line": c.start_line,
                    "start_side": "RIGHT",
                }
                if c.start_line is not None and c.start_line < c.line
                else {"body": c.body, "commit_id": commit_sha, "path": c.path, "line": c.line, "side": "RIGHT"},
            )
        except RequestFailed as error:
            if error.response.status_code == 422:
                raise ReviewRejected(str(error)) from error
            raise

    async def thread(self, repo: Repo, number: int, root_id: int) -> list[ThreadComment]:
        comments = [
            c for c in await self._review_comments(repo, number) if root_id in (c.get("id"), c.get("in_reply_to_id"))
        ]
        found = [
            ThreadComment(
                int(c["id"]),
                str((c.get("user") or {}).get("login") or ""),
                str(c.get("body") or ""),
                str(c.get("path") or ""),
            )
            for c in comments
        ]
        return sorted(found, key=lambda comment: comment.id != root_id)  # stable: the root, then replies in order

    async def reply(self, repo: Repo, number: int, root_id: int, body: str) -> None:
        github = self._as_installation(repo.installation_id)
        await github.rest.pulls.async_create_reply_for_review_comment(
            repo.owner, repo.name, number, root_id, data={"body": body}
        )

    async def review_threads(self, repo: Repo, number: int) -> list[ReviewThread]:
        github = self._as_installation(repo.installation_id)
        threads: list[ReviewThread] = []
        after: str | None = None
        while True:
            variables = {"owner": repo.owner, "name": repo.name, "number": number, "after": after}
            page = (await github.async_graphql(_REVIEW_THREADS, variables))["repository"]["pullRequest"][
                "reviewThreads"
            ]
            threads += [
                ReviewThread(
                    str(node["id"]),
                    int(root["databaseId"]),
                    bool(node["isResolved"]),
                    str(root["id"]),
                    bool(root["isMinimized"]),
                    # minimizedReason is a String, lowercased, not the classifier enum the mutation takes
                    # ("RESOLVED"): live on #44, Nov 19, comment 4635273385 read minimizedReason=resolved.
                    bool(root["isMinimized"]) and root.get("minimizedReason") == "resolved",
                )
                for node in page["nodes"]
                if (root := next(iter(node["comments"]["nodes"]), None))
            ]
            if not page["pageInfo"]["hasNextPage"]:
                return threads
            after = page["pageInfo"]["endCursor"]

    async def set_settled(self, repo: Repo, thread: ReviewThread, *, settled: bool) -> None:
        github = self._as_installation(repo.installation_id)
        await github.async_graphql(_SET_SETTLED[settled], {"id": thread.root_node_id})

    async def posted_comments(self, repo: Repo, number: int, *, author: str) -> list[PostedComment]:
        return [
            PostedComment(
                str(c.get("path") or ""),
                c.get("line"),
                str(c.get("body") or ""),
                int(c.get("id") or 0),
                c.get("in_reply_to_id"),
            )
            for c in await self._review_comments(repo, number)
            if (c.get("user") or {}).get("login") == author
        ]

    async def _review_comments(self, repo: Repo, number: int) -> list[Mapping[str, Any]]:
        """Every inline comment on the PR as GitHub sends it, oldest first, all pages. Raw JSON, not githubkit's
        models: callers read only a few fields."""
        github = self._as_installation(repo.installation_id)
        found: list[Mapping[str, Any]] = []
        page = 1
        while True:
            response = await github.rest.pulls.async_list_review_comments(
                repo.owner, repo.name, number, per_page=100, page=page
            )
            comments = response.json()
            found += comments
            if len(comments) < 100:
                return found
            page += 1

    async def review_bodies(self, repo: Repo, number: int, *, commit_sha: str, author: str) -> list[str]:
        return [r.body for r in await self.posted_reviews(repo, number, author=author) if r.commit_sha == commit_sha]

    async def posted_reviews(self, repo: Repo, number: int, *, author: str) -> list[PostedReview]:
        github = self._as_installation(repo.installation_id)
        found: list[PostedReview] = []
        page = 1
        while True:
            # Raw JSON, not githubkit's models: only a few fields matter and the full schema is large.
            response = await github.rest.pulls.async_list_reviews(
                repo.owner, repo.name, number, per_page=100, page=page
            )
            reviews = response.json()
            found += [
                PostedReview(
                    int(r.get("id") or 0),
                    str(r.get("commit_id") or ""),
                    cast(ReviewState, str(r.get("state") or "")),  # one GitHub adds later just never matches
                    str(r.get("body") or ""),
                )
                for r in reviews
                if (r.get("user") or {}).get("login") == author
            ]
            if len(reviews) < 100:
                return found
            page += 1

    async def dismiss_review(self, repo: Repo, number: int, review_id: int, *, message: str) -> None:
        github = self._as_installation(repo.installation_id)
        await github.rest.pulls.async_dismiss_review(repo.owner, repo.name, number, review_id, message=message)
