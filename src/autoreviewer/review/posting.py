"""Getting findings onto the PR: each one the moment it's confirmed, then the verdict.

Findings are posted as individual inline comments as they're confirmed, so the author (often another bot)
can start fixing while the review continues. The verdict comes last, as one review (REQUEST_CHANGES or
APPROVE) whose inline comments are only the findings that couldn't be posted earlier.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
from collections.abc import Iterable, Mapping
from dataclasses import replace
from typing import cast

from autoreviewer.github import (
    Conversations,
    PostedComment,
    PullRequests,
    Repo,
    ReviewComment,
    ReviewEvent,
    ReviewRejected,
    ReviewState,
    ReviewThread,
)
from autoreviewer.review.deadline import Deadline
from autoreviewer.review.pipeline import Outcome, Withdrawal, blocks, blocks_at, normal_title, same_defect
from autoreviewer.review.render import comment_body, posted_title, review_body
from autoreviewer.review.schemas import Finding

logger = logging.getLogger("autoreviewer.posting")

VERDICT_MARKER = "<!-- autoreviewer:verdict -->"
"""Ends every final verdict (approve or request changes), telling it apart from the empty reviews GitHub
creates around single comments, and from a partial one."""
PARTIAL_MARKER = "<!-- autoreviewer:partial -->"
"""Ends a "partially reviewed" COMMENT: it never stops a later full review of the commit from posting."""
CHECK_MARKER = "<!-- autoreviewer:check {} -->"
"""Names the attempt that posted a verdict (its check run's id goes in the braces), on the line before the
marker that ends it. GitHub's check run id is checkpointed: the same across a recovery, new for a fresh attempt."""
_MARKER = re.compile(r"<!-- autoreviewer:([0-9a-f]{16}) p([0-3])(?: ([a-z]+))? -->\Z")
"""Only the marker that ends a comment counts: code appends it last, after any model text or suggestion code,
so a marker an attacker plants in the PR (and gets quoted) can't make a real finding look already posted."""


def fingerprint(head_sha: str, finding: Finding) -> str:
    """Identifies a finding's comment on one commit, so replaying a review never posts it twice. Priority isn't
    part of it (a replay's verifier may rank the same finding differently); the marker records it separately,
    so only a more urgent restatement is posted again."""
    return _fingerprint(head_sha, finding.kind, finding.path, finding.line)


def _fingerprint(head_sha: str, kind: str | None, path: str, line: int | None) -> str:
    return hashlib.sha256(f"{head_sha}|{kind}|{path}|{line}".encode()).hexdigest()[:16]


WITHDRAWN_MARKER = "<!-- autoreviewer:withdrawn -->"
"""The last line of an answer that withdrew its thread's finding after re-checking the code (conversation.py)."""
ANSWERS_MARKER = "<!-- autoreviewer:answers {} -->"
"""Ends the bot's answer to one comment (its id goes in the braces); answer_markers writes it."""
_ANSWER_END = re.compile(
    re.escape(ANSWERS_MARKER).replace(r"\{\}", r"(\d+)") + rf"(\n{re.escape(WITHDRAWN_MARKER)})?\Z"
)
"""Only the markers that end an answer count: code appends them after the model's text, so nothing the model was
led to write can claim an answer or withdraw a finding."""
_VERDICT_STATE = re.compile(
    r"<!-- autoreviewer:state (APPROVE|REQUEST_CHANGES|COMMENT) outside=(\d+) complete=([01]) -->\n"
    + "(?:"
    + re.escape(CHECK_MARKER).replace(r"\{\}", r"(\d+)")
    + r"\n)?"
    + rf"(?:{re.escape(VERDICT_MARKER)}|{re.escape(PARTIAL_MARKER)})\Z"
)
"""The state in a verdict's footer, at the very end of its body (anchored there for the same reason): its
event, how many blocking findings it listed with no thread to withdraw them in, and whether the review covered
the whole change. A withdrawal only ever overturns a complete verdict that requested changes with outside=0."""


def overturnable(verdict: re.Match[str] | None) -> bool:
    """Whether a withdrawal may overturn this verdict state: a complete review that requested changes, with every
    blocking finding in a thread a withdrawal could happen in (outside=0)."""
    return verdict is not None and verdict.group(1, 2, 3) == ("REQUEST_CHANGES", "0", "1")


Placement = tuple[str, int, str]
"""(path, line on the current head, kind): where GitHub shows a finding comment."""


def answer_markers(comment_id: int, *, withdrawn: bool) -> str:
    """What ends an answer to `comment_id` (appended by code, after the model's text)."""
    return ANSWERS_MARKER.format(comment_id) + (f"\n{WITHDRAWN_MARKER}" if withdrawn else "")


def answered(body: str) -> tuple[int, bool] | None:
    """The id of the comment an answer answers, and whether it withdrew the finding, read from the markers that
    end it; None if it doesn't end with them."""
    return (int(match[1]), match[2] is not None) if (match := _ANSWER_END.search(body)) else None


def withdrawn_threads(comments: Iterable[PostedComment]) -> dict[int, PostedComment]:
    """The bot's finding comments whose thread withdrew them, by id: the latest answer decides, so a later answer
    can reinstate a finding."""
    comments = sorted(comments, key=lambda comment: comment.id)
    latest = {c.in_reply_to: answer for c in comments if c.in_reply_to is not None and (answer := answered(c.body))}
    findings = {c.id: c for c in comments if c.in_reply_to is None and finding_marker(c.body)}
    return {root: findings[root] for root, (_, withdrew) in latest.items() if withdrew and root in findings}


def withdrawals(comments: Iterable[PostedComment]) -> tuple[Withdrawal, ...]:
    """The withdrawn findings GitHub still shows on the current head, as a later review should see them."""
    found = []
    for root in withdrawn_threads(comments).values():
        match, title = finding_marker(root.body), posted_title(root.body)
        if root.line is not None and match and match[3] and title is not None:  # outdated, or no kind recorded
            found.append(Withdrawal(root.path, root.line, match[3], int(match[2]), normal_title(title)))
    return tuple(found)


def blocking_findings(comments: Iterable[PostedComment], block_at: int) -> list[tuple[PostedComment, re.Match[str]]]:
    """The bot's finding comments that block a merge, wherever GitHub shows them (outdated too), each with its
    marker. Withdrawn ones included: the caller decides what clears one."""
    return [
        (comment, match)
        for comment in comments
        if comment.in_reply_to is None
        and (match := finding_marker(comment.body))
        and blocks_at(match[3], int(match[2]), block_at)
    ]


def standing_blockers(comments: Iterable[PostedComment], block_at: int) -> list[tuple[PostedComment, re.Match[str]]]:
    """The bot's blocking finding comments GitHub still shows on the current head that their thread hasn't
    withdrawn, each with its marker."""
    comments = list(comments)
    withdrawn = withdrawn_threads(comments)
    return [
        (comment, match)
        for comment, match in blocking_findings(comments, block_at)
        if comment.line is not None and comment.id not in withdrawn
    ]


def finding_marker(body: str) -> re.Match[str] | None:
    """The marker that ends one of the bot's finding comments: (fingerprint, priority, kind or None)."""
    return _MARKER.search(body)


def covered[Key](posted: Mapping[Key, int], key: Key, finding: Finding) -> bool:
    """Whether a comment at least as urgent as `finding` is already posted under `key` (its fingerprint, or where
    GitHub shows a comment of its kind)."""
    return key in posted and posted[key] <= finding.priority


def placement(path: str, line: int, kind: str) -> Placement:
    """Where a comment of one kind sits on the current head: what a finding and a comment GitHub shows are matched
    by, across heads (the fingerprint, by contrast, is per head)."""
    return (path, line, kind)


def inline_comment(finding: Finding, *, head_sha: str, block_at: int) -> ReviewComment:
    body = comment_body(finding, blocking=blocks(finding, block_at))
    return ReviewComment(
        finding.path,
        finding.line,
        f"{body}\n\n<!-- autoreviewer:{fingerprint(head_sha, finding)} p{finding.priority} {finding.kind} -->",
        finding.start_line,
    )


class Publisher:
    """Posts each confirmed inline finding as its own comment, at once. Never raises: a finding that fails to
    post here, or whose turn comes after the checking cutoff, goes into the final review instead. Safe to
    replay: comments already on the PR (found by their fingerprint) aren't posted again."""

    def __init__(
        self,
        github: PullRequests,
        repo: Repo,
        number: int,
        *,
        head_sha: str,
        author: str,
        block_at: int,
        deadline: Deadline | None = None,
    ):
        self._github, self._repo, self._number = github, repo, number
        self._head_sha, self._author, self._block_at = head_sha, author, block_at
        self._deadline = deadline
        self.posted: dict[str, int] = {}
        """Fingerprint -> the most urgent priority posted for it (this review, or found on the PR)."""
        self._findings: list[Finding] = []
        """What this review posted, or found already standing on the PR for the same defect."""
        self._existing: dict[str, int] | None = None
        self._placed: dict[Placement, int] = {}
        """placement() -> the most urgent priority of the comments of that kind GitHub still shows there, from a
        superseded review: its comments cover the same defect on the new head. Only markers that record a kind
        count (older ones can't be placed)."""
        self._lock = asyncio.Lock()

    @property
    def findings(self) -> list[Finding]:
        """What this review's comments say, so the verdict counts them even if the lens behind one failed."""
        return list(self._findings)

    async def __call__(self, finding: Finding) -> None:
        key = fingerprint(self._head_sha, finding)
        # One post at a time, decided and recorded under the lock: two reports of one defect arriving together
        # must not both see "not posted yet". (Posts are few and quick.)
        async with self._lock:
            try:
                existing = self._existing if self._existing is not None else await self._load_existing()
                if covered(self.posted, key, finding):
                    return
                if covered(existing, key, finding):
                    self._findings.append(finding)  # reconfirmed this attempt, even if its lens later fails
                    self.posted[key] = existing[key]
                    return
                where = placement(finding.path, finding.line, finding.kind)
                if covered(self._placed, where, finding):  # posted for an earlier commit, still on this line
                    self.posted[key] = self._placed[where]
                    self._findings.append(finding)  # its comment exists: it stands in for the other lens's too
                    return
                # Already posted as at least as urgent by the other lens: a second comment would only repeat it.
                if any(same_defect(f, finding) and f.priority <= finding.priority for f in self._findings):
                    self.posted[key] = finding.priority
                    return
                if self._deadline is not None and self._deadline.past_cutoff:
                    # Queued behind another post when checking stopped waiting. Each write started now would
                    # come out of the time kept for the verdict, which carries this finding instead.
                    return
                comment = inline_comment(finding, head_sha=self._head_sha, block_at=self._block_at)
                await self._github.create_review_comment(
                    self._repo, self._number, commit_sha=self._head_sha, comment=comment
                )
            except Exception:
                logger.warning(
                    "could not post %s:%s now; it goes in the final review", finding.path, finding.line, exc_info=True
                )
                return
            self.posted[key] = finding.priority
            self._findings.append(finding)  # only once its comment exists can it stand in for a duplicate

    async def withdrawn(self) -> tuple[Withdrawal, ...]:
        """The findings withdrawn in conversation, read afresh (an answer may have come in during the review). Empty,
        and logged, if the PR's comments can't be read: the verdict then counts every finding."""
        try:
            return withdrawals(await self._github.posted_comments(self._repo, self._number, author=self._author))
        except Exception:
            logger.warning("could not read the PR's comments; withdrawals not applied", exc_info=True)
            return ()

    async def standing(self) -> bool:
        """Whether an earlier review of this same commit left a blocking finding standing: a finding comment
        its thread hasn't withdrawn, or one the latest verdict listed with no thread to withdraw it in. The code
        hasn't changed, so a rerun that misses a confirmed finding hasn't cleared it. Without this, asking for
        review after review until one pass misses the finding would turn a blocked commit green.

        Raises if GitHub can't be read: an approval must not rest on not having looked."""
        comments = await self._github.posted_comments(self._repo, self._number, author=self._author)
        if any(
            match[1] == _fingerprint(self._head_sha, match[3], comment.path, comment.line)
            for comment, match in standing_blockers(comments, self._block_at)
        ):
            return True
        bodies = await self._github.review_bodies(
            self._repo, self._number, commit_sha=self._head_sha, author=self._author
        )
        decided = [match for body in bodies if (match := _VERDICT_STATE.search(body)) and match[1] != "COMMENT"]
        return bool(decided) and decided[-1][1] == "REQUEST_CHANGES" and int(decided[-1][2]) > 0

    async def _load_existing(self) -> dict[str, int]:
        """Fingerprints (and placements) of the bot's comments on the PR. Recorded only once the fetch succeeds,
        so a failed fetch is retried with the next finding instead of switching dedupe off for the review."""
        comments = await self._github.posted_comments(self._repo, self._number, author=self._author)
        existing: dict[str, int] = {}
        for comment in comments:
            if not (match := _MARKER.search(comment.body)):
                continue
            fp, priority, kind = match[1], int(match[2]), match[3]
            existing[fp] = min(priority, existing.get(fp, 3))
            if kind and comment.line is not None:
                where = placement(comment.path, comment.line, kind)
                self._placed[where] = min(priority, self._placed.get(where, 3))
        self._existing = existing
        return existing


def published_event(bodies: Iterable[str], check_id: int) -> ReviewEvent | None:
    """The verdict actually posted by this attempt (not another review of the same commit)."""
    for body in bodies:
        if (match := _VERDICT_STATE.search(body)) and match[4] == str(check_id):
            return cast(ReviewEvent, match[1])
    return None


async def post_review(
    github: PullRequests,
    repo: Repo,
    number: int,
    *,
    head_sha: str,
    author: str,
    models_note: str,
    outcome: Outcome,
    posted: Mapping[str, int] | None = None,
    check_id: int | None = None,
) -> ReviewEvent | None:
    """The verdict, computed and never generated: a confirmed finding at the blocking priority blocks the
    merge. Inline findings already posted (`posted` fingerprints) aren't repeated."""
    # A review cut short by the time limit never approves: it hasn't seen the whole change. Its verdict is
    # partial: a later full review of the same commit (a retry) must still post its own.
    partial = outcome.cut_short and not outcome.blocking
    prefix = CHECK_MARKER.format(check_id) + "\n" if check_id is not None else ""
    marker = prefix + (PARTIAL_MARKER if partial else VERDICT_MARKER)
    bodies = await github.review_bodies(repo, number, commit_sha=head_sha, author=author)
    if check_id is not None:
        if previous := published_event(bodies, check_id):
            return previous  # return the published decision, not the replay's recomputed outcome
    else:
        # Legacy (unscoped) callers: a full verdict can follow a partial.
        prior = (VERDICT_MARKER, PARTIAL_MARKER) if partial else (VERDICT_MARKER,)
        if any(body.endswith(prior) for body in bodies):
            return None
    event: ReviewEvent = "REQUEST_CHANGES" if outcome.blocking else "COMMENT" if partial else "APPROVE"
    unposted = [f for f in outcome.inline if not covered(posted or {}, fingerprint(head_sha, f), f)]

    def footer(outside: int) -> str:
        state = state_marker(event, outside=outside, complete=not outcome.cut_short)
        return verdict_footer(models_note, head_sha, state, marker)

    outside = sum(outcome.is_blocking(f) for f in outcome.general)
    try:
        await github.create_review(
            repo,
            number,
            commit_sha=head_sha,
            event=event,
            body=review_body(outcome, footer=footer(outside)),
            comments=[inline_comment(f, head_sha=head_sha, block_at=outcome.block_at) for f in unposted],
        )
    except ReviewRejected:
        # GitHub's diff view can differ from ours (e.g. files it doesn't render). Keep every finding, just
        # not inline: same verdict, the unposted findings listed in the review body (with no thread: outside).
        logger.warning("inline comments rejected for %s/%s#%d; listing them in the body", repo.owner, repo.name, number)
        listed = replace(outcome, inline=[], general=[*outcome.general, *outcome.inline])
        outside += sum(outcome.is_blocking(f) for f in unposted)
        await github.create_review(
            repo,
            number,
            commit_sha=head_sha,
            event=event,
            body=review_body(listed, footer=footer(outside)),
            comments=[],
        )
    return None


def state_marker(event: ReviewEvent, *, outside: int, complete: bool) -> str:
    return f"<!-- autoreviewer:state {event} outside={outside} complete={int(complete)} -->"


def verdict_footer(note: str, head_sha: str, state: str, marker: str) -> str:
    """What ends every verdict: who and what reviewed, then the state line, then the marker, last."""
    return f"<sub>autoreviewer · {note} · `{head_sha[:7]}`</sub>\n\n{state}\n{marker}"


async def reconsider(github: PullRequests, repo: Repo, number: int, *, author: str, block_at: int) -> bool:
    """After a withdrawal: approve the current head if its latest verdict requested changes after a complete
    review, listed no blocking finding without a thread (outside=0), and no blocking finding comment on it is
    left standing (each was withdrawn in its own thread). Decided by code from the PR's comments, never by the
    model; True if it approved. A replay finds the approval as the latest verdict and does nothing."""
    state = await github.get_pull(repo, number)
    if state.state != "open":
        return False
    bodies = await github.review_bodies(repo, number, commit_sha=state.head_sha, author=author)
    verdicts = [match for body in bodies if (match := _VERDICT_STATE.search(body))]
    if not verdicts or not overturnable(verdicts[-1]):
        return False
    if standing_blockers(await github.posted_comments(repo, number, author=author), block_at):
        return False
    # The green check first: a retry after a failure here never leaves an approval beside a red check.
    check = await github.start_check(repo, state.head_sha)
    summary = "Each blocking finding was withdrawn in its thread after a re-check of the code."
    await github.finish_check(repo, check, conclusion="success", title="Approved", summary=summary)
    state_line = state_marker("APPROVE", outside=0, complete=True)
    footer = verdict_footer("re-checked in conversation", state.head_sha, state_line, VERDICT_MARKER)
    body = f"No blocking finding is left standing on this commit. {summary}\n\n{footer}"
    await github.create_review(repo, number, commit_sha=state.head_sha, event="APPROVE", body=body, comments=[])
    logger.info("%s/%s#%d at %s: approved after withdrawals", repo.owner, repo.name, number, state.head_sha[:7])
    return True


DECISIVE: frozenset[ReviewState] = frozenset({"APPROVED", "CHANGES_REQUESTED", "DISMISSED"})
"""The review states that set a reviewer's decision on GitHub: its latest such review is the one in force."""
STALE_REQUEST = (
    "Every blocking finding of this review was withdrawn in its thread after a re-check of the code, or the code "
    "moved past it. This is not an approval: the newer commit gets its own review."
)


async def retire_stale_request(github: Conversations, repo: Repo, number: int, *, author: str, block_at: int) -> bool:
    """After a withdrawal, once the PR has moved past the commit the bot requested changes on: dismiss that review
    if nothing it could have stood on is left. `reconsider` handles a request on the head itself, by approving.

    Without this the request stays in force until a later review posts a verdict, and one that times out never
    does (#68). Only a complete verdict with no blocker outside a thread (outside=0) qualifies, as for `reconsider`.
    Lines on an older commit can't be placed on the head, so every blocking finding comment the bot posted on the
    PR, outdated or not, must be withdrawn in its thread or settled by the bot (it saw the code move past it); a
    person resolving a thread doesn't count. Never an approval: the bot hasn't reviewed the head, and the head's
    check still decides. True if it dismissed. Decided by code, never by the model; a replay finds the review
    dismissed and does nothing."""
    state = await github.get_pull(repo, number)
    if state.state != "open":
        return False
    decisive = [
        review for review in await github.posted_reviews(repo, number, author=author) if review.state in DECISIVE
    ]
    if not decisive or (review := decisive[-1]).state != "CHANGES_REQUESTED" or review.commit_sha == state.head_sha:
        return False
    if not overturnable(_VERDICT_STATE.search(review.body)):
        return False
    comments = await github.posted_comments(repo, number, author=author)
    cleared = withdrawn_threads(comments).keys() | {
        thread.root_comment_id for thread in await github.review_threads(repo, number) if thread.settled
    }
    if any(comment.id not in cleared for comment, _ in blocking_findings(comments, block_at)):
        return False
    try:
        await github.dismiss_review(repo, number, review.id, message=STALE_REQUEST)
    except Exception:  # e.g. a protected branch that limits who may dismiss: the PR is just left as it was
        logger.warning(
            "could not dismiss review %d on %s/%s#%d", review.id, repo.owner, repo.name, number, exc_info=True
        )
        return False
    logger.info("%s/%s#%d: dismissed review %d at %s after withdrawals", repo.owner, repo.name, number, review.id,
                review.commit_sha[:7])  # fmt: skip
    return True


async def settle_fixed(
    github: PullRequests, repo: Repo, number: int, *, head_sha: str, author: str, outcome: Outcome
) -> list[int]:
    """Settle the threads of the bot's findings the code has moved past, after a review of `head_sha` (mark_thread:
    the finding minimized as resolved). GitHub shows the comment as outdated (the code under it changed), this
    review checked that kind in that file in full (Outcome.covered), and it reported nothing of that kind there,
    in any state. A note follows, once settled. Returns their comment ids.

    Conservative on purpose: a reworded finding, or another one in the file, keeps the thread open."""
    reported = {(f.kind, f.path) for f in outcome.reported}
    gone: list[int] = []
    for comment in await github.posted_comments(repo, number, author=author):
        if comment.in_reply_to is not None or comment.line is not None or not (match := finding_marker(comment.body)):
            continue  # a reply, still on the head's code, or not a finding
        if (place := (match[3], comment.path)) in outcome.covered and place not in reported:
            gone.append(comment.id)
    threads = await _threads_by_root(github, repo, number) if gone else {}
    resolved = []
    for root in gone:
        if (thread := threads.get(root)) is None or thread.resolved or thread.hidden:
            continue  # done already: resolved by a person, settled earlier, or hidden by a moderator
        note = (
            f"The code here changed, and the review of `{head_sha[:7]}` found nothing like this in the file: "
            "marking it resolved."
        )
        await _settle(github, repo, thread, settled=True)  # first: a failure here leaves no note behind
        await github.reply(repo, number, root, note)
        resolved.append(root)
    logger.info(
        "%s/%s#%d at %s: settled %d fixed finding(s)", repo.owner, repo.name, number, head_sha[:7], len(resolved)
    )
    return resolved


async def mark_thread(github: PullRequests, repo: Repo, number: int, root_id: int, *, settled: bool) -> None:
    """Settle the thread under comment `root_id` (its finding minimized as resolved), or show it again."""
    if (thread := (await _threads_by_root(github, repo, number)).get(root_id)) is not None:
        await _settle(github, repo, thread, settled=settled)


async def _settle(github: PullRequests, repo: Repo, thread: ReviewThread, *, settled: bool) -> None:
    """Minimize the thread's finding as resolved unless it's hidden already; show it again only if it was
    minimized as resolved. A moderator's hide (spam, off-topic) is theirs: never undone."""
    if (not thread.hidden) if settled else thread.settled:
        await github.set_settled(repo, thread, settled=settled)


async def _threads_by_root(github: PullRequests, repo: Repo, number: int) -> dict[int, ReviewThread]:
    return {thread.root_comment_id: thread for thread in await github.review_threads(repo, number)}
