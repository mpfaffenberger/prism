"""lens -> verify -> dedupe -> anchor. No GitHub here: input is the PR's text and git facts, output is findings.

The models explore the change with their git and file tools; the prompts carry only the PR's text and
the list of changed files, so prompt size no longer grows with the diff.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from pydantic_ai import Agent, Tool
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, ToolCallPart, ToolReturnPart
from pydantic_ai.toolsets import FunctionToolset

from autoreviewer.review.agents import FOCUS, LENS_LIMITS, LENS_RULES, REPORT_TOOL, VERIFIER_LIMITS
from autoreviewer.review.checkpoints import Ran
from autoreviewer.review.compaction import compaction_stats
from autoreviewer.review.diff import ChangedPath, commentable_lines
from autoreviewer.review.memory import PriorReview
from autoreviewer.review.progress import Progress
from autoreviewer.review.render import sanitize
from autoreviewer.review.schemas import ADVISORY_KINDS, Finding, LensKind, LensReport, Verification
from autoreviewer.review.sessions import run_role
from autoreviewer.web.config import AgentKind, verifier_role

if TYPE_CHECKING:
    from autoreviewer.review.sessions import RoleSessions

logger = logging.getLogger("autoreviewer.review")

MAX_FINDINGS = 25
MIN_CONFIDENCE = 4
"""Findings below this confidence (PR-Agent's "a hunch") aren't worth a verifier call."""
BLOCK_AT_PRIORITY = 1
"""Confirmed findings at this priority or more urgent (P0, P1) block the merge; the rest are advisory."""
MAX_DESCRIPTION_CHARS = 4000
MAX_LISTED_FILES = 300
VERIFY_CONCURRENCY = 3


@dataclass(frozen=True)
class PullRequestText:
    title: str
    body: str
    reviewable: list[ChangedPath]
    skipped: list[ChangedPath]
    patches: dict[str, str]
    """Hunks per reviewable path, from git: decides which lines take an inline comment."""


def blocks(finding: Finding, block_at: int) -> bool:
    """Whether a confirmed finding blocks the merge: urgent enough, and not merely advice."""
    return blocks_at(finding.kind, finding.priority, block_at)


def blocks_at(kind: str | None, priority: int, block_at: int) -> bool:
    """The blocking rule on its own; a kind of None (a marker from before kinds were recorded) may block."""
    return kind not in ADVISORY_KINDS and priority <= block_at


@dataclass(frozen=True)
class Withdrawal:
    """A finding the bot withdrew in its thread, whose comment GitHub still shows on the current head."""

    path: str
    line: int
    kind: str
    priority: int
    title: str
    """As posted, normalised (see `covers`)."""

    def covers(self, finding: Finding) -> bool:
        """Whether `finding` is this one found again: same place, kind, and title, and no more urgent. Anything
        else blocks as usual: another defect on that line, or the same one found more urgent."""
        same = (finding.path, finding.line, finding.kind) == (self.path, self.line, self.kind)
        return same and finding.priority >= self.priority and normal_title(finding.title) == self.title


def normal_title(title: str) -> str:
    """A title as a comment shows it (sanitised), with case and spacing ignored."""
    return " ".join(sanitize(title).split()).casefold()


@dataclass
class Outcome:
    inline: list[Finding] = field(default_factory=list[Finding])
    """Confirmed and on a line GitHub accepts a review comment on."""
    general: list[Finding] = field(default_factory=list[Finding])
    """Confirmed, but outside the diff hunks: goes in the review body."""
    rejected: list[tuple[Finding, str]] = field(default_factory=list[tuple[Finding, str]])
    """The verifier's verdict was "not a bug" (or the path isn't in the PR)."""
    unverified: list[tuple[Finding, str]] = field(default_factory=list[tuple[Finding, str]])
    """The verifier never reached a verdict (e.g. the model server failed). Not posted, but not judged
    either: later reviews must check these again rather than treat them as rejected."""
    low_confidence: list[Finding] = field(default_factory=list[Finding])
    """Below MIN_CONFIDENCE, so never sent to the verifier. Not judged, so never remembered as rejected:
    a later review is free to report them again with more confidence."""
    block_at: int = BLOCK_AT_PRIORITY
    overall_correct: bool = True
    """The lens's overall verdict on the change (Codex-style), alongside its findings."""
    reviewed: list[str] = field(default_factory=list[str])
    cut_short: bool = False
    """A lens that can block hit the time limit before covering the whole change, or failed: never an approval."""
    failed: set[str] = field(default_factory=set[str])
    """Lenses that raised (a model error after the client's retries, say), so found nothing (job._survive)."""
    added_lines: int = 0
    """Lines the PR adds in reviewable files: what simplifications are measured against."""
    covered: set[tuple[str, str]] = field(default_factory=set[tuple[str, str]])
    """(kind, path) pairs this review checked in full: the lens finished over that file, and wasn't cut short or
    capped. Only a comment on a covered pair can be called fixed (posting.settle_fixed)."""
    merged: list[Finding] = field(default_factory=list[Finding])
    """Findings `absorb` folded into another lens's on the same line: not posted, but still reported."""
    withdrawn: tuple[Withdrawal, ...] = ()
    """Findings withdrawn in conversation on unchanged code (posting.Publisher.withdrawn): one found again is not
    posted again, and doesn't block."""
    standing: bool = False
    """An earlier review of this same commit left a blocking finding that still stands
    (posting.Publisher.standing): unchanged code can't be approved over it, whatever this review found."""

    @property
    def posted(self) -> list[Finding]:
        return [*self.inline, *self.general]

    @property
    def failed_blocking(self) -> list[str]:
        """The lenses that could block and failed: without them, the review can't approve."""
        return sorted(self.failed - ADVISORY_KINDS)

    @property
    def why_partial(self) -> str:
        """Why a cut-short review covers only part of the change."""
        if failed := self.failed_blocking:
            return f"the {' and '.join(failed)} review failed"
        return "the review hit its time limit before covering the whole change"

    @property
    def reported(self) -> list[Finding]:
        """Everything the lenses reported, whatever became of it: posted, unverified, rejected, or a hunch."""
        return [
            *self.posted,
            *(f for f, _ in [*self.unverified, *self.rejected]),
            *self.low_confidence,
            *self.merged,
        ]

    def uncover(self, kinds: Iterable[str]) -> None:
        """Take `kinds` out of what this review checked in full (cut short, or capped)."""
        kinds = set(kinds)
        self.covered = {(kind, path) for kind, path in self.covered if kind not in kinds}

    def absorb(self, other: Outcome) -> None:
        """Add another lens's results. Where correctness and security posted on one line (likely one defect
        seen twice), the more urgent finding stays; other kinds (types, DRY, simplify) are separate issues."""
        self.rejected += other.rejected
        self.unverified += other.unverified
        self.failed |= other.failed
        self.covered |= other.covered
        self.merged += other.merged
        self.low_confidence += other.low_confidence
        for target, findings in ((self.inline, other.inline), (self.general, other.general)):
            for finding in findings:
                if (same := next((f for f in target if same_defect(f, finding)), None)) is None:
                    target.append(finding)
                elif finding.priority < same.priority:
                    target[target.index(same)] = finding
                    self.merged.append(same)
                else:
                    self.merged.append(finding)

    def is_blocking(self, finding: Finding) -> bool:
        return blocks(finding, self.block_at) and not any(w.covers(finding) for w in self.withdrawn)

    @property
    def blocking(self) -> bool:
        return self.standing or any(self.is_blocking(f) for f in self.posted)


@dataclass(frozen=True)
class Continuation:
    """Continue the last review of this PR instead of starting over."""

    prior: PriorReview
    history: list[ModelMessage]
    head_sha: str
    changed_since: list[ChangedPath]
    """What changed between the head the last review saw and this one."""


def _row(change: ChangedPath) -> str:
    return f"- {change.status} {change.path} (+{change.added} -{change.removed})"


def lens_prompt(pr: PullRequestText) -> str:
    listed = pr.reviewable[:MAX_LISTED_FILES]
    more = len(pr.reviewable) - len(listed)
    lines = [
        f"# Pull request\n\nTitle: {pr.title}\n\nDescription:\n{pr.body[:MAX_DESCRIPTION_CHARS] or '(none)'}\n",
        f"# Changed files to review ({len(pr.reviewable)})\n",
        *(_row(change) for change in listed),
    ]
    if more:
        lines.append(f"- ... and {more} more: call changed_files() for the full list")
    if pr.skipped:
        lines.append(f"\n{len(pr.skipped)} generated, vendored, binary, or deleted file(s) are not reviewed.")
    lines.append("\nStart with diff(path) for each file above.")
    return "\n".join(lines)


def continuation_prompt(pr: PullRequestText, continuation: Continuation) -> str:
    prior = continuation.prior
    listed = continuation.changed_since[:MAX_LISTED_FILES]
    more = len(continuation.changed_since) - len(listed)
    lines = [
        f"# New commits\n\nThe pull request moved from `{prior.head_sha[:7]}` (your last review) to "
        f"`{continuation.head_sha[:7]}`. Tool results earlier in this conversation describe the old head; the "
        "checkout and every tool now show the new one.\n",
        f"Title: {pr.title}\n",
        f"# Changed since your last review ({len(continuation.changed_since)} files)\n",
        *(_row(change) for change in listed),
    ]
    if more:
        lines.append(f"- ... and {more} more: use diff_since_last_review() to see them")
    lines.append("\n# Findings you posted last time (line numbers refer to the old head)\n")
    lines.extend(f"- {f['path']}:{f['line']} {f['title']}" for f in prior.findings)
    if not prior.findings:
        lines.append("- none")
    if prior.unverified:
        lines.append("\n# Candidates that could not be verified last time (check them again)\n")
        lines.extend(f"- {f['path']}:{f['line']} {f['title']}" for f in prior.unverified)
    if prior.dropped:
        lines.append("\n# Candidates the verifier rejected last time (don't re-report unless the code changed)\n")
        lines.extend(f"- {f['path']}:{f['line']} {f['title']}: {f['reason']}" for f in prior.dropped)
    lines.append(
        "\nRe-check each finding above against the new code (it may be fixed, moved, or still there), review "
        "what changed with diff_since_last_review(path), and return the complete list of findings for the PR "
        "as it is now, with line numbers in the new head."
    )
    return "\n".join(lines)


def verifier_prompt(pr: PullRequestText, finding: Finding) -> str:
    return (
        f"# Reported finding\n```json\n{json.dumps(finding.model_dump(), indent=2)}\n```\n\n"
        f"# Pull request title\n{pr.title}\n\n"
        f'Check it: start with diff(path="{finding.path}") and read the surrounding code.\n'
    )


def same_defect(a: Finding, b: Finding) -> bool:
    """Correctness and security findings on one line: likely one defect seen by two lenses."""
    return {a.kind, b.kind} <= {"correctness", "security"} and (a.path, a.line) == (b.path, b.line)


def _dedupe(findings: list[Finding]) -> list[Finding]:
    seen: set[tuple[str, int]] = set()
    unique = []
    # Most urgent, then most confident, first: they win ties on a line and survive the cap.
    for finding in sorted(findings, key=lambda f: (f.priority, -f.confidence)):
        if (key := (finding.path, finding.line)) not in seen:
            seen.add(key)
            unique.append(finding)
    return unique[:MAX_FINDINGS]


def place(outcome: Outcome, finding: Finding, patch: str) -> bool:
    """File `finding` as inline (on a line GitHub can comment on) or general; True if inline."""
    inline = finding.line in commentable_lines(patch)
    (outcome.inline if inline else outcome.general).append(finding)
    return inline


Publish = Callable[[Finding], Awaitable[None]]
"""Post one confirmed inline finding right away (see posting.Publisher). Must not raise."""


type CheckAgent[T] = Agent[None, T] | Callable[[], Agent[None, T]]


def select_checker[T](checker: CheckAgent[T]) -> Agent[None, T]:
    return checker if isinstance(checker, Agent) else checker()


def current_workspace(history: list[ModelMessage] | None) -> Literal["new"] | None:
    """Never restore a previous checkout; agents without a workspace don't need a new one."""
    if history:
        last = next((msg for msg in reversed(history) if isinstance(msg, ModelResponse)), None)
        if last and last.workspace_ref is not None:
            return "new"
    return None


def finder_prompt(prompt: str, history: list[ModelMessage] | None) -> str:
    if not history:
        return prompt
    return (
        prompt + "\nEarlier tool results are stale. Re-read the current code and complete the whole "
        "review in scope for the current HEAD, not just a delta from the previous review."
    )


async def check_candidate[T](
    checker: CheckAgent[T],
    kind: AgentKind,
    subject: str,
    prompt: str,
    watch: AbstractCapability[None],
    sessions: RoleSessions | None,
) -> T:
    agent = select_checker(checker)
    verify = (
        prompt + "\nIndependently verify this candidate: you MUST re-read the current HEAD code. "
        "Earlier tool results are stale and a previous verdict is not evidence."
    )
    ran = await run_role(
        sessions,
        verifier_role(kind),
        agent,
        lambda messages, resuming, recorders: agent.run(
            None if resuming else verify,
            message_history=messages,
            workspace=current_workspace(messages),
            usage_limits=VERIFIER_LIMITS,
            capabilities=[*recorders, watch],
        ),
        subject=subject,
    )
    return ran.output


class Judge:
    """Verifies a lens's findings as they arrive (from `report_finding` mid-run, then the final report) and
    sorts each into `outcome`. Each confirmed inline finding is handed to `publish` at once."""

    def __init__(
        self,
        pr: PullRequestText,
        verifier: CheckAgent[Verification],
        progress: Progress,
        outcome: Outcome,
        *,
        kind: Literal["correctness"] | LensKind = "correctness",
        label: str,
        publish: Publish | None = None,
        limit: int = MAX_FINDINGS,
        sessions: RoleSessions | None = None,
    ) -> None:
        """`limit`: how many findings are verified at most, counting only those past the free filters."""
        self._pr, self._verifier, self._progress, self._outcome = pr, verifier, progress, outcome
        self._kind: Literal["correctness"] | LensKind = kind
        self._label, self._publish, self._limit = label, publish, limit
        self._sessions = sessions
        self._reviewable = {change.path for change in pr.reviewable}
        self._slots = asyncio.Semaphore(VERIFY_CONCURRENCY)
        self._tasks: list[asyncio.Task[None]] = []
        self._latest: dict[tuple[str, int], Finding] = {}
        """The most urgent finding reported for each line: only its verdict counts."""
        self._filtered: set[tuple[str, int]] = set()

    def toolset(self) -> FunctionToolset[None]:
        async def report(finding: Finding) -> str:
            """Report one finding now, while you keep reviewing. Same fields as a finding in the final report."""
            return self.submit(finding)

        return FunctionToolset([Tool(report, name=REPORT_TOOL)])

    def replay(self, history: Sequence[ModelMessage]) -> None:
        """Rebuild what an interrupted lens run reported: resubmit, in order, each `report_finding` call that was
        recorded (answered with success, not retried or closed out as interrupted). Verdicts that finished before
        the restart come back from their checkpoints; only unfinished ones run again. A report that compaction
        summarized away is recovered later, from the final report, which lists every finding."""
        answered = {
            part.tool_call_id
            for message in history
            if isinstance(message, ModelRequest)
            for part in message.parts
            if isinstance(part, ToolReturnPart) and part.tool_name == REPORT_TOOL and part.outcome == "success"
        }
        for message in history:
            if not isinstance(message, ModelResponse):
                continue
            for part in message.parts:
                if isinstance(part, ToolCallPart) and part.tool_name == REPORT_TOOL and part.tool_call_id in answered:
                    try:
                        finding = Finding.model_validate(part.args_as_dict()["finding"])
                    except (KeyError, ValueError):
                        continue  # it was answered with a retry, not recorded
                    self.submit(finding)

    def submit(self, finding: Finding) -> str:
        """Start verifying `finding`, unless it's out of scope, a hunch, or not more urgent than what was
        already reported for its line (a strictly more urgent report replaces the earlier one)."""
        finding = finding.model_copy(update={"kind": self._kind})
        key = (finding.path, finding.line)
        current = self._latest.get(key)
        if current is not None and finding.priority >= current.priority:
            return "Already reported for this line. Keep reviewing."
        if finding.path not in self._reviewable:  # hallucinated or out of scope, not worth a verifier call
            self._record(self._outcome.rejected, key, (finding, "path is not among the changed files"))
            return f"{finding.path} is not among the changed files: not reported."
        if finding.confidence < MIN_CONFIDENCE:
            self._record(self._outcome.low_confidence, key, finding)
            return "Below the confidence needed to verify it: not reported."
        if current is None and len(self._latest) >= self._limit:
            return f"You have reported {self._limit} findings, the most that are checked: not reported."
        if current is not None:  # superseded: its verdict (if any) no longer counts
            self._outcome.unverified[:] = [(f, why) for f, why in self._outcome.unverified if (f.path, f.line) != key]
            self._outcome.rejected[:] = [(f, why) for f, why in self._outcome.rejected if (f.path, f.line) != key]
        self._latest[key] = finding
        self._tasks.append(asyncio.create_task(self._verify(finding)))
        self._progress.note(f"reported: {finding.title}")
        return "Recorded: it is being verified now and will be posted if confirmed. Keep reviewing."

    def _record[T](self, records: list[T], key: tuple[str, int], record: T) -> None:
        """Note a finding filtered out without verification, once per line."""
        if key not in self._filtered:
            self._filtered.add(key)
            records.append(record)

    async def finish(self, findings: list[Finding]) -> None:
        """Verify the final report's findings that weren't reported along the way, and wait for every verdict."""
        for finding in _dedupe(findings):
            self.submit(finding)
        if pending := [task for task in self._tasks if not task.done()]:
            self._progress.set_phase(f"Verifying {len(pending)} candidate finding(s)")
        await asyncio.gather(*self._tasks)

    def cancel(self) -> None:
        for task in self._tasks:
            task.cancel()

    async def _verify(self, reported: Finding) -> None:
        finding = reported
        async with self._slots:
            try:
                watch = self._progress.watch(f"{self._label} ({finding.path}:{finding.line})")
                prompt = verifier_prompt(self._pr, finding)
                subject = json.dumps([finding.path, finding.line, normal_title(finding.title)])
                verdict = await check_candidate(self._verifier, self._kind, subject, prompt, watch, self._sessions)
            except Exception as error:  # precision first: an unverifiable finding is not posted
                logger.warning("verifying %s:%s failed: %r", finding.path, finding.line, error)
                if self._latest.get((finding.path, finding.line)) is reported:  # not superseded meanwhile
                    self._outcome.unverified.append((finding, f"verification failed: {type(error).__name__}"))
                return
        self._progress.note(f"{'confirmed' if verdict.confirmed else 'rejected'}: {finding.title}")
        if self._latest.get((finding.path, finding.line)) is not reported:
            return  # a more urgent report for this line superseded it while it was being verified
        if not verdict.confirmed:
            self._outcome.rejected.append((finding, verdict.reason))
            return
        # The verifier read the code too: its corrections win.
        corrections = {"priority": verdict.priority, "explanation": verdict.explanation}
        finding = finding.model_copy(update={k: v for k, v in corrections.items() if v is not None})
        for confirmed in (self._outcome.inline, self._outcome.general):  # an earlier, less urgent report
            confirmed[:] = [
                f for f in confirmed if (f.kind, f.path, f.line) != (finding.kind, finding.path, finding.line)
            ]
        if place(self._outcome, finding, self._pr.patches.get(finding.path, "")) and self._publish is not None:
            await self._publish(finding)


LensRun = Callable[[list[FunctionToolset[None]], Callable[[list[ModelMessage]], None]], Awaitable[Ran[LensReport]]]
"""Starts a lens run with these extra toolsets, and what to call with its history if it resumes after a restart."""


async def _run_judged(judge: Judge, run: LensRun, *, report_as_you_go: bool = True) -> Ran[LensReport]:
    """Run a lens (with `report_finding` if it reports as it goes), then judge its final report; no verification
    outlives a failure."""
    try:
        result = await run([judge.toolset()] if report_as_you_go else [], judge.replay)
        await judge.finish(result.output.findings)
    except BaseException:
        judge.cancel()
        raise
    return result


async def review_lens(
    pr: PullRequestText,
    lens: Agent[None, LensReport],
    verifier: CheckAgent[Verification],
    *,
    kind: LensKind,
    instructions: str | None,
    progress: Progress,
    block_at: int = BLOCK_AT_PRIORITY,
    publish: Publish | None = None,
    capabilities: Sequence[AbstractCapability[None]] = (),
    max_posted: int = MAX_FINDINGS,
    sessions: RoleSessions | None = None,
) -> Outcome:
    """An extra lens (security, DRY) over the current `pr`, checked by its own verifier.

    With a `max_posted` cap, only its best findings are verified (and so posted): it doesn't report as it goes,
    so the cut is made on its ranked final report rather than on whatever it found first."""
    outcome = Outcome(block_at=block_at)
    judge = Judge(
        pr,
        verifier,
        progress,
        outcome,
        kind=kind,
        label=f"{kind} verifier",
        publish=publish,
        limit=max_posted,
        sessions=sessions,
    )
    report_as_you_go = max_posted >= MAX_FINDINGS
    combined = "\n\n".join(part for part in (instructions, LENS_RULES if report_as_you_go else FOCUS) if part)
    run = await _run_judged(
        judge,
        lambda toolsets, replay: run_role(
            sessions,
            kind,
            lens,
            lambda messages, resuming, recorders: lens.run(
                None if resuming else finder_prompt(lens_prompt(pr), messages),
                message_history=messages,
                workspace=current_workspace(messages),
                instructions=combined,
                toolsets=toolsets,
                capabilities=[*recorders, *capabilities, progress.watch(kind)],
                usage_limits=LENS_LIMITS,
            ),
            on_resume=replay,
        ),
        report_as_you_go=report_as_you_go,
    )
    logger.info("%s: %d requests, %d posted", kind, run.usage.requests, len(outcome.posted))
    return outcome


async def review(
    pr: PullRequestText,
    lens: Agent[None, LensReport],
    verifier: CheckAgent[Verification],
    *,
    continuation: Continuation | None = None,
    sessions: RoleSessions | None = None,
    lens_capabilities: Sequence[AbstractCapability[None]] = (),
    conversation: str | None = None,
    block_at: int = BLOCK_AT_PRIORITY,
    checked: Sequence[Finding] = (),
    progress: Progress | None = None,
    publish: Publish | None = None,
) -> Outcome:
    progress = progress or Progress()
    outcome = Outcome(block_at=block_at)
    for finding in checked:  # proven by a tool, not a model: nothing to verify, so post at once
        if place(outcome, finding, pr.patches.get(finding.path, "")) and publish is not None:
            await publish(finding)
    prompt = lens_prompt(pr)
    if continuation:
        history = continuation.history
        prompt = continuation_prompt(pr, continuation) + "\n\n" + prompt
    else:
        history = await sessions.history("correctness", lens) if sessions else None
    progress.set_phase("Continuing the last review" if continuation or history else "Reviewing the change")
    judge = Judge(pr, verifier, progress, outcome, label="verifier", publish=publish, sessions=sessions)
    lens_run = await _run_judged(
        judge,
        lambda toolsets, replay: run_role(
            sessions,
            "correctness",
            lens,
            lambda messages, resuming, recorders: lens.run(
                None if resuming else finder_prompt(prompt, messages),
                instructions=LENS_RULES,
                toolsets=toolsets,
                message_history=messages,
                capabilities=[*recorders, *lens_capabilities, progress.watch("reviewer")],
                conversation_id=conversation,
                # The history names the last review's checkout; this review has a new one (the new head).
                workspace="new" if continuation else current_workspace(messages),
                usage_limits=LENS_LIMITS,
            ),
            history=history,
            on_resume=replay,
        ),
    )
    report, usage = lens_run.output, lens_run.usage
    outcome.overall_correct, outcome.reviewed = report.overall_correct, report.reviewed
    compacted = compaction_stats(lens_run.all_messages())
    logger.info(
        "review: lens made %d requests and %d tool calls (compaction cleared %d, deduplicated %d results); "
        "%d found, %d inline, %d general, %d rejected, %d unverified, %d low-confidence",
        usage.requests,
        usage.tool_calls,
        compacted["cleared"],
        compacted["superseded"],
        len(report.findings),
        len(outcome.inline),
        len(outcome.general),
        len(outcome.rejected),
        len(outcome.unverified),
        len(outcome.low_confidence),
    )
    return outcome
