"""The three phases of reviewing one PR push. `durable.py` runs each as a DBOS step.

    start_check  ->  review_and_post  ->  finish_check

Each phase is safe to replay after a crash: the check run is created once (its step's result is
checkpointed), and posting recognizes a verdict from the same check run.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import tempfile
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

from pydantic_ai import Agent

from autoreviewer.github import (
    CommitNotFound,
    Conclusion,
    PullRequestReader,
    PullRequests,
    PullRequestState,
    Repo,
    ReviewEvent,
)
from autoreviewer.review.checkpoints import TurnCheckpoints
from autoreviewer.review.deadline import DEFAULT_SOFT_S, MAX_REVIEW_S, Deadline
from autoreviewer.review.diff import MAX_CHANGED_LINES, added_lines, select
from autoreviewer.review.lint import LINT_KINDS, MAX_LINT_FINDINGS, LintError, lint_findings
from autoreviewer.review.memory import PriorReview, ReviewMemory, conversation_id
from autoreviewer.review.pipeline import (
    BLOCK_AT_PRIORITY,
    MAX_FINDINGS,
    CheckAgent,
    Continuation,
    Outcome,
    Publish,
    PullRequestText,
    place,
    review,
    review_lens,
)
from autoreviewer.review.posting import Publisher, post_review, published_event, settle_fixed
from autoreviewer.review.progress import Progress
from autoreviewer.review.schemas import ADVISORY_KINDS, LensKind, LensReport, Verification
from autoreviewer.review.sessions import RoleSessions
from autoreviewer.review.simplify import SimplifierFactory, suggest
from autoreviewer.review.workspace import RepoSource, Workspace, read_changes, read_changes_since_previous
from autoreviewer.web.routing import PullRequestUpdated, ThreadReply

logger = logging.getLogger("autoreviewer.job")

SUPERSEDED_CHECK_S = 60.0
"""How often a running review checks whether its PR moved on (a newer push, or closed)."""
PROGRESS_INTERVAL_S = 30.0
"""How often a running review updates its check run, at most (GitHub rate-limits App writes)."""

AgentFactory = Callable[[Workspace], tuple[Agent[None, LensReport], CheckAgent[Verification]]]


@dataclass(frozen=True)
class Lens:
    """A lens beside correctness: reviews the PR fresh each time, with its own verifier."""

    kind: LensKind
    agents: AgentFactory
    instructions: Callable[[Sequence[str]], str] | None = None
    """Run instructions from the paths it reviews (e.g. the security lens's language references)."""
    max_posted: int = MAX_FINDINGS
    """Verify and post at most this many of its findings: advice should stay short."""
    whole_pr: bool = True
    """On a continued review: the whole PR again (required for a lens that blocks, or a finding from an
    earlier commit would quietly stop blocking), or only files changed since the last review."""


async def _run_lens(
    spec: Lens,
    workspace: Workspace,
    pr: PullRequestText,
    *,
    progress: Progress,
    block_at: int,
    publish: Publish | None,
    deadline: Deadline | None,
    sessions: RoleSessions | None = None,
) -> Outcome:
    """One extra lens over `pr`."""
    lens, verifier = spec.agents(workspace)
    instructions = spec.instructions([change.path for change in pr.reviewable]) if spec.instructions else None
    return await review_lens(
        pr,
        lens,
        verifier,
        kind=spec.kind,
        instructions=instructions,
        progress=progress,
        block_at=block_at,
        publish=publish,
        capabilities=[deadline.watch(spec.kind)] if deadline else (),
        max_posted=spec.max_posted,
        sessions=sessions,
    )


async def _survive(kind: str, block_at: int, run: Awaitable[Outcome], paths: Iterable[str]) -> Outcome:
    """The lens's outcome; if it fails (a model error after the client's retries, say), an empty one naming it,
    and the other lenses carry on. An advisory lens costs only its advice; a lens that can block leaves the
    review cut short, which never approves, so nothing fails open. Cancellation isn't caught: a superseded
    review still stops. A lens that finishes covers its kind over `paths`, the files it reviewed."""
    try:
        outcome = await run
    except Exception:
        logger.warning("the %s lens failed; the review goes on without it", kind, exc_info=True)
        return Outcome(block_at=block_at, failed={kind})
    outcome.covered |= {(kind, path) for path in paths}
    return outcome


def _only(pr: PullRequestText, paths: set[str]) -> PullRequestText:
    return replace(pr, reviewable=[change for change in pr.reviewable if change.path in paths])


@dataclass(frozen=True)
class ReviewDeps:
    github: PullRequests
    source: RepoSource
    """Where the code comes from: a git checkout of the PR head plus its merge-base."""
    agents: AgentFactory
    """Build (lens, verifier) over a checkout."""
    work_dir: Path
    bot_login: str
    """Our own GitHub login, e.g. example-reviewer[bot]; used to avoid reviewing a commit twice."""
    models_note: str
    """Shown in the review footer: which model found and which verified, per lens (web/app.py models_note)."""
    memory: ReviewMemory | None = None
    """Where each PR's lens conversation is kept, so a new commit continues the last review."""
    session_namespace: str = "primary"
    select_correctness: Callable[[], tuple[AgentFactory, str]] | None = None
    lens_version: str = ""
    """Fingerprint of the lens prompt and model (see agents.lens_version)."""
    block_at: int = BLOCK_AT_PRIORITY
    """Confirmed findings at this priority or more urgent block the merge (1: P0 and P1)."""
    simplifier: SimplifierFactory | None = None
    """Optional one-click simplifications (see simplify.py); None skips them."""
    lenses: tuple[Lens, ...] = ()
    """Extra lenses (security, DRY) beside correctness."""
    soft_deadline_s: float = DEFAULT_SOFT_S
    """After this, every lens wraps up with what it has (see deadline.py)."""
    hard_deadline_s: float = MAX_REVIEW_S
    """After this, the review is cancelled: no verdict (findings already posted stay)."""


@dataclass(frozen=True)
class CheckResult:
    title: str
    summary: str
    conclusion: Conclusion
    details: str = ""
    """What the review did (see progress.py), kept on the finished check run."""


@dataclass(frozen=True)
class Reviewed:
    outcome: Outcome
    remember: PriorReview | None
    sessions: RoleSessions | None = None
    """What to save for the next review of this PR; None without memory."""


class Superseded(Exception):
    """The PR moved on (new push, closed) while we worked; the newer event gets its own review."""


def repo_of(event: PullRequestUpdated | ThreadReply) -> Repo:
    """The repository a webhook event is about, as the GitHub client addresses it."""
    return Repo(event.installation_id, event.owner, event.repo)


async def start_check(deps: ReviewDeps, pull: PullRequestUpdated) -> int:
    return await deps.github.start_check(repo_of(pull), pull.head_sha)


async def _published_result(deps: ReviewDeps, pull: PullRequestUpdated, check_id: int) -> CheckResult | None:
    bodies = await deps.github.review_bodies(
        repo_of(pull), pull.number, commit_sha=pull.head_sha, author=deps.bot_login
    )
    return _recovered_result(event) if (event := published_event(bodies, check_id)) else None


def _recovered_result(event: ReviewEvent) -> CheckResult:
    if event == "APPROVE":
        return CheckResult("No blocking issues", "Approved; see the published review.", "success")
    if event == "REQUEST_CHANGES":
        return CheckResult("Changes requested", "See the blocking findings in the published review.", "failure")
    return CheckResult("Partially reviewed", "The published review was incomplete; not approved.", "neutral")


async def finish_check(deps: ReviewDeps, pull: PullRequestUpdated, check_id: int, result: CheckResult) -> None:
    timed_out = result.title == "Review timed out"
    if timed_out or result.title == "Review failed":
        # No review budget left, or the POST response was lost. Reconcile the external write in the
        # retriable completion step, without running models or granting the review extra time.
        try:
            result = await _published_result(deps, pull, check_id) or result
        except Exception:
            logger.exception("Could not reconcile check %s; keeping its neutral result", check_id)
    await deps.github.finish_check(
        repo_of(pull),
        check_id,
        conclusion=result.conclusion,
        title=result.title,
        summary=result.summary,
        text=result.details,
    )
    if timed_out:
        # The timeout cancels the review step. Clean up in the completion step, after publishing the check.
        repo = repo_of(pull)
        await _forget_if_closed(deps, repo, pull, conversation_id(repo, pull.number))


async def review_and_post(
    deps: ReviewDeps,
    pull: PullRequestUpdated,
    check_id: int | None = None,
    *,
    remaining_s: float | None = None,
    attempt: str | None = None,
) -> CheckResult:
    """Review within one wall-clock budget, including GitHub reads, posting and progress updates.

    On recovery, `remaining_s` is the time left on the durable hard deadline; neither clock restarts. When review
    memory is configured and `attempt` gives the DBOS workflow ID, each agent run checkpoints its settled turns, so
    a recovery resumes them instead of repeating them. Without memory, nothing is checkpointed and a recovery runs
    every turn again. External cancellation propagates rather than becoming a verdict, and keeps the checkpoints
    for that recovery.
    """
    result = await _review_within_budget(deps, pull, check_id, remaining_s=remaining_s, attempt=attempt)
    if attempt is not None and deps.memory is not None:
        TurnCheckpoints.discard(deps.memory.directory(conversation_id(repo_of(pull), pull.number)), attempt)
    return result


async def _review_within_budget(
    deps: ReviewDeps,
    pull: PullRequestUpdated,
    check_id: int | None,
    *,
    remaining_s: float | None,
    attempt: str | None,
) -> CheckResult:
    started = asyncio.get_running_loop().time()
    hard_s = min(deps.hard_deadline_s, MAX_REVIEW_S)
    budget = hard_s if remaining_s is None else min(hard_s, remaining_s)
    elapsed = hard_s - budget
    deadline = Deadline(deps.soft_deadline_s, started=time.monotonic() - elapsed)
    progress = Progress()
    repo = repo_of(pull)
    result: CheckResult | None = None
    try:
        if budget <= 0:
            raise TimeoutError
        async with asyncio.timeout_at(started + budget), _publishing(deps, repo, check_id, progress):
            reviewed = await _review(deps, repo, pull, progress, deadline, check_id, attempt)
            if isinstance(reviewed, CheckResult):
                result = reviewed
            else:
                # Posting succeeded. Neither thread cleanup nor a pending progress write changes that verdict.
                result = _summary(reviewed)
                try:
                    await settle_fixed(
                        deps.github, repo, pull.number, head_sha=pull.head_sha, author=deps.bot_login, outcome=reviewed
                    )
                except Exception:
                    logger.warning("could not settle fixed findings for %s#%d", repo, pull.number, exc_info=True)
    except Superseded:
        result = CheckResult("Superseded", "A newer push replaced this commit before the review finished.", "neutral")
    except TimeoutError:
        if result is not None:
            return replace(result, details=progress.details())
        logger.warning("reviewing %s/%s#%d at %s timed out", pull.owner, pull.repo, pull.number, pull.head_sha[:7])
        result = CheckResult(
            "Review timed out",
            f"Stopped after {hard_s / 60:g} minutes, before a verdict. Findings confirmed until then "
            "are posted as comments. Push again or request another review to retry.",
            "neutral",
        )
    except Exception:
        logger.exception("reviewing %s/%s#%d at %s failed", pull.owner, pull.repo, pull.number, pull.head_sha[:7])
        result = CheckResult(
            "Review failed",
            "Push again or request another review to retry. The bot's logs have the failure details.",
            "neutral",
        )
    return replace(result, details=progress.details())


@contextlib.asynccontextmanager
async def _publishing(deps: ReviewDeps, repo: Repo, check_id: int | None, progress: Progress) -> AsyncIterator[None]:
    """Publish `progress` to the check run for the duration of the block (if there is a check run)."""
    if check_id is None:
        yield
        return
    done = asyncio.Event()
    publisher = asyncio.create_task(_publish(deps, repo, check_id, progress, done))
    try:
        yield
    except asyncio.CancelledError:
        publisher.cancel()  # The review was cancelled; an in-flight update must not hold it open.
        raise
    finally:
        done.set()
        # On normal completion, wait: an update cut off mid-request could still reach GitHub after the final result
        # and overwrite it with progress.
        await publisher


async def _publish(deps: ReviewDeps, repo: Repo, check_id: int, progress: Progress, done: asyncio.Event) -> None:
    """Show progress on the check run every PROGRESS_INTERVAL_S while something changed, until `done`.
    Best-effort: GitHub limits how often an App may write, and a failed update must never touch the review."""
    published = progress.version
    while True:
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(done.wait(), PROGRESS_INTERVAL_S)
            return
        if progress.version == published:
            continue
        published = progress.version
        try:
            await deps.github.update_check(
                repo, check_id, title=progress.phase, summary=progress.summary(), text=progress.details()
            )
        except Exception:
            logger.warning("could not show progress on check run %d", check_id, exc_info=True)


async def review_change(
    *,
    reader: PullRequestReader,
    source: RepoSource,
    agents: AgentFactory,
    work_dir: Path,
    repo: Repo,
    number: int,
    state: PullRequestState,
    head_sha: str,
    memory: ReviewMemory | None = None,
    session_namespace: str = "primary",
    lens_version: str = "",
    block_at: int = BLOCK_AT_PRIORITY,
    progress: Progress | None = None,
    simplifier: SimplifierFactory | None = None,
    lenses: Sequence[Lens] = (),
    publish: Publish | None = None,
    deadline: Deadline | None = None,
    attempt: str | None = None,
) -> Reviewed | CheckResult:
    """Review one PR head: check it out, find what changed, run lens and verifier.

    With `memory`, a head that adds commits on top of the last reviewed one continues that review's
    conversation; anything else (first review, force-push, moved merge-base, changed prompt or model)
    starts fresh. Returns the findings (plus what to remember, which the caller saves only once it knows
    the PR is still open), or a CheckResult when there is nothing to review or it is too large. Shared
    by the bot's job (which then posts) and on-demand reviews (which print).
    """
    base = (await reader.compare(repo, state.base_sha, head_sha)).merge_base
    conversation = conversation_id(repo, number)
    comparisons: dict[str, bool] = {}
    ancestry_lock = asyncio.Lock()

    async def compatible(previous: str) -> bool:
        async with ancestry_lock:
            if previous not in comparisons:
                try:
                    comparisons[previous] = (await reader.compare(repo, previous, head_sha)).status == "ahead"
                except Exception:
                    logger.warning("cannot verify history ancestry for %s; starting fresh", conversation)
                    comparisons[previous] = False
            return comparisons[previous]

    sessions = (
        RoleSessions(
            memory.directory(conversation),
            head_sha,
            base,
            compatible,
            f"{repo.installation_id}:{session_namespace}",
            # A restarted attempt resumes each run from its settled turns (overflow too: its namespace is in the key).
            turns=TurnCheckpoints(memory.directory(conversation), attempt) if attempt else None,
        )
        if memory
        else None
    )
    # Overflow may reuse its own sessions but must neither read nor overwrite the primary continuation index.
    if session_namespace != "primary":
        memory = None
    prior = await _resumable(memory, reader, repo, conversation, base=base, head_sha=head_sha, version=lens_version)

    with tempfile.TemporaryDirectory(prefix="review-", dir=work_dir) as scratch:
        workspace = await source.checkout(
            repo, head=head_sha, base=base, dest=Path(scratch) / "repo", previous=prior.head_sha if prior else None
        )
        changes, patches = await read_changes(workspace)
        selection = select(changes)
        if not selection.reviewable:
            return CheckResult(
                "Nothing to review", "Only removed, generated, vendored, or binary files changed.", "success"
            )
        if selection.changed_lines > MAX_CHANGED_LINES:
            return CheckResult(
                "Too large to review",
                f"{selection.changed_lines} changed lines; the limit is {MAX_CHANGED_LINES}.",
                "neutral",
            )
        continuation = None
        if prior is not None and memory is not None:
            history = await memory.history(conversation, prior)
            if history:
                changed_since = await read_changes_since_previous(workspace)
                continuation = Continuation(prior, history, head_sha, changed_since)
        persistence, run_id = memory.persistence(conversation) if memory else (None, "")
        try:
            unchecked: set[str] = set()
            linted = await lint_findings(workspace.root, selection.reviewable, patches, unchecked=unchecked)
            lint_ran = True
        except LintError:  # a broken linter costs the lint checks, never the review
            logger.exception("%s at %s: lint checks skipped", conversation, head_sha[:7])
            linted, lint_ran, unchecked = [], False, set()
        lens, verifier = agents(workspace)
        pr = PullRequestText(state.title, state.body, selection.reviewable, selection.skipped, patches)
        progress = progress or Progress()
        # Continuing: only files changed since the last review, so each push doesn't repeat suggestions.
        scope = {change.path for change in continuation.changed_since} if continuation else None
        # Each lens survives the others' failures (_survive): one model error costs that lens, not the review.
        async with asyncio.TaskGroup() as lenses_group:
            correctness = lenses_group.create_task(
                _survive(
                    "correctness",
                    block_at,
                    review(
                        pr,
                        lens,
                        verifier,
                        continuation=continuation,
                        sessions=sessions,
                        lens_capabilities=[
                            *([persistence] if persistence else ()),
                            *([deadline.watch("reviewer")] if deadline else ()),
                        ],
                        conversation=conversation,
                        block_at=block_at,
                        checked=linted,
                        progress=progress,
                        publish=publish,
                    ),
                    # Continuing, it covers only what it re-read: the files changed since its last review.
                    [change.path for change in (pr if scope is None else _only(pr, scope)).reviewable],
                )
            )
            extra = []
            for spec in lenses:
                target = pr if spec.whole_pr or scope is None else _only(pr, scope)
                if target.reviewable:
                    run = _run_lens(
                        spec,
                        workspace,
                        target,
                        progress=progress,
                        block_at=block_at,
                        publish=publish,
                        deadline=deadline,
                        sessions=sessions,
                    )
                    paths = [change.path for change in target.reviewable]
                    extra.append(lenses_group.create_task(_survive(spec.kind, block_at, run, paths)))
            simplify_task = None
            if simplifier is not None:
                simplify_task = lenses_group.create_task(
                    suggest(
                        pr,
                        simplifier(workspace),
                        workspace.root,
                        progress,
                        paths=scope,
                        sessions=sessions,
                        publish=publish,
                        capabilities=[deadline.watch("simplifier")] if deadline else (),
                    )
                )
        outcome = correctness.result()
        # The lint findings reach the outcome through the correctness lens (`checked`): without it, they're lost.
        if lint_ran and "correctness" not in outcome.failed and len(linted) < MAX_LINT_FINDINGS:
            outcome.covered |= {
                (kind, change.path)
                for kind in LINT_KINDS
                for change in pr.reviewable
                if not (kind == "lint" and change.path in unchecked)  # its unused imports weren't checked
            }
        for task in extra:
            outcome.absorb(task.result())
        suggestions = simplify_task.result() if simplify_task is not None else []
        outcome.added_lines = sum(len(added_lines(patches.get(change.path, ""))) for change in pr.reviewable)
        for suggestion in suggestions:
            place(outcome, suggestion, patches.get(suggestion.path, ""))
    logger.info(
        "%s at %s: %s", conversation, head_sha[:7], "continued the last review" if continuation else "fresh review"
    )
    blocking_lenses = {"reviewer", *(spec.kind for spec in lenses if spec.kind not in ADVISORY_KINDS)}
    timed_out = deadline is not None and bool(deadline.wrapped_up & blocking_lenses)
    outcome.cut_short = timed_out or bool(outcome.failed_blocking)
    if deadline is not None:
        outcome.uncover("correctness" if lens == "reviewer" else lens for lens in deadline.wrapped_up)
    # A capped lens verifies only its best candidates and drops the rest: at its cap, it may have dropped one.
    caps = {"correctness": MAX_FINDINGS, **{spec.kind: spec.max_posted for spec in lenses}}
    outcome.uncover(kind for kind, cap in caps.items() if sum(f.kind == kind for f in outcome.reported) >= cap)
    # A review cut short isn't continued from: files it never read wouldn't count as "changed since".
    remember = None
    if memory is not None and not outcome.cut_short:
        remember = _prior_review(run_id, head_sha, base, lens_version, outcome)
    return Reviewed(outcome, remember, sessions)


async def _resumable(
    memory: ReviewMemory | None,
    reader: PullRequestReader,
    repo: Repo,
    conversation: str,
    *,
    base: str,
    head_sha: str,
    version: str,
) -> PriorReview | None:
    """The last review, if this head simply adds commits on top of what it saw. Otherwise None."""
    if memory is None or (prior := memory.last(conversation)) is None:
        return None
    if prior.version != version or prior.base_sha != base or prior.head_sha == head_sha:
        return None  # another prompt/model, a moved merge-base, or the same head again: start fresh
    try:
        comparison = await reader.compare(repo, prior.head_sha, head_sha)
    except CommitNotFound:
        return None  # the reviewed head is gone (force-pushed away, then garbage-collected)
    if comparison.status != "ahead":
        return None  # force-push, rebase, or reset: the old conversation describes code that's gone
    return prior


def _prior_review(run_id: str, head_sha: str, base: str, version: str, outcome: Outcome) -> PriorReview:
    return PriorReview(
        run_id=run_id,
        head_sha=head_sha,
        base_sha=base,
        version=version,
        # Only what the lens said: linter findings are recomputed every review, and aren't its to re-check.
        findings=[
            {"path": f.path, "line": f.line, "title": f"[P{f.priority}] {f.title}"}
            for f in outcome.posted
            if f.kind == "correctness"
        ],
        dropped=[
            {"path": f.path, "line": f.line, "title": f.title, "reason": reason}
            for f, reason in outcome.rejected
            if f.kind == "correctness"
        ],
        # Never as "dropped": a verifier failure is no verdict, and the next review must look again.
        unverified=[
            {"path": f.path, "line": f.line, "title": f.title} for f, _ in outcome.unverified if f.kind == "correctness"
        ],
    )


async def _review(
    deps: ReviewDeps,
    repo: Repo,
    pull: PullRequestUpdated,
    progress: Progress,
    deadline: Deadline,
    check_id: int | None,
    attempt: str | None,
) -> CheckResult | Outcome:
    if check_id is not None and (published := await _published_result(deps, pull, check_id)) is not None:
        return published  # crashed after posting: never run models or post findings for this attempt again
    state = await _current(deps, repo, pull)
    agents, version = deps.select_correctness() if deps.select_correctness else (deps.agents, deps.lens_version)
    conversation = conversation_id(repo, pull.number)
    publisher = Publisher(
        deps.github, repo, pull.number, head_sha=pull.head_sha, author=deps.bot_login, block_at=deps.block_at
    )
    try:
        result = await _unless_superseded(
            deps,
            repo,
            pull,
            review_change(
                reader=deps.github,
                source=deps.source,
                agents=agents,
                work_dir=deps.work_dir,
                repo=repo,
                number=pull.number,
                state=state,
                head_sha=pull.head_sha,
                memory=deps.memory,
                session_namespace=deps.session_namespace,
                lens_version=version,
                block_at=deps.block_at,
                progress=progress,
                simplifier=deps.simplifier,
                lenses=deps.lenses,
                publish=publisher,
                deadline=deadline,
                attempt=attempt,
            ),
        )

    except Exception:
        await _forget_if_closed(deps, repo, pull, conversation)  # snapshots written mid-run must not outlive a close
        raise
    if isinstance(result, CheckResult):
        return result

    # The review took minutes. Forgetting wins: a PR closed meanwhile keeps nothing (its webhook already
    # deleted the memory, and this review must not write it back). A newer head keeps this analysis, which
    # the next review can continue from, but gets no post: it has its own review coming.
    latest = await deps.github.get_pull(repo, pull.number)
    if latest.state != "open":
        if deps.memory is not None:
            deps.memory.forget(conversation)
        raise Superseded
    if result.sessions is not None:
        result.sessions.commit()  # context only, never proof of coverage
    if deps.memory is not None and result.remember is not None:
        deps.memory.remember(conversation, result.remember)
    if latest.head_sha != pull.head_sha:
        raise Superseded
    # A finding withdrawn in conversation and found again on unchanged code doesn't block (the verdict and the
    # check below both count with it).
    result.outcome.withdrawn = await publisher.withdrawn()
    # Asking again without pushing gets a fresh attempt, but never a way around a finding that still stands.
    result.outcome.standing = await publisher.standing()
    # A lens that failed after posting: its comments are on the PR, so the verdict counts them (#43's review).
    result.outcome.absorb(Outcome(inline=[f for f in publisher.findings if f not in result.outcome.posted]))
    previous = await post_review(
        deps.github,
        repo,
        pull.number,
        head_sha=pull.head_sha,
        author=deps.bot_login,
        models_note=deps.models_note,
        outcome=result.outcome,
        posted=publisher.posted,
        check_id=check_id,
    )
    return _recovered_result(previous) if previous else result.outcome


async def _unless_superseded(
    deps: ReviewDeps, repo: Repo, pull: PullRequestUpdated, review: Awaitable[Reviewed | CheckResult]
) -> Reviewed | CheckResult:
    """Run `review`, cancelling it as soon as the PR moves on (a newer push, or closed): its result would be
    thrown away, and the newer head's review may be waiting for this slot."""
    reviewing = asyncio.ensure_future(review)
    watching = asyncio.ensure_future(_moved_on(deps, repo, pull))
    try:
        await asyncio.wait({reviewing, watching}, return_when=asyncio.FIRST_COMPLETED)
    finally:
        for task in (reviewing, watching):
            if not task.done():
                task.cancel()
        await asyncio.gather(reviewing, watching, return_exceptions=True)
    if watching.done() and not watching.cancelled() and watching.result():
        raise Superseded
    return reviewing.result()


async def _moved_on(deps: ReviewDeps, repo: Repo, pull: PullRequestUpdated) -> bool:
    """Return True once the PR's head is no longer `pull.head_sha` or the PR is closed; check every minute."""
    while True:
        await asyncio.sleep(SUPERSEDED_CHECK_S)
        try:
            state = await deps.github.get_pull(repo, pull.number)
        except Exception:  # a GitHub hiccup is no reason to stop a review
            logger.warning("could not check whether %s/%s#%d moved on", pull.owner, pull.repo, pull.number)
            continue
        if state.head_sha != pull.head_sha or state.state != "open":
            logger.info(
                "%s/%s#%d moved on from %s; stopping its review", pull.owner, pull.repo, pull.number, pull.head_sha[:7]
            )
            return True


async def _forget_if_closed(deps: ReviewDeps, repo: Repo, pull: PullRequestUpdated, conversation: str) -> None:
    if deps.memory is None:
        return
    try:
        if (await deps.github.get_pull(repo, pull.number)).state != "open":
            deps.memory.forget(conversation)
    except Exception:  # never let cleanup hide the original failure
        logger.exception("could not check whether %s was closed", conversation)


async def _current(deps: ReviewDeps, repo: Repo, pull: PullRequestUpdated) -> PullRequestState:
    state = await deps.github.get_pull(repo, pull.number)
    if state.head_sha != pull.head_sha or state.state != "open":
        raise Superseded
    return state


def _summary(outcome: Outcome) -> CheckResult:
    notes = ""
    blocking = [f for f in outcome.posted if outcome.is_blocking(f)]
    advisory = [f for f in outcome.posted if not outcome.is_blocking(f) and f.kind != "simplify"]
    simplify = [f for f in outcome.posted if f.kind == "simplify"]
    if advisory:
        notes += f" {len(advisory)} advisory (non-blocking) finding(s) posted."
    if simplify:
        notes += f" {len(simplify)} optional simplification(s) suggested."
    if outcome.rejected:
        notes += f" {len(outcome.rejected)} candidate finding(s) were dropped by verification."
    if outcome.unverified:
        notes += f" {len(outcome.unverified)} candidate finding(s) could not be verified and will be checked again."
    if blocking:
        return CheckResult(
            f"{len(blocking)} blocking issue(s)", f"Changes requested; see the review comments.{notes}", "failure"
        )
    if outcome.standing:
        return CheckResult(
            "Earlier blocking finding stands",
            f"Changes requested: a blocking finding from an earlier review of this commit still stands.{notes}",
            "failure",
        )
    if outcome.cut_short:
        return CheckResult(
            "Partially reviewed",
            f"Partially reviewed: {outcome.why_partial}; not approved.{notes}",
            "neutral",
        )
    return CheckResult("No blocking issues", f"Approved.{notes}", "success")
