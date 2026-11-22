"""Time limits for a review.

On its next step, a lens with five minutes left is asked to focus. With one minute left it loses
its function tools and must return its final report. Both windows shrink proportionally for soft
budgets under the default (`DEFAULT_SOFT_S`). After soft expiry the instruction says time is up.

Checking stops waiting at the checking cutoff, shortly before the hard deadline: a lens that still
hasn't reported, and any candidate still unjudged, is left out, and the verdict is posted from what
was judged by then. The hard deadline (`asyncio.timeout` in the job) is the backstop that cancels
whatever remains. A review cut short by any of these limits may request changes for what it found,
but incomplete blocking work never approves.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable, Collection, Container
from dataclasses import dataclass, field

from pydantic_ai import RunContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import ModelRequest, UserPromptPart
from pydantic_ai.models import ModelRequestContext
from pydantic_ai.tools import ToolDefinition

MAX_REVIEW_S = 1200.0
"""No review attempt runs longer than this, whatever a deployment configures: the one definition of the cap."""
DEFAULT_SOFT_S = 900.0
"""The default soft budget leaves a quarter of the cap for checking and sets the scale for the countdown."""
POSTING_RESERVE_S = 60.0
"""Kept between the checking cutoff and the hard stop, for reading the PR's state and posting the verdict."""
FOCUS_S = 300
REPORT_ONLY_S = 60

FIVE_MINUTES = (
    "Five minutes or less remain before review wrap-up. Prioritize unresolved high-impact evidence; "
    "finish your report as soon as coverage is complete. Do not start speculative investigations."
)
ONE_MINUTE = (
    "One minute or less remains before review wrap-up. Finish immediately: return the final structured "
    "result using evidence already gathered. Do not claim coverage you did not complete."
)

WRAP_UP = (
    "Time is up for this review. Stop exploring: return your final report now with what you have found so "
    "far, and list in `reviewed` only the files you actually inspected."
)


@dataclass
class Deadline:
    soft_s: float
    """Seconds after `started` when every lens must wrap up."""
    started: float = field(default_factory=time.monotonic)
    wrapped_up: set[str] = field(default_factory=set[str])
    """Labels of lenses forced to report, whether by instruction or by removal of function tools, or cut off
    at the checking cutoff before they reported."""
    checking_s: float | None = None
    """Seconds after `started` when checking stops waiting (see `checking_cutoff`). None waits for every check."""

    @property
    def remaining(self) -> float:
        return self.soft_s - (time.monotonic() - self.started)

    @property
    def checking_remaining(self) -> float | None:
        """Seconds until the checking cutoff (never negative), or None without one."""
        if self.checking_s is None:
            return None
        return max(0.0, self.checking_s - (time.monotonic() - self.started))

    @property
    def past_cutoff(self) -> bool:
        """Whether the checking cutoff has come: nothing new is started after it."""
        return self.checking_remaining == 0

    def watch(self, label: str) -> AbstractCapability[None]:
        """Warn the labelled lens as time runs short and force it to report before soft expiry."""
        return _WrapUp(self, label)


@dataclass
class _WrapUp(AbstractCapability[None]):
    deadline: Deadline
    label: str
    last_warning: str | None = None

    def _notice(self) -> str | None:
        remaining = self.deadline.remaining
        scale = min(1.0, max(0.0, self.deadline.soft_s) / DEFAULT_SOFT_S)
        if remaining <= REPORT_ONLY_S * scale:
            self.deadline.wrapped_up.add(self.label)
            return WRAP_UP if remaining <= 0 else ONE_MINUTE
        if remaining <= FOCUS_S * scale:
            return FIVE_MINUTES
        return None

    async def prepare_tools(self, ctx: RunContext[None], tool_defs: list[ToolDefinition]) -> list[ToolDefinition]:
        # Returning [] removes only function tools. The separate output tools still allow a final report.
        return [] if self._notice() in (ONE_MINUTE, WRAP_UP) else tool_defs

    async def before_model_request(
        self, ctx: RunContext[None], request_context: ModelRequestContext
    ) -> ModelRequestContext:
        warning = self._notice()
        if warning is None:
            return request_context
        last = request_context.messages[-1]
        if isinstance(last, ModelRequest) and warning != self.last_warning:
            last.parts = [*last.parts, UserPromptPart(warning)]
            self.last_warning = warning
        return request_context


def checking_cutoff(hard_s: float) -> float:
    """Seconds into a review when checking stops waiting, so a verdict goes out before the hard stop at
    `hard_s`: `POSTING_RESERVE_S` earlier, or a tenth of a budget too short for that."""
    return hard_s - min(POSTING_RESERVE_S, hard_s / 10)


async def before_cutoff[T](deadline: Deadline | None, start: Callable[[], Awaitable[T]]) -> T | None:
    """The result of the work `start` begins, or None if the checking cutoff comes first. Work under way is
    cancelled then, and work not yet begun is never started: a timeout of zero would still let it run to its
    first suspension, far enough to send a model request or even to finish."""
    if deadline is not None and deadline.past_cutoff:
        return None
    cutoff = asyncio.timeout(deadline.checking_remaining if deadline else None)  # None never expires
    try:
        async with cutoff:
            return await start()
    except TimeoutError:
        if not cutoff.expired():  # the work's own timeout, not the cutoff
            raise
        return None


class Finishing:
    """The tasks past their decision, with only its result left to write down (posting a confirmed finding,
    say). `until_cutoff` lets those complete: a post cancelled after it reached GitHub would go unrecorded and
    be posted again with the verdict."""

    def __init__(self) -> None:
        self._tasks: set[asyncio.Task[object]] = set()

    def mark(self) -> None:
        """The calling task has decided. Call this before its first write, with no await in between."""
        if (task := asyncio.current_task()) is not None:
            self._tasks.add(task)

    def __contains__(self, task: object) -> bool:
        return task in self._tasks


async def until_cutoff[T](
    deadline: Deadline | None,
    tasks: Collection[asyncio.Task[T]],
    *,
    finishing: Container[object] = (),
) -> list[asyncio.Task[T]]:
    """Wait for `tasks` until the checking cutoff, then cancel and return the ones still running. A task
    that finished with an error raises here, as `asyncio.gather` would.

    A task in `finishing` (see `Finishing`) is left to complete, not cancelled and not returned."""
    if not tasks:
        return []
    done, pending = await asyncio.wait(tasks, timeout=deadline.checking_remaining if deadline else None)
    cut = [task for task in tasks if task in pending and task not in finishing]
    for task in cut:
        task.cancel()
    await asyncio.gather(*pending, return_exceptions=True)  # cancellations land, and what was finishing completes
    for task in done:
        task.result()
    return cut
