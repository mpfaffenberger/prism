"""Placement hints for lenses; request admission remains the resolved models' job."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from threading import Lock
from time import monotonic

from pydantic_ai.models import Model

from autoreviewer.model_config import PrismConfig
from autoreviewer.review.agents import _lanes  # pyright: ignore[reportPrivateUsage]

# A burst of synchronous lens constructions may precede any provider activity.
# Use fixed (not sliding) windows so even uninterrupted arrivals shed stale hints.
_BURST_SECONDS = 0.25

type Choice = tuple[Model, int]


@dataclass(frozen=True)
class _Resource:
    spec: str
    capacity: int
    choice: Choice

    def demand(self) -> int:
        lanes = _lanes(self.spec, self.capacity)
        return lanes.running_count + lanes.waiting_count


class ModelPool:
    """Pin an ordered finder/checker pair, with no leases or provider switching.

    First use the earliest eligible resource with spare lanes. When all are
    full, minimize demand per lane, breaking ties in config order. Both roles
    see all real requests (including queued requests from other lenses).
    Within a construction burst each stage also counts its own placements:
    finders start first, and an unstarted verifier need not displace a finder.
    These short-lived estimates may overlap real load but never block a caller.
    With a dedicated checker (`checks: "checker"`), the pool places only finders (see `check`).
    """

    def __init__(self, config: PrismConfig):
        if not config.pool:
            raise ValueError("ModelPool requires a configured pool")
        self._resources = [_Resource(spec, capacity, config.resolve(spec)) for spec, capacity in config.pool.items()]
        if len({resource.choice[0].model_name for resource in self._resources}) < 2:
            raise ValueError("pool needs two different resolved model names for independent verification")
        self._checker = config.resolve(config.checker) if config.checks == "checker" else None
        self._lock = Lock()
        self._expires = 0.0
        self._finders: dict[str, int] = {}
        self._checkers: dict[str, int] = {}

    def key(self, model: Model) -> str:
        return next(resource.spec for resource in self._resources if resource.choice[0] is model)

    def check(self, finder: str) -> Choice:
        """Place a check of findings by the model named `finder`: on the dedicated checker if there is one, unless
        it is the finder's own model; then, like every check without one, on another pool model."""
        if self._checker is not None and self._checker[0].model_name != finder:
            return self._checker
        return self.choose(excluding=finder)

    def choose(self, *, excluding: str | None = None) -> Choice:
        """Place a new run, excluding the originating finder when placing a check."""
        with self._lock:
            now = monotonic()
            if now >= self._expires:
                self._finders.clear()
                self._checkers.clear()
                self._expires = now + _BURST_SECONDS
            hints = self._checkers if excluding is not None else self._finders

            def pressure(resource: _Resource) -> Fraction:
                demand = resource.demand() + hints.get(resource.spec, 0)
                return Fraction(0) if demand < resource.capacity else Fraction(demand, resource.capacity)

            resource = min((r for r in self._resources if r.choice[0].model_name != excluding), key=pressure)
            hints[resource.spec] = hints.get(resource.spec, 0) + 1
            return resource.choice
