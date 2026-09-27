"""Deterministic replica lifecycle: pending -> active -> terminating.

Each pending replica has a **physical** remaining startup time (its realized
startup delay, which a startup-delay model may vary per replica, #81) and a
**nominal** remaining time (the configured ``startup_delay_seconds`` minus its
age). Activation uses the physical time; the controller-visible readiness
buckets (:meth:`ReplicaPool.pending_by_ticks_until_active`) use only the
nominal time, so a sampled future startup realization never leaks into what a
controller can see. With the default fixed startup the two are identical.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

from scalerl.environment.config import ReplicaConfig

# Absorbs float rounding when many short advances add up to the startup delay.
_TIME_TOLERANCE_SECONDS = 1e-9


@dataclass(slots=True)
class PendingReplica:
    """One requested replica still starting up (internal lifecycle state)."""

    remaining_seconds: float  # physical: realized delay minus elapsed time
    nominal_remaining_seconds: float  # controller-visible: nominal delay minus elapsed time
    realized_delay_seconds: float
    multiplier: float


@dataclass(frozen=True, slots=True)
class StartedReplica:
    """Physical diagnostics of one replica requested by the latest scale-up."""

    realized_delay_seconds: float
    multiplier: float


class ReplicaPool:
    """Track pending, active, and terminating replicas in simulated time.

    State changes only through :meth:`scale_up`, :meth:`scale_down`,
    :meth:`advance`, and :meth:`reset`. The desired replica count
    (``active + pending``) always stays within the configured bounds;
    requests beyond a bound are clamped, and the applied change is returned.

    ``startup_multiplier`` (optional) draws one startup-delay multiplier per
    newly requested replica, in request order; without it (fixed startup) no
    draw happens and every replica takes exactly ``startup_delay_seconds``.
    """

    def __init__(
        self,
        config: ReplicaConfig,
        *,
        startup_multiplier: Callable[[], float] | None = None,
    ) -> None:
        self._config = config
        self._startup_multiplier = startup_multiplier
        self._active = 0
        self._pending: list[PendingReplica] = []
        self._terminating = 0
        self._last_started: tuple[StartedReplica, ...] = ()
        self.reset()

    @property
    def config(self) -> ReplicaConfig:
        """Return the replica configuration."""
        return self._config

    @property
    def active_count(self) -> int:
        """Return the number of replicas available to serve traffic."""
        return self._active

    @property
    def pending_count(self) -> int:
        """Return the number of requested replicas still starting up."""
        return len(self._pending)

    @property
    def terminating_count(self) -> int:
        """Return the number of replicas awaiting finalization."""
        return self._terminating

    @property
    def desired_count(self) -> int:
        """Return the replica count the pool is converging to (active + pending)."""
        return self._active + self.pending_count

    @property
    def last_started(self) -> tuple[StartedReplica, ...]:
        """Physical startup realizations of the replicas added by the latest scale-up call.

        Physical truth for diagnostics only; never part of a controller's view.
        """
        return self._last_started

    def scale_up(self, count: int = 1) -> int:
        """Request up to ``count`` new replicas and return how many were added.

        New replicas start pending, each with its own realized startup delay.
        With zero startup delay they activate immediately (and nothing is
        drawn). The request is clamped so ``desired_count`` never exceeds
        ``max_replicas``.
        """
        _check_count(count)
        added = min(count, self._config.max_replicas - self.desired_count)
        delay = self._config.startup_delay_seconds
        if delay == 0:
            self._active += added
            self._last_started = tuple(StartedReplica(0.0, 1.0) for _ in range(added))
            return added
        started = []
        for _ in range(added):
            multiplier = 1.0 if self._startup_multiplier is None else self._startup_multiplier()
            realized = delay if multiplier == 1.0 else delay * multiplier
            self._pending.append(PendingReplica(realized, delay, realized, multiplier))
            started.append(StartedReplica(realized, multiplier))
        self._last_started = tuple(started)
        return added

    def scale_down(self, count: int = 1) -> int:
        """Remove up to ``count`` replicas and return how many were removed.

        The most recently requested pending replicas are cancelled first (a
        cancelled replica never activates); any remaining reduction moves
        active replicas to terminating, which stops them serving immediately.
        The request is clamped so ``desired_count`` never falls below
        ``min_replicas``.
        """
        _check_count(count)
        removed = min(count, self.desired_count - self._config.min_replicas)

        cancelled = min(removed, self.pending_count)
        del self._pending[self.pending_count - cancelled :]

        terminated = removed - cancelled
        self._active -= terminated
        self._terminating += terminated
        return removed

    def advance(self, elapsed_seconds: float) -> None:
        """Advance lifecycle state by ``elapsed_seconds`` of simulated time.

        Terminating replicas are finalized, and pending replicas whose
        realized startup delay has fully elapsed become active.
        """
        elapsed_seconds = float(elapsed_seconds)
        if not math.isfinite(elapsed_seconds) or elapsed_seconds <= 0:
            raise ValueError("elapsed_seconds must be finite and greater than zero")

        self._terminating = 0
        for replica in self._pending:
            replica.remaining_seconds -= elapsed_seconds
            replica.nominal_remaining_seconds -= elapsed_seconds
        still_pending = [r for r in self._pending if r.remaining_seconds > _TIME_TOLERANCE_SECONDS]
        self._active += len(self._pending) - len(still_pending)
        self._pending = still_pending

    def pending_by_ticks_until_active(self, interval_seconds: float) -> tuple[int, ...]:
        """Count pending replicas by the **nominal** number of advances until ready.

        Entry ``i`` counts replicas nominally ready after ``i + 1`` advances. The
        length is the number of advances a fresh replica nominally needs (zero
        when startup delay is zero), so it depends only on configuration. This
        is the controller-visible view: it uses each replica's age against the
        nominal startup delay, never its sampled realization. A replica past
        its nominal readiness that is still physically pending (a late stochastic
        startup) is counted in the first bucket ("due now"). Under fixed startup
        this equals the physical readiness schedule.
        """
        interval_seconds = float(interval_seconds)
        if not math.isfinite(interval_seconds) or interval_seconds <= 0:
            raise ValueError("interval_seconds must be finite and greater than zero")

        counts = [0] * startup_ticks(self._config.startup_delay_seconds, interval_seconds)
        for replica in self._pending:
            ticks = _advances_until_active(replica.nominal_remaining_seconds, interval_seconds)
            counts[min(ticks, len(counts)) - 1] += 1
        return tuple(counts)

    def physical_pending_by_ticks_until_active(self, interval_seconds: float) -> tuple[int, ...]:
        """Pending replicas by *realized* advances until active (diagnostics only).

        Physical truth, which may extend beyond the nominal schedule; never
        part of a controller's view.
        """
        interval_seconds = float(interval_seconds)
        if not math.isfinite(interval_seconds) or interval_seconds <= 0:
            raise ValueError("interval_seconds must be finite and greater than zero")
        ticks = [
            _advances_until_active(r.remaining_seconds, interval_seconds) for r in self._pending
        ]
        counts = [0] * max(ticks, default=0)
        for tick in ticks:
            counts[tick - 1] += 1
        return tuple(counts)

    def reset(self) -> None:
        """Return to ``initial_replicas`` active replicas with nothing pending or terminating."""
        self._active = self._config.initial_replicas
        self._pending = []
        self._terminating = 0
        self._last_started = ()


def startup_ticks(startup_delay_seconds: float, control_interval_seconds: float) -> int:
    """Lifecycle advances a newly requested replica needs before it is active (0 = immediate).

    A replica requested by the action at tick ``t`` first serves tick
    ``t + startup_ticks`` (it serves tick ``t`` itself when this is 0).
    """
    delay = float(startup_delay_seconds)
    interval = float(control_interval_seconds)
    if not math.isfinite(delay) or delay < 0:
        raise ValueError("startup_delay_seconds must be finite and non-negative")
    if not math.isfinite(interval) or interval <= 0:
        raise ValueError("control_interval_seconds must be finite and greater than zero")
    if delay == 0:
        return 0
    return _advances_until_active(delay, interval)


def _advances_until_active(remaining_seconds: float, interval_seconds: float) -> int:
    """Mirror :meth:`ReplicaPool.advance`: active once remaining time is within tolerance."""
    return max(1, math.ceil((remaining_seconds - _TIME_TOLERANCE_SECONDS) / interval_seconds))


def _check_count(count: int) -> None:
    if isinstance(count, bool) or not isinstance(count, int):
        raise TypeError("count must be an integer")
    if count <= 0:
        raise ValueError("count must be greater than zero")
