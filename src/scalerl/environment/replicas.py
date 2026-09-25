"""Deterministic replica lifecycle: pending -> active -> terminating."""

from __future__ import annotations

import math

from scalerl.environment.config import ReplicaConfig

# Absorbs float rounding when many short advances add up to the startup delay.
_TIME_TOLERANCE_SECONDS = 1e-9


class ReplicaPool:
    """Track pending, active, and terminating replicas in simulated time.

    State changes only through :meth:`scale_up`, :meth:`scale_down`,
    :meth:`advance`, and :meth:`reset`. The desired replica count
    (``active + pending``) always stays within the configured bounds;
    requests beyond a bound are clamped, and the applied change is returned.
    """

    def __init__(self, config: ReplicaConfig) -> None:
        self._config = config
        self._active = 0
        self._pending_remaining_seconds: list[float] = []
        self._terminating = 0
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
        return len(self._pending_remaining_seconds)

    @property
    def terminating_count(self) -> int:
        """Return the number of replicas awaiting finalization."""
        return self._terminating

    @property
    def desired_count(self) -> int:
        """Return the replica count the pool is converging to (active + pending)."""
        return self._active + self.pending_count

    def scale_up(self, count: int = 1) -> int:
        """Request up to ``count`` new replicas and return how many were added.

        New replicas start pending. With zero startup delay they activate
        immediately. The request is clamped so ``desired_count`` never
        exceeds ``max_replicas``.
        """
        _check_count(count)
        added = min(count, self._config.max_replicas - self.desired_count)
        if self._config.startup_delay_seconds == 0:
            self._active += added
        else:
            self._pending_remaining_seconds.extend([self._config.startup_delay_seconds] * added)
        return added

    def scale_down(self, count: int = 1) -> int:
        """Remove up to ``count`` replicas and return how many were removed.

        The most recently requested pending replicas are cancelled first;
        any remaining reduction moves active replicas to terminating, which
        stops them serving immediately. The request is clamped so
        ``desired_count`` never falls below ``min_replicas``.
        """
        _check_count(count)
        removed = min(count, self.desired_count - self._config.min_replicas)

        cancelled = min(removed, self.pending_count)
        del self._pending_remaining_seconds[self.pending_count - cancelled :]

        terminated = removed - cancelled
        self._active -= terminated
        self._terminating += terminated
        return removed

    def advance(self, elapsed_seconds: float) -> None:
        """Advance lifecycle state by ``elapsed_seconds`` of simulated time.

        Terminating replicas are finalized, and pending replicas whose
        startup delay has fully elapsed become active.
        """
        elapsed_seconds = float(elapsed_seconds)
        if not math.isfinite(elapsed_seconds) or elapsed_seconds <= 0:
            raise ValueError("elapsed_seconds must be finite and greater than zero")

        self._terminating = 0
        remaining = [seconds - elapsed_seconds for seconds in self._pending_remaining_seconds]
        still_pending = [seconds for seconds in remaining if seconds > _TIME_TOLERANCE_SECONDS]
        self._active += len(remaining) - len(still_pending)
        self._pending_remaining_seconds = still_pending

    def pending_by_ticks_until_active(self, interval_seconds: float) -> tuple[int, ...]:
        """Count pending replicas by how many ``advance(interval_seconds)`` calls activate them.

        Entry ``i`` counts replicas that become active after ``i + 1`` advances.
        The length is the number of advances a fresh replica needs (zero when
        startup delay is zero), so it depends only on configuration.
        """
        interval_seconds = float(interval_seconds)
        if not math.isfinite(interval_seconds) or interval_seconds <= 0:
            raise ValueError("interval_seconds must be finite and greater than zero")

        counts = [0] * startup_ticks(self._config.startup_delay_seconds, interval_seconds)
        for seconds in self._pending_remaining_seconds:
            counts[_advances_until_active(seconds, interval_seconds) - 1] += 1
        return tuple(counts)

    def reset(self) -> None:
        """Return to ``initial_replicas`` active replicas with nothing pending or terminating."""
        self._active = self._config.initial_replicas
        self._pending_remaining_seconds = []
        self._terminating = 0


def startup_ticks(startup_delay_seconds: float, control_interval_seconds: float) -> int:
    """Lifecycle advances a newly requested replica needs before it is active (0 = immediate).

    A replica requested by the action at tick ``t`` first serves tick
    ``t + startup_ticks`` (it serves tick ``t`` itself when this is 0).
    """
    if startup_delay_seconds <= 0:
        return 0
    return _advances_until_active(startup_delay_seconds, control_interval_seconds)


def _advances_until_active(remaining_seconds: float, interval_seconds: float) -> int:
    """Mirror :meth:`ReplicaPool.advance`: active once remaining time is within tolerance."""
    return max(1, math.ceil((remaining_seconds - _TIME_TOLERANCE_SECONDS) / interval_seconds))


def _check_count(count: int) -> None:
    if isinstance(count, bool) or not isinstance(count, int):
        raise TypeError("count must be an integer")
    if count <= 0:
        raise ValueError("count must be greater than zero")
