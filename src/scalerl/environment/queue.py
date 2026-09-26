"""Deterministic request queue and per-tick service-capacity model."""

from __future__ import annotations

import math
from dataclasses import dataclass

from scalerl.environment.config import ReplicaConfig, TimingConfig


@dataclass(frozen=True, slots=True)
class QueueStepResult:
    """Request volumes for one tick. All values are request counts, not rates."""

    arrived_requests: float
    processed_requests: float
    queued_requests: float
    dropped_requests: float


class RequestQueue:
    """Serve backlog plus new arrivals up to active capacity each tick.

    The v1 queue is unbounded: work beyond capacity waits for later ticks and
    nothing is dropped. Request volumes are continuous floats.
    """

    def __init__(self, replicas: ReplicaConfig, timing: TimingConfig) -> None:
        self._requests_per_replica = replicas.service_capacity_rps * timing.control_interval_seconds
        if not math.isfinite(self._requests_per_replica):
            raise ValueError("per-replica capacity per tick overflows; reduce capacity or interval")
        self._control_interval_seconds = timing.control_interval_seconds
        self._queue_depth = 0.0

    @property
    def queue_depth(self) -> float:
        """Return the requests waiting after the most recent tick."""
        return self._queue_depth

    def step(
        self, request_rate: float, active_replicas: int, capacity_multiplier: float = 1.0
    ) -> QueueStepResult:
        """Process one tick of ``request_rate`` demand with ``active_replicas`` serving.

        Tick capacity is ``active_replicas * service_capacity_rps *
        capacity_multiplier * control_interval_seconds`` requests, applied to the
        existing backlog and this tick's arrivals together. ``capacity_multiplier``
        is the tick's realized capacity factor (1.0 in the nominal simulator,
        which leaves the arithmetic unchanged).
        """
        request_rate = float(request_rate)
        if not math.isfinite(request_rate) or request_rate < 0:
            raise ValueError("request_rate must be finite and non-negative")
        if isinstance(active_replicas, bool) or not isinstance(active_replicas, int):
            raise TypeError("active_replicas must be an integer")
        if active_replicas < 0:
            raise ValueError("active_replicas must be non-negative")
        _check_multiplier(capacity_multiplier)

        arrived = request_rate * self._control_interval_seconds
        capacity = active_replicas * (self._requests_per_replica * capacity_multiplier)
        work = self._queue_depth + arrived
        # Finite inputs can still overflow; fail before mutating the queue.
        if not (math.isfinite(work) and math.isfinite(capacity)):
            raise ValueError("request volume overflows; request_rate or replicas too large")
        processed = min(work, capacity)
        self._queue_depth = work - processed

        return QueueStepResult(
            arrived_requests=arrived,
            processed_requests=processed,
            queued_requests=self._queue_depth,
            dropped_requests=0.0,
        )

    def reset(self) -> None:
        """Clear all queued work."""
        self._queue_depth = 0.0


def _check_multiplier(multiplier: float) -> None:
    if isinstance(multiplier, bool) or not isinstance(multiplier, int | float):
        raise TypeError("capacity_multiplier must be a number")
    if not math.isfinite(multiplier) or multiplier <= 0:
        raise ValueError("capacity_multiplier must be finite and positive")
