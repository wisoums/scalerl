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
        self._control_interval_seconds = timing.control_interval_seconds
        self._queue_depth = 0.0

    @property
    def queue_depth(self) -> float:
        """Return the requests waiting after the most recent tick."""
        return self._queue_depth

    def step(self, request_rate: float, active_replicas: int) -> QueueStepResult:
        """Process one tick of ``request_rate`` demand with ``active_replicas`` serving.

        Tick capacity is ``active_replicas * service_capacity_rps *
        control_interval_seconds`` requests, applied to the existing backlog and
        this tick's arrivals together.
        """
        request_rate = float(request_rate)
        if not math.isfinite(request_rate) or request_rate < 0:
            raise ValueError("request_rate must be finite and non-negative")
        if isinstance(active_replicas, bool) or not isinstance(active_replicas, int):
            raise TypeError("active_replicas must be an integer")
        if active_replicas < 0:
            raise ValueError("active_replicas must be non-negative")

        arrived = request_rate * self._control_interval_seconds
        capacity = active_replicas * self._requests_per_replica
        work = self._queue_depth + arrived
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
