"""Deterministic simulation clock used by ScaleRL environments."""

from __future__ import annotations

import math


class SimulationClock:
    """Track simulated time in fixed-size control intervals.

    The clock is intentionally independent of wall-clock time. It advances only
    when :meth:`step` is called, making simulator trajectories reproducible.
    """

    def __init__(self, tick_seconds: float) -> None:
        tick_seconds = float(tick_seconds)
        if not math.isfinite(tick_seconds) or tick_seconds <= 0:
            raise ValueError("tick_seconds must be finite and greater than zero")

        self._tick_seconds = tick_seconds
        self._step_count = 0

    @property
    def tick_seconds(self) -> float:
        """Return the duration of one simulation tick in seconds."""
        return self._tick_seconds

    @property
    def step_count(self) -> int:
        """Return the number of ticks elapsed since the last reset."""
        return self._step_count

    @property
    def time_seconds(self) -> float:
        """Return total simulated time elapsed in seconds."""
        return self._step_count * self._tick_seconds

    def step(self, ticks: int = 1) -> float:
        """Advance simulated time by ``ticks`` intervals and return new time."""
        if isinstance(ticks, bool) or not isinstance(ticks, int):
            raise TypeError("ticks must be an integer")
        if ticks <= 0:
            raise ValueError("ticks must be greater than zero")

        self._step_count += ticks
        return self.time_seconds

    def reset(self) -> None:
        """Reset simulated time to zero without changing tick duration."""
        self._step_count = 0
