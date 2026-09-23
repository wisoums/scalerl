"""Deterministic request-rate traces and their replay state."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True, slots=True, init=False)
class WorkloadTrace:
    """Immutable request-rate trace with one sample per simulation tick.

    Sample ``i`` is the demand, in requests per second, during tick ``i``.
    Synthetic generators and recorded traces share this representation.
    """

    request_rates: tuple[float, ...]
    control_interval_seconds: float

    def __init__(self, request_rates: Iterable[float], control_interval_seconds: float) -> None:
        rates = tuple(float(rate) for rate in request_rates)
        if not rates:
            raise ValueError("request_rates must contain at least one sample")
        for tick, rate in enumerate(rates):
            if not math.isfinite(rate) or rate < 0:
                raise ValueError(
                    f"request rate at tick {tick} must be finite and non-negative, got {rate}"
                )

        interval = float(control_interval_seconds)
        if not math.isfinite(interval) or interval <= 0:
            raise ValueError("control_interval_seconds must be finite and greater than zero")

        object.__setattr__(self, "request_rates", rates)
        object.__setattr__(self, "control_interval_seconds", interval)

    def __len__(self) -> int:
        """Return the number of ticks in the trace."""
        return len(self.request_rates)

    @property
    def duration_seconds(self) -> float:
        """Return the simulated time covered by the trace."""
        return len(self) * self.control_interval_seconds

    def demand_at(self, tick: int) -> float:
        """Return the request rate for ``tick``; out-of-range ticks raise instead of wrapping."""
        if isinstance(tick, bool) or not isinstance(tick, int):
            raise TypeError("tick must be an integer")
        if not 0 <= tick < len(self):
            raise IndexError(f"tick {tick} is outside trace of length {len(self)}")
        return self.request_rates[tick]


class WorkloadReplay:
    """Sequential cursor over a :class:`WorkloadTrace`.

    The trace stays immutable; only the replay position changes.
    """

    def __init__(self, trace: WorkloadTrace) -> None:
        self._trace = trace
        self._position = 0

    @property
    def trace(self) -> WorkloadTrace:
        """Return the trace being replayed."""
        return self._trace

    @property
    def position(self) -> int:
        """Return the tick whose demand the next call to :meth:`next_demand` returns."""
        return self._position

    @property
    def remaining(self) -> int:
        """Return the number of ticks not yet replayed."""
        return len(self._trace) - self._position

    @property
    def is_exhausted(self) -> bool:
        """Return whether every tick of the trace has been replayed."""
        return self.remaining == 0

    def next_demand(self) -> float:
        """Return the current tick's request rate and advance by one tick."""
        if self.is_exhausted:
            raise IndexError("workload trace is exhausted; call reset() to replay it")
        demand = self._trace.demand_at(self._position)
        self._position += 1
        return demand

    def reset(self) -> None:
        """Rewind replay to the first tick of the trace."""
        self._position = 0
