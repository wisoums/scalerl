"""Predictive autoscaler: a transparent linear-trend forecast of recent demand.

Information set: only ``info["request_rate"]`` of completed ticks. The
controller never reads the workload trace, the RL observation, or generator
parameters, so it cannot know a future value before that tick has run.

Each decision (v1 defaults in parentheses):

1. add the latest completed tick's request rate to a window of the most
   recent ``history_window_ticks`` samples (4);
2. forecast demand ``forecast_horizon_ticks`` ahead of the latest sample:
   persistence with one sample, otherwise an ordinary least-squares line over
   the window's ``(tick, rate)`` points, extrapolated; negative forecasts are
   clamped to 0;
3. size capacity with headroom:
   ``desired = ceil(forecast / (service_capacity_rps * target_utilization))``
   (target utilization 0.8), clamped to ``[min_replicas, max_replicas]``;
4. compare with committed capacity ``active + pending``: scale up if below,
   down if above (a scale-down cancels the newest pending replica first),
   otherwise hold. The environment moves at most one replica per tick.

**Horizon.** A decision is made after tick ``t`` completes; its scale-up is
applied during tick ``t + 1`` and the new replica first serves tick
``t + 1 + startup_ticks`` (see :func:`scalerl.environment.startup_ticks`,
the simulator's own lifecycle arithmetic). So
``forecast_horizon_ticks = 1 + startup_ticks``: the forecast targets the first
tick that capacity requested now can actually serve. With 60 s startup and
30 s ticks that is 3 ticks (90 s) ahead.

A truly random spike with no prior trend cannot be forecast; this baseline
only helps when recent demand carries a signal.
"""

from __future__ import annotations

import math
import statistics
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from scalerl.environment.config import SimulatorConfig
from scalerl.environment.gym_env import HOLD, SCALE_DOWN, SCALE_UP, Observation
from scalerl.environment.replicas import startup_ticks

PredictiveReason = Literal["no_sample", "scale_up", "scale_down", "at_target", "at_min", "at_max"]
FORECAST_METHOD = "linear-trend"
_CEIL_TOLERANCE = 1e-9


@dataclass(frozen=True, slots=True)
class ForecastRecord:
    """A forecast as issued: no actual value is ever stored here."""

    source_tick: int
    target_tick: int
    forecast_rps: float
    sample_count: int


@dataclass(frozen=True, slots=True)
class PredictiveDecision:
    """Why the controller chose its latest action."""

    latest_request_rate: float | None
    forecast_rps: float | None
    forecast_horizon_ticks: int
    desired_replicas: int | None
    active_replicas: int
    pending_replicas: int
    action: int
    reason: PredictiveReason


class PredictiveController:
    """Scale toward the capacity a linear-trend demand forecast needs."""

    def __init__(
        self,
        *,
        min_replicas: int,
        max_replicas: int,
        service_capacity_rps: float,
        startup_delay_seconds: float,
        control_interval_seconds: float,
        history_window_ticks: int = 4,
        target_utilization: float = 0.8,
    ) -> None:
        for name, count in (("min_replicas", min_replicas), ("max_replicas", max_replicas)):
            _require_int(name, count)
        if min_replicas < 1:
            raise ValueError("min_replicas must be at least 1")
        if max_replicas < min_replicas:
            raise ValueError("max_replicas must be at least min_replicas")
        _require_int("history_window_ticks", history_window_ticks)
        if history_window_ticks < 1:
            raise ValueError("history_window_ticks must be at least 1")
        _require_number("target_utilization", target_utilization)
        if not 0 < target_utilization <= 1:
            raise ValueError("target_utilization must satisfy 0 < target_utilization <= 1")
        _require_number("service_capacity_rps", service_capacity_rps)
        if service_capacity_rps <= 0:
            raise ValueError("service_capacity_rps must be greater than zero")
        _require_number("startup_delay_seconds", startup_delay_seconds)
        if startup_delay_seconds < 0:
            raise ValueError("startup_delay_seconds must be non-negative")
        _require_number("control_interval_seconds", control_interval_seconds)
        if control_interval_seconds <= 0:
            raise ValueError("control_interval_seconds must be greater than zero")

        self._min_replicas = min_replicas
        self._max_replicas = max_replicas
        self._capacity_per_replica = float(service_capacity_rps) * float(target_utilization)
        self._history_window = history_window_ticks
        self._target_utilization = float(target_utilization)
        self._horizon = 1 + startup_ticks(
            float(startup_delay_seconds), float(control_interval_seconds)
        )
        self._samples: deque[tuple[int, float]] = deque(maxlen=history_window_ticks)
        self._forecasts: list[ForecastRecord] = []
        self._consumed_tick: Any = None
        self._last_decision: PredictiveDecision | None = None

    @classmethod
    def from_config(
        cls,
        config: SimulatorConfig,
        *,
        history_window_ticks: int = 4,
        target_utilization: float = 0.8,
    ) -> PredictiveController:
        """Take bounds, capacity, and timing from the simulator configuration."""
        return cls(
            min_replicas=config.replicas.min_replicas,
            max_replicas=config.replicas.max_replicas,
            service_capacity_rps=config.replicas.service_capacity_rps,
            startup_delay_seconds=config.replicas.startup_delay_seconds,
            control_interval_seconds=config.timing.control_interval_seconds,
            history_window_ticks=history_window_ticks,
            target_utilization=target_utilization,
        )

    @property
    def history_window_ticks(self) -> int:
        return self._history_window

    @property
    def target_utilization(self) -> float:
        return self._target_utilization

    @property
    def forecast_horizon_ticks(self) -> int:
        """Ticks from the latest completed sample to the first tick new capacity serves."""
        return self._horizon

    @property
    def request_history(self) -> tuple[float, ...]:
        """Observed request rates in the window, oldest first."""
        return tuple(rate for _, rate in self._samples)

    @property
    def forecasts(self) -> tuple[ForecastRecord, ...]:
        """Every forecast issued since reset, in order."""
        return tuple(self._forecasts)

    @property
    def last_decision(self) -> PredictiveDecision | None:
        return self._last_decision

    def reset(self, seed: int | None = None) -> None:
        """Clear history, forecasts, and diagnostics; ``seed`` is unused (deterministic)."""
        self._samples.clear()
        self._forecasts.clear()
        self._consumed_tick = None
        self._last_decision = None

    def act(self, observation: Observation, info: Mapping[str, Any]) -> int:
        self._record_sample(info)
        active, pending = int(info["active_replicas"]), int(info["pending_replicas"])

        if not self._samples:
            self._last_decision = PredictiveDecision(
                None, None, self._horizon, None, active, pending, HOLD, "no_sample"
            )
            return HOLD

        source_tick, latest = self._samples[-1]
        forecast = self._forecast(source_tick + self._horizon)
        self._forecasts.append(
            ForecastRecord(source_tick, source_tick + self._horizon, forecast, len(self._samples))
        )

        needed = math.ceil(forecast / self._capacity_per_replica - _CEIL_TOLERANCE)
        desired = min(max(needed, self._min_replicas), self._max_replicas)
        committed = active + pending

        reason: PredictiveReason
        if desired > committed:
            action, reason = SCALE_UP, "scale_up"
        elif desired < committed:
            action, reason = SCALE_DOWN, "scale_down"
        elif needed > self._max_replicas:
            action, reason = HOLD, "at_max"
        elif needed < self._min_replicas:
            action, reason = HOLD, "at_min"
        else:
            action, reason = HOLD, "at_target"

        self._last_decision = PredictiveDecision(
            latest, forecast, self._horizon, desired, active, pending, action, reason
        )
        return action

    def _record_sample(self, info: Mapping[str, Any]) -> None:
        """Add the latest completed tick's demand once, keyed by its tick."""
        if "request_rate" not in info:
            return  # reset info: no completed tick yet
        tick = info.get("tick")
        if tick is not None and tick == self._consumed_tick:
            return
        self._consumed_tick = tick
        index = (
            int(tick) if tick is not None else (self._samples[-1][0] + 1 if self._samples else 0)
        )
        self._samples.append((index, float(info["request_rate"])))

    def _forecast(self, target_tick: int) -> float:
        """Persistence for one sample, else an OLS line extrapolated to ``target_tick``."""
        ticks = [float(tick) for tick, _ in self._samples]
        rates = [rate for _, rate in self._samples]
        if len(rates) == 1:
            return max(0.0, rates[0])
        slope, intercept = statistics.linear_regression(ticks, rates)
        return max(0.0, intercept + slope * target_tick)


def _require_int(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")


def _require_number(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{name} must be a number")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
