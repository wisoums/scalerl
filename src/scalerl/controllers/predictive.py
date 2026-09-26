"""Predictive autoscaler: a transparent linear-trend forecast of recent demand.

Information set: the forecast uses only ``info["request_rate"]`` of completed
ticks; capacity sizing additionally reads the current backlog
``info["queued_requests"]`` of the latest completed tick. The controller never
reads the workload trace, the RL observation, or generator parameters, so it
cannot know a future value before that tick has run. The queue never feeds
the forecast.

Each decision (v1 defaults in parentheses):

1. add the latest completed tick's request rate to a window of the most
   recent ``history_window_ticks`` samples (4);
2. forecast demand ``forecast_horizon_ticks`` ahead of the latest sample:
   persistence with one sample, otherwise an ordinary least-squares line over
   the window's ``(tick, rate)`` points, extrapolated; negative forecasts are
   clamped to 0;
3. size capacity for forecast arrivals plus backlog recovery
   (capacity policy ``forecast-plus-backlog-v1``):
   ``backlog_recovery_rps = queued_requests / control_interval_seconds``
   (clear the current backlog over one control interval),
   ``effective_demand_rps = forecast_rps + backlog_recovery_rps``, and
   ``desired = ceil(effective_demand_rps / (service_capacity_rps * target_utilization))``
   (target utilization 0.8), clamped to ``[min_replicas, max_replicas]``;
4. compare with committed capacity ``active + pending``: scale up if below,
   otherwise hold, and scale down only when no requests are queued (a
   scale-down cancels the newest pending replica first). Never removing
   capacity while backlog remains may keep capacity longer, trading cost for
   backlog/SLA recovery.

**Action contracts (#79).** Under ``delta-v1`` (the default, and the only
contract before #79) the decision is emitted as one step toward ``desired``:
code 2 / 0 / 1, i.e. at most one replica per tick. Under
``desired-replicas-v1`` the *same* computed ``desired`` is encoded directly as
the target (``committed`` when the decision holds, e.g. ``backlog_hold``), so
the whole correction is requested in one decision and still waits out the
startup delay. Forecast, history, target utilization and backlog recovery are
identical under both contracts.

With an empty queue the rule is exactly the forecast-only sizing of #14.

**Horizon.** A decision is made after tick ``t`` completes; its scale-up is
applied during tick ``t + 1`` and the new replica first serves tick
``t + 1 + startup_ticks`` (see :func:`scalerl.environment.startup_ticks`,
the simulator's own lifecycle arithmetic). So
``forecast_horizon_ticks = 1 + startup_ticks``: the forecast targets the first
tick that capacity requested now can actually serve. With 60 s startup and
30 s ticks that is 3 ticks (90 s) ahead.

A truly random spike with no prior trend cannot be forecast. Backlog
recovery only helps after overload has actually built a queue.
"""

from __future__ import annotations

import math
import statistics
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from scalerl.environment.actions import HOLD, SCALE_DOWN, SCALE_UP, ActionContract
from scalerl.environment.config import DELTA_V1, SimulatorConfig
from scalerl.environment.gym_env import Observation
from scalerl.environment.replicas import startup_ticks

PredictiveReason = Literal[
    "no_sample",
    "scale_up",
    "queue_recovery",
    "scale_down",
    "backlog_hold",
    "at_target",
    "at_min",
    "at_max",
]
FORECAST_METHOD = "linear-trend"
CAPACITY_POLICY = "forecast-plus-backlog-v1"
BACKLOG_RECOVERY_TICKS = 1  # fixed v1 rule, not a tuning parameter
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
    """Why the controller chose its latest action.

    ``forecast_rps`` is the arrival forecast alone; ``effective_demand_rps``
    adds ``backlog_recovery_rps`` for requests already queued. Reasons:
    ``scale_up`` (the forecast alone needs more capacity), ``queue_recovery``
    (only the backlog term does), ``backlog_hold`` (a scale-down blocked
    because requests are still queued). ``action`` is the emitted environment
    action code; ``target_replicas`` is the committed capacity it requests
    (set only when the controller was given an action contract).
    """

    latest_request_rate: float | None
    forecast_rps: float | None
    forecast_horizon_ticks: int
    queued_requests: float | None
    backlog_recovery_rps: float | None
    effective_demand_rps: float | None
    desired_replicas: int | None
    active_replicas: int
    pending_replicas: int
    action: int
    reason: PredictiveReason
    target_replicas: int | None = None


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
        action_contract: ActionContract | None = None,
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
        self._contract = action_contract
        self._capacity_per_replica = float(service_capacity_rps) * float(target_utilization)
        self._control_interval = float(control_interval_seconds)
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
        action_contract: ActionContract | None = None,
    ) -> PredictiveController:
        """Take bounds, capacity, timing, and the action contract from ``config``.

        The controller always acts under ``config``'s action contract, so it
        can never emit ``delta-v1`` codes into a ``desired-replicas-v1``
        environment. An explicit ``action_contract`` is accepted only if it is
        exactly that contract.
        """
        derived = ActionContract.from_config(config)
        if action_contract is not None and action_contract != derived:
            raise ValueError(
                f"action_contract {action_contract} does not match the config's {derived}"
            )
        return cls(
            min_replicas=config.replicas.min_replicas,
            max_replicas=config.replicas.max_replicas,
            service_capacity_rps=config.replicas.service_capacity_rps,
            startup_delay_seconds=config.replicas.startup_delay_seconds,
            control_interval_seconds=config.timing.control_interval_seconds,
            history_window_ticks=history_window_ticks,
            target_utilization=target_utilization,
            action_contract=derived,
        )

    @property
    def action_contract(self) -> ActionContract | None:
        return self._contract

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
            code, target = self._encode(HOLD, active + pending, active + pending)
            self._last_decision = PredictiveDecision(
                None,
                None,
                self._horizon,
                None,
                None,
                None,
                None,
                active,
                pending,
                code,
                "no_sample",
                target,
            )
            return code

        source_tick, latest = self._samples[-1]
        forecast = self._forecast(source_tick + self._horizon)
        self._forecasts.append(
            ForecastRecord(source_tick, source_tick + self._horizon, forecast, len(self._samples))
        )

        queued = _queued_requests(info)
        self._last_decision = self._decide(latest, forecast, queued, active, pending)
        return self._last_decision.action

    def _decide(
        self, latest: float, forecast: float, queued: float, active: int, pending: int
    ) -> PredictiveDecision:
        """Size capacity for forecast arrivals plus backlog recovery and pick an action."""
        recovery = backlog_recovery_rate(queued, self._control_interval)
        effective = effective_sizing_demand(forecast, queued, self._control_interval)
        needed, desired = self._size(effective)
        _, forecast_only_desired = self._size(forecast)
        committed = active + pending

        action: int
        reason: PredictiveReason
        if desired > committed:
            action = SCALE_UP
            reason = "scale_up" if forecast_only_desired > committed else "queue_recovery"
        elif desired < committed:
            if queued > 0:
                action, reason = HOLD, "backlog_hold"  # never remove capacity into a backlog
            else:
                action, reason = SCALE_DOWN, "scale_down"
        elif needed > self._max_replicas:
            action, reason = HOLD, "at_max"
        elif needed < self._min_replicas:
            action, reason = HOLD, "at_min"
        else:
            action, reason = HOLD, "at_target"

        action, target = self._encode(action, desired if action != HOLD else committed, committed)
        return PredictiveDecision(
            latest_request_rate=latest,
            forecast_rps=forecast,
            forecast_horizon_ticks=self._horizon,
            queued_requests=queued,
            backlog_recovery_rps=recovery,
            effective_demand_rps=effective,
            desired_replicas=desired,
            active_replicas=active,
            pending_replicas=pending,
            action=action,
            reason=reason,
            target_replicas=target,
        )

    def _encode(self, delta_code: int, target: int, committed: int) -> tuple[int, int | None]:
        """Emit the decision under the action contract: ``(code, requested target)``.

        Without a contract, or under ``delta-v1``, the historical one-step code is
        emitted unchanged; under ``desired-replicas-v1`` the target itself.
        """
        if self._contract is None:
            return delta_code, None
        if self._contract.semantics == DELTA_V1:
            code = delta_code
        else:
            code = self._contract.code_for_target(target, committed)
        return code, self._contract.target_for(code, committed)

    def _size(self, demand_rps: float) -> tuple[int, int]:
        """Return ``(needed, desired)`` replicas for ``demand_rps`` with headroom."""
        needed = math.ceil(demand_rps / self._capacity_per_replica - _CEIL_TOLERANCE)
        return needed, min(max(needed, self._min_replicas), self._max_replicas)

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


def backlog_recovery_rate(queued_requests: float, control_interval_seconds: float) -> float:
    """Service rate that clears ``queued_requests`` in one control interval."""
    queued = _require_non_negative("queued_requests", queued_requests)
    interval = float(control_interval_seconds)
    if not math.isfinite(interval) or interval <= 0:
        raise ValueError("control_interval_seconds must be finite and greater than zero")
    return queued / (BACKLOG_RECOVERY_TICKS * interval)


def effective_sizing_demand(
    forecast_rps: float, queued_requests: float, control_interval_seconds: float
) -> float:
    """Forecast arrivals plus backlog recovery: the demand capacity is sized for."""
    forecast = _require_non_negative("forecast_rps", forecast_rps)
    return forecast + backlog_recovery_rate(queued_requests, control_interval_seconds)


def _queued_requests(info: Mapping[str, Any]) -> float:
    if "queued_requests" not in info:
        raise ValueError("info must include queued_requests once a tick has completed")
    return _require_non_negative("queued_requests", info["queued_requests"])


def _require_non_negative(name: str, value: object) -> float:
    _require_number(name, value)
    assert isinstance(value, int | float)
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return float(value)


def _require_int(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")


def _require_number(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{name} must be a number")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
