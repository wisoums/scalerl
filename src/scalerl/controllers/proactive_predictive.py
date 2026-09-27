"""Cloud-style proactive predictive baseline ``predictive-seasonal-v1`` (#80).

A separate controller next to the unchanged ``predictive-v1`` (``linear-trend`` +
``forecast-plus-backlog-v1``). It is inspired by the core idea of cloud
predictive scaling (historical recurring patterns + capacity provisioned in
advance of startup delay); it is **not** an emulation of any provider's
service. Everything is fixed (predeclared), transparent, and deterministic.

**Forecast method ``historical-profile-plus-linear-v1``.** At each decision,
after tick ``s`` (the latest completed sample) has run:

* horizon ``h = 1 + startup_ticks(startup_delay, control_interval)`` (the exact
  ``predictive-v1`` arithmetic; 3 ticks = 90 s at 60 s startup / 30 s ticks), so
  both forecasts target tick ``t = s + h``, the first tick capacity requested
  now can serve;
* ``linear`` = the ``predictive-v1`` linear-trend forecast of tick ``t`` from
  the latest ``history_window_ticks`` (4) completed samples;
* with a :class:`HistoricalDemandProfile` (built from TRAIN traces only) that
  covers tick ``t``: ``profile = profile[t] * level_factor``, where
  ``level_factor`` is the median of ``observed_i / profile[tick_i]`` over the
  same (at most 4) completed samples whose profile value is at least
  ``MIN_PROFILE_RPS_FOR_RATIO``; with no valid ratio it is 1.0;
* ``proactive = max(linear, profile)`` when a profile value exists, else
  ``linear`` (no blend weight).

**Capacity policy ``proactive-scaleout-conservative-scalein-v1``:**

* sizing demand = ``proactive + queued_requests / control_interval`` (the #63
  backlog recovery), ``desired = ceil(demand / (capacity * 0.8))`` clamped to
  the replica bounds; committed capacity is ``active + pending``;
* **scale out directly**: if ``desired > committed`` the full target is
  requested at once (under ``desired-replicas-v1``; the replicas still start
  pending and wait out the startup delay);
* **never scale in with a backlog**: if requests are queued, hold;
* **scale in conservatively**: otherwise ``safe = max(observed_desired,
  desired)`` where ``observed_desired`` sizes the latest *observed* demand; if
  ``safe < committed`` the target is ``max(safe, committed - 1)``: at most one
  replica fewer per decision, never below observed or forecast need.

Targets are encoded through the environment's :class:`ActionContract` (taken
from the simulator config). The controller only reads completed-tick
``request_rate`` and ``queued_requests`` from ``info`` and the replica counts;
it never sees the trace, generator parameters, or any future value.
Every decision with a forecast leaves a :class:`ProactiveForecastRecord`
(nothing actual is stored; scoring joins actual demand after the episode).
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal

from scalerl.benchmarks import build_workloads, load_benchmark_manifest
from scalerl.controllers.predictive import (
    backlog_recovery_rate,
    linear_trend_forecast,
    size_replicas,
)
from scalerl.environment.actions import ActionContract
from scalerl.environment.config import SimulatorConfig
from scalerl.environment.gym_env import Observation
from scalerl.environment.replicas import startup_ticks
from scalerl.workloads import WorkloadTrace

VARIANT: Final = "predictive-seasonal-v1"
FORECAST_METHOD: Final = "historical-profile-plus-linear-v1"
CAPACITY_POLICY: Final = "proactive-scaleout-conservative-scalein-v1"
PROFILE_VERSION: Final = "historical-demand-profile-v1"
PROFILE_AGGREGATION: Final = "median"
HISTORY_WINDOW_TICKS: Final = 4
TARGET_UTILIZATION: Final = 0.8
# Profile values below this are too close to zero for a meaningful observed/profile
# ratio (predeclared; far below any rate the 30 s benchmark traces can resolve).
MIN_PROFILE_RPS_FOR_RATIO: Final = 1e-3

ProactiveReason = Literal[
    "no_sample",
    "scale_out",
    "scale_in",
    "backlog_hold",
    "hold_for_observed_or_forecast",
    "at_target",
]


# --- historical profile -----------------------------------------------------------------------


@dataclass(frozen=True)
class HistoricalDemandProfile:
    """Expected demand by aligned tick position: the per-tick median of TRAIN traces.

    ``rates[k] = median(trace_1[k], ..., trace_n[k])``. All source traces must
    have the same control interval and length (the tick alignment is the
    position within equally aligned windows, e.g. the same hour of different
    days). Immutable; ``profile_id`` hashes its full content.
    """

    source_workload_ids: tuple[str, ...]
    control_interval_seconds: float
    rates: tuple[float, ...]
    aggregation: Literal["median"] = PROFILE_AGGREGATION
    version: Literal["historical-demand-profile-v1"] = PROFILE_VERSION

    def __post_init__(self) -> None:
        if not self.source_workload_ids or not self.rates:
            raise ValueError("a historical profile needs sources and at least one tick")
        if len(set(self.source_workload_ids)) != len(self.source_workload_ids):
            raise ValueError("profile source workloads must be unique")
        if not all(math.isfinite(rate) and rate >= 0 for rate in self.rates):
            raise ValueError("profile rates must be finite and non-negative")

    @classmethod
    def from_traces(cls, traces: Mapping[str, WorkloadTrace]) -> HistoricalDemandProfile:
        """Median profile of ``traces`` (keyed by source ID) with identical timing."""
        if not traces:
            raise ValueError("a historical profile needs at least one trace")
        intervals = {trace.control_interval_seconds for trace in traces.values()}
        if len(intervals) != 1:
            raise ValueError(f"profile traces need one control interval, got {sorted(intervals)}")
        lengths = {len(trace) for trace in traces.values()}
        if len(lengths) != 1:
            raise ValueError(
                f"profile traces need equal lengths (aligned ticks), got {sorted(lengths)}"
            )
        columns = zip(*(trace.request_rates for trace in traces.values()), strict=True)
        return cls(
            source_workload_ids=tuple(traces),
            control_interval_seconds=float(intervals.pop()),
            rates=tuple(float(statistics.median(column)) for column in columns),
        )

    @classmethod
    def from_benchmark(
        cls, workload_ids: Sequence[str], *, azure_csv_path: str | Path | None = None
    ) -> HistoricalDemandProfile:
        """Profile of benchmark workloads; only TRAIN workloads may be profile history."""
        manifest = load_benchmark_manifest()
        entries = []
        for workload_id in workload_ids:
            entry = manifest.get(workload_id)
            if entry.split != "train":
                kind = "held-out test" if entry.split == "test" else entry.split
                raise ValueError(
                    f"profile history uses TRAIN workloads only; {workload_id!r} is {kind}"
                )
            entries.append(entry)
        return cls.from_traces(build_workloads(entries, azure_csv_path=azure_csv_path))

    @property
    def ticks(self) -> int:
        return len(self.rates)

    def rate_at(self, tick: int) -> float | None:
        """Expected demand at aligned ``tick``; ``None`` outside the profile."""
        return self.rates[tick] if 0 <= tick < len(self.rates) else None

    def to_json(self) -> str:
        return json.dumps(
            {
                "version": self.version,
                "aggregation": self.aggregation,
                "source_workload_ids": list(self.source_workload_ids),
                "control_interval_seconds": self.control_interval_seconds,
                "rates": list(self.rates),
            },
            sort_keys=True,
        )

    @property
    def profile_id(self) -> str:
        return hashlib.sha256(self.to_json().encode()).hexdigest()[:12]

    @classmethod
    def from_json(cls, payload: str) -> HistoricalDemandProfile:
        data = json.loads(payload)
        return cls(
            source_workload_ids=tuple(data["source_workload_ids"]),
            control_interval_seconds=float(data["control_interval_seconds"]),
            rates=tuple(float(rate) for rate in data["rates"]),
            aggregation=data["aggregation"],
            version=data["version"],
        )


def profile_level_factor(
    samples: Sequence[tuple[int, float]],
    profile: HistoricalDemandProfile,
    *,
    window: int = HISTORY_WINDOW_TICKS,
) -> tuple[float, int]:
    """``(median(observed / profile), valid ratio count)`` over the latest ``window`` samples.

    ``samples`` are completed ``(tick, observed_rps)`` pairs. A sample counts only
    if the profile covers its tick with a value of at least
    ``MIN_PROFILE_RPS_FOR_RATIO``; with no valid ratio the factor is 1.0.
    """
    ratios = []
    for tick, observed in list(samples)[-window:]:
        expected = profile.rate_at(tick)
        if expected is not None and expected >= MIN_PROFILE_RPS_FOR_RATIO:
            ratios.append(observed / expected)
    if not ratios:
        return 1.0, 0
    return float(statistics.median(ratios)), len(ratios)


# --- decisions and forecast records --------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ProactiveForecastRecord:
    """A forecast as issued at decision time; no actual demand is stored here."""

    source_tick: int
    target_tick: int
    horizon_ticks: int
    horizon_seconds: float
    latest_observed_rps: float
    linear_forecast_rps: float
    historical_profile_forecast_rps: float | None
    profile_level_factor: float | None
    proactive_forecast_rps: float
    queued_requests: float
    backlog_recovery_rps: float
    effective_sizing_demand_rps: float
    desired_replicas: int
    requested_replica_target: int
    profile_used: bool

    @property
    def forecast_rps(self) -> float:
        """The forecast the capacity decision used (for :func:`score_forecasts`)."""
        return self.proactive_forecast_rps


@dataclass(frozen=True, slots=True)
class ProactivePredictiveDecision:
    """Why the controller chose its latest action (``action`` is the emitted code)."""

    reason: ProactiveReason
    active_replicas: int
    pending_replicas: int
    target_replicas: int
    action: int
    record: ProactiveForecastRecord | None = None
    observed_desired_replicas: int | None = None


# --- controller ---------------------------------------------------------------------------


class ProactivePredictiveController:
    """``predictive-seasonal-v1``: proactive scale-out, conservative scale-in (see module doc)."""

    variant = VARIANT
    forecast_method = FORECAST_METHOD
    capacity_policy = CAPACITY_POLICY

    def __init__(
        self,
        *,
        min_replicas: int,
        max_replicas: int,
        service_capacity_rps: float,
        startup_delay_seconds: float,
        control_interval_seconds: float,
        action_contract: ActionContract,
        profile: HistoricalDemandProfile | None = None,
    ) -> None:
        for name, count in (("min_replicas", min_replicas), ("max_replicas", max_replicas)):
            if isinstance(count, bool) or not isinstance(count, int):
                raise TypeError(f"{name} must be an integer")
        if not 1 <= min_replicas <= max_replicas:
            raise ValueError("replica bounds must satisfy 1 <= min_replicas <= max_replicas")
        if not (math.isfinite(service_capacity_rps) and service_capacity_rps > 0):
            raise ValueError("service_capacity_rps must be finite and positive")
        if (action_contract.min_replicas, action_contract.max_replicas) != (
            min_replicas,
            max_replicas,
        ):
            raise ValueError("the action contract's replica bounds differ from the controller's")
        if profile is not None and not math.isclose(
            profile.control_interval_seconds, control_interval_seconds, rel_tol=1e-9
        ):
            raise ValueError("the historical profile's control interval differs from the config")
        self._min = min_replicas
        self._max = max_replicas
        self._capacity_per_replica = float(service_capacity_rps) * TARGET_UTILIZATION
        self._interval = float(control_interval_seconds)
        self._horizon = 1 + startup_ticks(float(startup_delay_seconds), self._interval)
        self._contract = action_contract
        self._profile = profile
        self._samples: deque[tuple[int, float]] = deque(maxlen=HISTORY_WINDOW_TICKS)
        self._consumed_tick: Any = None
        self._records: list[ProactiveForecastRecord] = []
        self._last_decision: ProactivePredictiveDecision | None = None

    @classmethod
    def from_config(
        cls, config: SimulatorConfig, *, profile: HistoricalDemandProfile | None = None
    ) -> ProactivePredictiveController:
        """Bounds, capacity, timing, and the action contract all come from ``config``."""
        return cls(
            min_replicas=config.replicas.min_replicas,
            max_replicas=config.replicas.max_replicas,
            service_capacity_rps=config.replicas.service_capacity_rps,
            startup_delay_seconds=config.replicas.startup_delay_seconds,
            control_interval_seconds=config.timing.control_interval_seconds,
            action_contract=ActionContract.from_config(config),
            profile=profile,
        )

    @property
    def action_contract(self) -> ActionContract:
        return self._contract

    @property
    def profile(self) -> HistoricalDemandProfile | None:
        return self._profile

    @property
    def forecast_horizon_ticks(self) -> int:
        return self._horizon

    @property
    def forecasts(self) -> tuple[ProactiveForecastRecord, ...]:
        """Every forecast issued since reset, in order."""
        return tuple(self._records)

    @property
    def last_decision(self) -> ProactivePredictiveDecision | None:
        return self._last_decision

    def reset(self, seed: int | None = None) -> None:
        """Clear per-episode history and records; the profile is never modified."""
        self._samples.clear()
        self._consumed_tick = None
        self._records.clear()
        self._last_decision = None

    def act(self, observation: Observation, info: Mapping[str, Any]) -> int:
        self._record_sample(info)
        active, pending = int(info["active_replicas"]), int(info["pending_replicas"])
        committed = active + pending
        if not self._samples:
            code = self._contract.code_for_target(committed, committed)
            self._last_decision = ProactivePredictiveDecision(
                "no_sample", active, pending, self._contract.target_for(code, committed), code
            )
            return code

        source_tick, latest = self._samples[-1]
        target_tick = source_tick + self._horizon
        linear = linear_trend_forecast(tuple(self._samples), target_tick)
        profile_forecast: float | None = None
        factor: float | None = None
        if self._profile is not None:
            future = self._profile.rate_at(target_tick)
            if future is not None:
                factor, _ = profile_level_factor(tuple(self._samples), self._profile)
                profile_forecast = future * factor
        proactive = linear if profile_forecast is None else max(linear, profile_forecast)
        queued = _queued_requests(info)
        recovery = backlog_recovery_rate(queued, self._interval)
        effective = proactive + recovery
        _, desired = self._size(effective)

        observed_desired: int | None = None
        reason: ProactiveReason
        if desired > committed:
            target, reason = desired, "scale_out"
        elif desired == committed:
            target, reason = committed, "at_target"
        elif queued > 0:
            target, reason = committed, "backlog_hold"  # never remove capacity into a backlog
        else:
            _, observed_desired = self._size(latest)
            safe = max(observed_desired, desired)
            if safe < committed:
                target, reason = max(safe, committed - 1), "scale_in"
            else:
                target, reason = committed, "hold_for_observed_or_forecast"

        code = self._contract.code_for_target(target, committed)
        requested = self._contract.target_for(code, committed)
        record = ProactiveForecastRecord(
            source_tick=source_tick,
            target_tick=target_tick,
            horizon_ticks=self._horizon,
            horizon_seconds=self._horizon * self._interval,
            latest_observed_rps=latest,
            linear_forecast_rps=linear,
            historical_profile_forecast_rps=profile_forecast,
            profile_level_factor=factor,
            proactive_forecast_rps=proactive,
            queued_requests=queued,
            backlog_recovery_rps=recovery,
            effective_sizing_demand_rps=effective,
            desired_replicas=desired,
            requested_replica_target=requested,
            profile_used=profile_forecast is not None,
        )
        self._records.append(record)
        self._last_decision = ProactivePredictiveDecision(
            reason, active, pending, requested, code, record, observed_desired
        )
        return code

    def _size(self, demand_rps: float) -> tuple[int, int]:
        return size_replicas(
            demand_rps,
            capacity_per_replica_rps=self._capacity_per_replica,
            min_replicas=self._min,
            max_replicas=self._max,
        )

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
        rate = float(info["request_rate"])
        if not (math.isfinite(rate) and rate >= 0):
            raise ValueError("request_rate must be finite and non-negative")
        self._samples.append((index, rate))


def _queued_requests(info: Mapping[str, Any]) -> float:
    if "queued_requests" not in info:
        raise ValueError("info must include queued_requests once a tick has completed")
    queued = float(info["queued_requests"])
    if not (math.isfinite(queued) and queued >= 0):
        raise ValueError("queued_requests must be finite and non-negative")
    return queued
