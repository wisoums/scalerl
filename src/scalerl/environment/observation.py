"""Shared learned-policy observation builder, ``scalerl-observation-v1`` (#72).

One pure function, :func:`build_observation`, turns controller-visible values
into the policy observation. ``AutoscalingEnv`` calls it, and a future live
(Knative) adapter must call the **same** function with live-measured inputs, so
feature order, units and normalization cannot drift between simulation and
the real system. The semantics are those learned policies were trained on;
nothing here changes them.

Feature order (``h = traffic_history_ticks``, v1: 4; ``k`` = nominal
pending-readiness buckets, v1: 2 at 60 s startup / 30 s ticks; 12 in total):

======  ==============================  ========================================================
index   feature                         value (all in [0, 1], no clipping needed)
======  ==============================  ========================================================
0..h-1  ``demand_pressure_t-{age}``     ``rate / (rate + max_replicas * service_capacity_rps)``,
                                        request rate (RPS) of the latest h completed ticks,
                                        newest first; 0 where no completed tick exists yet
h       ``utilization``                 latest completed tick's utilization of active capacity
h+1     ``queue_pressure``              ``queued / (queued + max_replicas * capacity * interval)``
h+2     ``latency_pressure``            ``p95 / (p95 + latency_target_seconds)``
h+3     ``active_replicas_fraction``    current active replicas / ``max_replicas``
h+4     ``tick_cost_fraction``          **latest completed tick's** cost / cost of max_replicas for
                                        one tick (never accumulated episode cost)
h+5     ``episode_progress``            completed ticks / episode ticks
h+6..   ``pending_ready_in_{i}``        pending replicas nominally ready after i more ticks /
                                        ``max_replicas`` (nominal readiness, #81)
======  ==============================  ========================================================

Measurements (rates, utilization, queue, latency, cost) come from the latest
completed tick visible under the telemetry delay; replica counts, pending
buckets and progress are current. Before any measurement exists every
measurement feature is 0.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray

from scalerl.environment.config import SimulatorConfig
from scalerl.environment.replicas import startup_ticks
from scalerl.environment.reward import max_tick_capacity_of, max_tick_cost_of

OBSERVATION_VERSION: Final = "scalerl-observation-v1"


@dataclass(frozen=True, slots=True)
class ObservationConstants:
    """The normalization constants of one simulator configuration."""

    history_ticks: int
    max_replicas: int
    max_service_rate_rps: float  # max_replicas * service_capacity_rps
    max_tick_capacity_requests: float  # max_replicas * service_capacity_rps * interval
    max_tick_cost: float  # cost of max_replicas for one control interval
    latency_target_seconds: float
    pending_buckets: int

    @classmethod
    def from_config(cls, config: SimulatorConfig) -> ObservationConstants:
        replicas = config.replicas
        return cls(
            history_ticks=config.observation.traffic_history_ticks,
            max_replicas=replicas.max_replicas,
            max_service_rate_rps=replicas.max_replicas * replicas.service_capacity_rps,
            max_tick_capacity_requests=max_tick_capacity_of(config),
            max_tick_cost=max_tick_cost_of(config),
            latency_target_seconds=config.sla.latency_target_seconds,
            pending_buckets=startup_ticks(
                replicas.startup_delay_seconds, config.timing.control_interval_seconds
            ),
        )


@dataclass(frozen=True, slots=True)
class Measurement:
    """Monitoring values of one completed tick (the latest visible one)."""

    utilization: float
    queued_requests: float
    p95_latency_seconds: float
    infrastructure_cost: float  # this tick's cost, not accumulated


def feature_names(constants: ObservationConstants) -> tuple[str, ...]:
    return (
        *(f"demand_pressure_t-{age}" for age in range(constants.history_ticks)),
        "utilization",
        "queue_pressure",
        "latency_pressure",
        "active_replicas_fraction",
        "tick_cost_fraction",
        "episode_progress",
        *(f"pending_ready_in_{ticks}" for ticks in range(1, constants.pending_buckets + 1)),
    )


def build_observation(
    constants: ObservationConstants,
    *,
    request_rates_newest_first: Sequence[float],
    latest: Measurement | None,
    active_replicas: int,
    pending_by_ticks: Sequence[int],
    completed_ticks: int,
    episode_ticks: int,
) -> NDArray[np.float32]:
    """The ``scalerl-observation-v1`` vector (see the module docstring)."""
    if len(request_rates_newest_first) > constants.history_ticks:
        raise ValueError("more demand samples than the traffic history holds")
    if len(pending_by_ticks) != constants.pending_buckets:
        raise ValueError("pending readiness buckets do not match the configuration")
    history = [
        rate / (rate + constants.max_service_rate_rps) for rate in request_rates_newest_first
    ]
    history += [0.0] * (constants.history_ticks - len(history))
    max_replicas = constants.max_replicas

    if latest is None:
        queued = utilization = latency_pressure = cost_fraction = 0.0
    else:
        queued = latest.queued_requests
        latency = latest.p95_latency_seconds
        utilization = latest.utilization
        latency_pressure = latency / (latency + constants.latency_target_seconds)
        cost_fraction = (
            latest.infrastructure_cost / constants.max_tick_cost
            if constants.max_tick_cost > 0
            else 0.0
        )

    return np.array(
        [
            *history,
            utilization,
            queued / (queued + constants.max_tick_capacity_requests),
            latency_pressure,
            active_replicas / max_replicas,
            cost_fraction,
            completed_ticks / episode_ticks,
            *(count / max_replicas for count in pending_by_ticks),
        ],
        dtype=np.float32,
    )
