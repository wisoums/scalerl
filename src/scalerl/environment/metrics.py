"""Per-tick latency, SLA, and infrastructure-cost metrics.

Latency is a deterministic **p95 proxy**, not a simulated request-latency
distribution. With ``mu`` the per-replica ``service_capacity_rps``::

    service_time      = 1 / mu
    capacity          = active_replicas * mu * control_interval_seconds
    offered_work      = processed_requests + queued_requests
    utilization       = min(offered_work / capacity, 1)
    saturation_factor = 1 / max(1 - utilization, 0.05)
    queue_delay       = queued_requests / (active_replicas * mu)
    p95_latency       = service_time * saturation_factor + queue_delay

With zero active replicas nothing can be served. Idle ticks (no offered work)
use ``utilization = 0`` and no queue delay. Otherwise ``utilization = 1`` and
queued work is assumed to wait out the whole tick and then drain at one
replica's rate: ``queue_delay = control_interval_seconds + queued_requests / mu``.
Such ticks always violate the SLA.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from scalerl.environment.config import SimulatorConfig
from scalerl.environment.queue import QueueStepResult

# Caps the saturation factor at 20x service time so latency stays finite.
_MIN_SATURATION_HEADROOM = 0.05
_SECONDS_PER_HOUR = 3600.0


@dataclass(frozen=True, slots=True)
class TickMetrics:
    """Latency, SLA, and cost for one tick."""

    utilization: float
    p95_latency_seconds: float
    sla_violated: bool
    infrastructure_cost: float


def compute_tick_metrics(
    queue_result: QueueStepResult,
    *,
    active_replicas: int,
    pending_replicas: int,
    config: SimulatorConfig,
) -> TickMetrics:
    """Derive tick metrics from queue outcome and replica counts without side effects.

    Active and pending replicas are billable; terminating replicas are not.
    """
    _check_replica_count("active_replicas", active_replicas)
    _check_replica_count("pending_replicas", pending_replicas)

    service_capacity_rps = config.replicas.service_capacity_rps
    interval = config.timing.control_interval_seconds
    queued = queue_result.queued_requests
    offered_work = queue_result.processed_requests + queued

    if active_replicas > 0:
        service_rate = active_replicas * service_capacity_rps
        utilization = min(offered_work / (service_rate * interval), 1.0)
        queue_delay = queued / service_rate
    elif offered_work > 0:
        utilization = 1.0
        queue_delay = interval + queued / service_capacity_rps
    else:
        utilization = 0.0
        queue_delay = 0.0

    saturation_factor = 1.0 / max(1.0 - utilization, _MIN_SATURATION_HEADROOM)
    p95_latency = saturation_factor / service_capacity_rps + queue_delay
    sla_violated = p95_latency > config.sla.latency_target_seconds or (
        active_replicas == 0 and offered_work > 0
    )

    billable = active_replicas + pending_replicas
    # Scale the price to one tick first so large hourly prices don't overflow early.
    cost_per_replica_tick = config.replicas.cost_per_hour * (interval / _SECONDS_PER_HOUR)
    cost = billable * cost_per_replica_tick

    if not (math.isfinite(p95_latency) and math.isfinite(cost)):
        raise ValueError("tick metrics overflow; inputs are too large")

    return TickMetrics(
        utilization=utilization,
        p95_latency_seconds=p95_latency,
        sla_violated=sla_violated,
        infrastructure_cost=cost,
    )


def _check_replica_count(name: str, count: int) -> None:
    if isinstance(count, bool) or not isinstance(count, int):
        raise TypeError(f"{name} must be an integer")
    if count < 0:
        raise ValueError(f"{name} must be non-negative")
