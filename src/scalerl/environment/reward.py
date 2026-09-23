"""Provisional, configurable autoscaling reward.

``reward = -(latency + cost + sla + queue + churn)`` penalties, where each
penalty is a weight times a normalized term in ``[0, 1]``:

- latency: ``p95 / (p95 + latency_target)`` (0.5 exactly at the SLA target);
- cost:    tick cost / cost of ``max_replicas`` for one tick;
- sla:     1 if the SLA was violated, else 0;
- queue:   ``queued / (queued + max-fleet capacity per tick)``;
- churn:   1 if a scaling action actually changed desired replicas, else 0.

Default weights are provisional placeholders for reward ablations, not tuned
values.
"""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, Field

from scalerl.environment.config import SimulatorConfig
from scalerl.environment.metrics import TickMetrics
from scalerl.environment.queue import QueueStepResult


class RewardWeights(BaseModel):
    """Non-negative weights applied to each normalized penalty."""

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid", allow_inf_nan=False)

    latency: float = Field(default=1.0, ge=0, description="Weight of the latency penalty.")
    cost: float = Field(default=1.0, ge=0, description="Weight of the infrastructure-cost penalty.")
    sla: float = Field(default=1.0, ge=0, description="Weight of the SLA-violation penalty.")
    queue: float = Field(default=1.0, ge=0, description="Weight of the queue-backlog penalty.")
    churn: float = Field(default=0.1, ge=0, description="Weight of the scaling-churn penalty.")


@dataclass(frozen=True, slots=True)
class RewardBreakdown:
    """Weighted penalties for one tick and the resulting reward."""

    latency_penalty: float
    cost_penalty: float
    sla_penalty: float
    queue_penalty: float
    churn_penalty: float
    reward: float


def compute_reward(
    metrics: TickMetrics,
    queue_result: QueueStepResult,
    *,
    applied_replica_change: int,
    config: SimulatorConfig,
    weights: RewardWeights,
) -> RewardBreakdown:
    """Combine one tick's outcome into a decomposed, finite reward."""
    latency = metrics.p95_latency_seconds
    queued = queue_result.queued_requests
    max_tick_cost = max_tick_cost_of(config)

    latency_penalty = weights.latency * latency / (latency + config.sla.latency_target_seconds)
    cost_penalty = weights.cost * (
        metrics.infrastructure_cost / max_tick_cost if max_tick_cost > 0 else 0.0
    )
    sla_penalty = weights.sla * float(metrics.sla_violated)
    queue_penalty = weights.queue * queued / (queued + max_tick_capacity_of(config))
    churn_penalty = weights.churn * float(applied_replica_change != 0)

    return RewardBreakdown(
        latency_penalty=latency_penalty,
        cost_penalty=cost_penalty,
        sla_penalty=sla_penalty,
        queue_penalty=queue_penalty,
        churn_penalty=churn_penalty,
        reward=-(latency_penalty + cost_penalty + sla_penalty + queue_penalty + churn_penalty),
    )


def max_tick_capacity_of(config: SimulatorConfig) -> float:
    """Return the requests ``max_replicas`` can serve in one tick."""
    replicas = config.replicas
    return (
        replicas.max_replicas
        * replicas.service_capacity_rps
        * config.timing.control_interval_seconds
    )


def max_tick_cost_of(config: SimulatorConfig) -> float:
    """Return the cost of billing ``max_replicas`` for one tick."""
    tick_hours = config.timing.control_interval_seconds / 3600.0
    return config.replicas.max_replicas * config.replicas.cost_per_hour * tick_hours
