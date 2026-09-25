"""Controller-agnostic episode metrics shared by tuning and evaluation."""

from __future__ import annotations

import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from scalerl.controllers import Controller, run_episode
from scalerl.environment import AutoscalingEnv, SimulatorConfig
from scalerl.environment.reward import max_tick_capacity_of, max_tick_cost_of


@dataclass(frozen=True)
class EpisodeMetrics:
    """System metrics of one controller episode on one workload."""

    infrastructure_cost: float
    normalized_cost: float
    sla_violation_rate: float
    mean_p95_latency_seconds: float
    max_p95_latency_seconds: float
    mean_queue_depth: float
    max_queue_depth: float
    queue_pressure: float
    scaling_actions: int
    churn_rate: float
    episode_reward: float

    def as_metrics(self) -> dict[str, float]:
        return {name: float(value) for name, value in vars(self).items()}


@dataclass(frozen=True)
class EpisodeEvaluation:
    """An episode's step ``info`` dicts and their summary metrics."""

    infos: tuple[Mapping[str, Any], ...]
    metrics: EpisodeMetrics


def summarize_episode(
    infos: Sequence[Mapping[str, Any]], config: SimulatorConfig
) -> EpisodeMetrics:
    """Summarize completed step ``info`` dicts.

    Normalized cost divides by the cost of ``max_replicas`` for the whole
    episode; queue pressure averages ``queued / (queued + max-fleet tick
    capacity)``, as in the observation. Churn counts ticks whose applied
    replica change was non-zero, so bound-clipped requests are not churn.
    Reward is included as a secondary metric.
    """
    if not infos:
        raise ValueError("cannot summarize an empty episode")
    ticks = len(infos)
    max_capacity = max_tick_capacity_of(config)
    max_cost = max_tick_cost_of(config) * ticks
    cost = sum(info["infrastructure_cost"] for info in infos)
    latencies = [info["p95_latency_seconds"] for info in infos]
    queues = [info["queued_requests"] for info in infos]
    changes = sum(1 for info in infos if info["applied_replica_change"] != 0)
    return EpisodeMetrics(
        infrastructure_cost=cost,
        normalized_cost=cost / max_cost if max_cost > 0 else 0.0,
        sla_violation_rate=sum(info["sla_violated"] for info in infos) / ticks,
        mean_p95_latency_seconds=statistics.fmean(latencies),
        max_p95_latency_seconds=max(latencies),
        mean_queue_depth=statistics.fmean(queues),
        max_queue_depth=max(queues),
        queue_pressure=statistics.fmean(q / (q + max_capacity) for q in queues),
        scaling_actions=changes,
        churn_rate=changes / ticks,
        episode_reward=sum(info["reward"] for info in infos),
    )


def evaluate_controller_episode(
    env: AutoscalingEnv, controller: Controller, *, seed: int | None = 0
) -> EpisodeEvaluation:
    """Run one full episode through the shared runner and summarize it."""
    infos = run_episode(env, controller, seed=seed)
    return EpisodeEvaluation(tuple(infos), summarize_episode(infos, env.config))
