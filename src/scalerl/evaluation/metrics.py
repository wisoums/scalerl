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


@dataclass(frozen=True)
class ActionMagnitudeMetrics:
    """How much capacity an episode's scaling decisions moved (#79 diagnostics).

    ``scaling_events`` is the existing ``scaling_actions`` (ticks with a
    non-zero applied change), which is also what the reward's churn term
    counts: one event per scaling tick, whatever its size. Under
    ``desired-replicas-v1`` one event can move several replicas, so the
    magnitude is reported separately and fewer scaling ticks do not imply less
    total scaling. These are diagnostics only; they do not enter the reward
    or any selection rule.

    * ``total_absolute_replica_change``: sum of ``|applied_replica_change|``;
    * ``mean_absolute_replica_change_when_scaling``: that sum per scaling event
      (0.0 when the episode never scaled);
    * ``max_absolute_replica_change_in_one_tick``;
    * ``max_pending_replicas``: most replicas starting up at once (after a
      tick's action).
    """

    scaling_events: int
    total_absolute_replica_change: int
    mean_absolute_replica_change_when_scaling: float
    max_absolute_replica_change_in_one_tick: int
    max_pending_replicas: int

    def as_metrics(self) -> dict[str, float]:
        return {f"action.{name}": float(value) for name, value in vars(self).items()}


def summarize_action_magnitude(infos: Sequence[Mapping[str, Any]]) -> ActionMagnitudeMetrics:
    """Summarize the applied replica changes of completed step ``info`` dicts."""
    if not infos:
        raise ValueError("cannot summarize an empty episode")
    changes = [abs(int(info["applied_replica_change"])) for info in infos]
    events = sum(1 for change in changes if change)
    total = sum(changes)
    return ActionMagnitudeMetrics(
        scaling_events=events,
        total_absolute_replica_change=total,
        mean_absolute_replica_change_when_scaling=total / events if events else 0.0,
        max_absolute_replica_change_in_one_tick=max(changes),
        max_pending_replicas=max(int(info["pending_replicas"]) for info in infos),
    )


def evaluate_controller_episode(
    env: AutoscalingEnv, controller: Controller, *, seed: int | None = 0
) -> EpisodeEvaluation:
    """Run one full episode through the shared runner and summarize it."""
    infos = run_episode(env, controller, seed=seed)
    return EpisodeEvaluation(tuple(infos), summarize_episode(infos, env.config))
