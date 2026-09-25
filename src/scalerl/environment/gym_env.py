"""Gymnasium environment composing the ScaleRL cloud simulator.

Actions (``Discrete(3)``): ``0`` scale down by one, ``1`` hold, ``2`` scale
up by one. Scaling past ``min_replicas``/``max_replicas`` is a no-op, reported
as ``applied_replica_change == 0`` in ``info``.

Each ``step`` runs one tick in this order:

1. apply the action to the replica pool;
2. read this tick's demand from the workload replay;
3. serve requests with the replicas active after the action;
4. compute latency, SLA, and cost metrics;
5. compute the reward;
6. advance replica lifecycle by one control interval;
7. advance the simulation clock;
8. build the next observation and ``info``.

New replicas therefore serve no traffic until their startup delay has elapsed,
unless the delay is zero.

Observations (``Box([0, 1]^(h + 6 + k))``) describe the state after the last
completed tick; at reset there is no completed tick, so traffic features are zero.
With ``h = observation.traffic_history_ticks`` (v1: 4):

0 ... h - 1. demand pressure of the latest ``h`` completed ticks, newest first
   (``0`` = latest, ``h - 1`` = oldest; zero until enough ticks have run), each
   ``rate / (rate + max_replicas * service_capacity_rps)``
h. utilization of active capacity
h + 1. queue pressure: ``queued / (queued + max-fleet capacity per tick)``
h + 2. latency pressure: ``p95 / (p95 + latency_target)``
h + 3. active replicas / ``max_replicas``
h + 4. tick cost / cost of ``max_replicas`` for one tick
h + 5. episode progress: completed ticks / episode ticks
h + 6 ... h + 6 + k - 1. pending replicas that become active after 1 ... k more
   ticks, each / ``max_replicas``

``observation_features`` names every position. The traffic history holds only
demand consumed by completed ticks, never the next workload value.
``h = 1`` reproduces the earlier single-demand layout (``7 + k``).

``k = ceil(startup_delay_seconds / control_interval_seconds)`` (0 with no
delay). Bucketing pending replicas by readiness keeps the observation Markov:
equal pending counts with different remaining startup times look different.
The shape is fixed for a given config, but changing startup delay or control
interval can change it, so a trained policy only fits environments whose
observation and action spaces match its training environment.

Episodes never terminate; they are truncated once the workload trace, which
must span exactly one episode, is exhausted.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Any, SupportsFloat

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray

from scalerl.environment.clock import SimulationClock
from scalerl.environment.config import SimulatorConfig
from scalerl.environment.metrics import TickMetrics, compute_tick_metrics
from scalerl.environment.queue import QueueStepResult, RequestQueue
from scalerl.environment.replicas import ReplicaPool
from scalerl.environment.reward import (
    RewardWeights,
    compute_reward,
    max_tick_capacity_of,
    max_tick_cost_of,
)
from scalerl.workloads.trace import WorkloadReplay, WorkloadTrace

SCALE_DOWN, HOLD, SCALE_UP = 0, 1, 2

Observation = NDArray[np.float32]


class AutoscalingEnv(gym.Env[Observation, np.int64]):
    """Autoscaling MDP over a fixed workload trace."""

    metadata: dict[str, Any] = {"render_modes": []}

    def __init__(
        self,
        config: SimulatorConfig,
        trace: WorkloadTrace,
        reward_weights: RewardWeights | None = None,
    ) -> None:
        self._episode_ticks = _episode_ticks(config, trace)
        self.config = config
        self.trace = trace
        self.reward_weights = reward_weights or RewardWeights()

        self._clock = SimulationClock(config.timing.control_interval_seconds)
        self._replay = WorkloadReplay(trace)
        self._pool = ReplicaPool(config.replicas)

        pending_buckets = len(self._pending_buckets())
        self.action_space = spaces.Discrete(3)
        self._history_ticks = config.observation.traffic_history_ticks
        self._demand_history: deque[float] = deque(maxlen=self._history_ticks)
        self._observation_features = (
            *(f"demand_pressure_t-{age}" for age in range(self._history_ticks)),
            "utilization",
            "queue_pressure",
            "latency_pressure",
            "active_replicas_fraction",
            "tick_cost_fraction",
            "episode_progress",
            *(f"pending_ready_in_{ticks}" for ticks in range(1, pending_buckets + 1)),
        )
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(len(self._observation_features),), dtype=np.float32
        )
        self._queue = RequestQueue(config.replicas, config.timing)
        self._max_service_rate = config.replicas.max_replicas * config.replicas.service_capacity_rps
        self._max_tick_capacity = max_tick_capacity_of(config)
        self._max_tick_cost = max_tick_cost_of(config)
        self._last_metrics: TickMetrics | None = None
        self._needs_reset = True

    @property
    def episode_ticks(self) -> int:
        """Return the number of steps in one episode."""
        return self._episode_ticks

    @property
    def observation_features(self) -> tuple[str, ...]:
        """Name of every observation position, in order."""
        return self._observation_features

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Observation, dict[str, Any]]:
        super().reset(seed=seed)
        self._clock.reset()
        self._replay.reset()
        self._pool.reset()
        self._queue.reset()
        self._demand_history.clear()
        self._last_metrics = None
        self._needs_reset = False
        return self._observation(), self.replica_counts | {"tick": 0, "time_seconds": 0.0}

    def step(
        self, action: np.int64 | int
    ) -> tuple[Observation, SupportsFloat, bool, bool, dict[str, Any]]:
        if self._needs_reset:
            raise RuntimeError("episode has ended or not started; call reset() before step()")
        if isinstance(action, bool) or not self.action_space.contains(action):
            raise ValueError(f"invalid action {action!r}; expected 0, 1, or 2")
        requested = int(action)

        # 1. apply the action
        if requested == SCALE_UP:
            applied = self._pool.scale_up()
        elif requested == SCALE_DOWN:
            applied = -self._pool.scale_down()
        else:
            applied = 0
        tick_replicas = self.replica_counts

        # 2-4. serve this tick's demand with the currently active replicas
        tick = self._clock.step_count
        request_rate = self._replay.next_demand()
        queue_result = self._queue.step(request_rate, self._pool.active_count)
        metrics = compute_tick_metrics(
            queue_result,
            active_replicas=self._pool.active_count,
            pending_replicas=self._pool.pending_count,
            config=self.config,
        )

        # 5. reward
        breakdown = compute_reward(
            metrics,
            queue_result,
            applied_replica_change=applied,
            config=self.config,
            weights=self.reward_weights,
        )

        # 6-7. advance lifecycle and time
        self._pool.advance(self.config.timing.control_interval_seconds)
        self._clock.step()

        # 8. next observation and info: history gains only the demand this tick consumed
        self._demand_history.appendleft(self._demand_pressure(request_rate))
        self._last_metrics = metrics
        truncated = self._replay.is_exhausted
        self._needs_reset = truncated

        info: dict[str, Any] = {
            "tick": tick,
            "time_seconds": self._clock.time_seconds,
            "request_rate": request_rate,
            "requested_action": requested,
            "applied_replica_change": applied,
            **tick_replicas,
            **_queue_info(queue_result),
            "utilization": metrics.utilization,
            "p95_latency_seconds": metrics.p95_latency_seconds,
            "sla_violated": metrics.sla_violated,
            "infrastructure_cost": metrics.infrastructure_cost,
            "reward_components": {
                "latency_penalty": breakdown.latency_penalty,
                "cost_penalty": breakdown.cost_penalty,
                "sla_penalty": breakdown.sla_penalty,
                "queue_penalty": breakdown.queue_penalty,
                "churn_penalty": breakdown.churn_penalty,
            },
            "reward": breakdown.reward,
        }
        return self._observation(), breakdown.reward, False, truncated, info

    def _demand_pressure(self, rate: float) -> float:
        return rate / (rate + self._max_service_rate)

    def _observation(self) -> Observation:
        history = [*self._demand_history]
        history += [0.0] * (self._history_ticks - len(history))
        queued = self._queue.queue_depth
        max_replicas = self.config.replicas.max_replicas
        metrics = self._last_metrics

        if metrics is None:
            utilization = latency_pressure = cost_fraction = 0.0
        else:
            latency = metrics.p95_latency_seconds
            target = self.config.sla.latency_target_seconds
            utilization = metrics.utilization
            latency_pressure = latency / (latency + target)
            cost_fraction = (
                metrics.infrastructure_cost / self._max_tick_cost
                if self._max_tick_cost > 0
                else 0.0
            )

        return np.array(
            [
                *history,
                utilization,
                queued / (queued + self._max_tick_capacity),
                latency_pressure,
                self._pool.active_count / max_replicas,
                cost_fraction,
                self._clock.step_count / self._episode_ticks,
                *(count / max_replicas for count in self._pending_buckets()),
            ],
            dtype=np.float32,
        )

    def _pending_buckets(self) -> tuple[int, ...]:
        return self._pool.pending_by_ticks_until_active(self.config.timing.control_interval_seconds)

    @property
    def replica_counts(self) -> dict[str, int]:
        """Return current replica counts, i.e. the state the next action applies to.

        ``info`` from ``step()`` reports the replicas that served that tick,
        before pending replicas could activate at its end.
        """
        return {
            "active_replicas": self._pool.active_count,
            "pending_replicas": self._pool.pending_count,
            "terminating_replicas": self._pool.terminating_count,
        }


def _queue_info(result: QueueStepResult) -> dict[str, float]:
    return {
        "arrived_requests": result.arrived_requests,
        "processed_requests": result.processed_requests,
        "queued_requests": result.queued_requests,
        "dropped_requests": result.dropped_requests,
    }


def _episode_ticks(config: SimulatorConfig, trace: WorkloadTrace) -> int:
    """Return the episode length in ticks, requiring the trace to match it exactly."""
    interval = config.timing.control_interval_seconds
    if not math.isclose(trace.control_interval_seconds, interval, rel_tol=1e-9):
        raise ValueError(
            f"trace control interval {trace.control_interval_seconds} s does not match "
            f"configured control interval {interval} s"
        )

    ratio = config.timing.episode_duration_seconds / interval
    ticks = round(ratio)
    if not math.isclose(ratio, ticks, rel_tol=1e-9):
        raise ValueError("episode_duration_seconds must be a whole number of control intervals")
    if len(trace) != ticks:
        raise ValueError(
            f"trace has {len(trace)} ticks but the configured episode needs {ticks}; "
            "traces are not truncated or resampled"
        )
    return ticks
