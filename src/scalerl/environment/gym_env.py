"""Gymnasium environment composing the ScaleRL cloud simulator.

Actions are integer **codes** from a ``Discrete`` space whose meaning is the
versioned action contract ``config.action.semantics`` (#79, see
:mod:`scalerl.environment.actions`):

* ``delta-v1`` (default): ``Discrete(3)``, ``0`` scale down by one, ``1`` hold,
  ``2`` scale up by one (effects -1/0/+1; ``-1`` is not a valid code);
* ``desired-replicas-v1``: ``Discrete(max_replicas - min_replicas + 1)``, code
  ``c`` requests ``min_replicas + c`` committed replicas.

The code becomes a requested target for committed capacity (``active +
pending``), and ``|target - committed|`` replicas are started or cancelled in
that one step through the normal lifecycle. Scaling past
``min_replicas``/``max_replicas`` is clipped; ``info`` reports the code
(``requested_action``), the bounded target (``requested_replica_target``) and
the signed change actually applied (``applied_replica_change``, which can
exceed one replica under ``desired-replicas-v1``).

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
   ticks, each / ``max_replicas``, by their **nominal** readiness (age against
   the configured startup delay). Under a stochastic startup-delay model (#81)
   the sampled realization is hidden: a replica past its nominal readiness but
   still physically pending counts in the first bucket, and the shape is
   unchanged, so already-trained policies can be evaluated under it.

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

Robustness dynamics (#65, ``config.dynamics``; nominal defaults change nothing):

* **Capacity jitter.** Each tick draws a multiplier ``m ~ Uniform(1 - f, 1 + f)``
  (``f = capacity_jitter_fraction``) from a per-environment RNG seeded by
  ``dynamics_seed`` and re-created on every ``reset``, so the sequence depends
  only on the dynamics realization and the tick, never on actions or on other
  environments. The queue and the tick metrics both use the realized
  ``service_capacity_rps * m``. With ``f = 0``, ``m`` is exactly 1.0 and no RNG
  is drawn. Normalizers (demand/queue pressure, max-fleet capacity) stay
  nominal so metrics remain comparable across scenarios.
* **Physical truth vs controller telemetry.** ``step`` ``info`` always describes
  what physically happened on that tick (plus the realized capacity). What a
  controller sees is a separate view: the observation and :meth:`decision_info`
  take load/queue/latency/cost measurements from the completed tick
  ``telemetry_delay_ticks`` ago, from a bounded buffer of completed-tick
  snapshots, while replica counts, pending readiness, the clock, and the
  latest requested/applied action stay current. Before a delayed measurement
  exists, observation telemetry is zero (as at reset) and ``decision_info``
  contains no measurement fields (rule-based controllers then see "no sample",
  never a fabricated value).
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, SupportsFloat

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray

from scalerl.environment.actions import HOLD, SCALE_DOWN, SCALE_UP, ActionContract
from scalerl.environment.clock import SimulationClock
from scalerl.environment.config import SimulatorConfig
from scalerl.environment.metrics import compute_tick_metrics
from scalerl.environment.observation import (
    Measurement,
    ObservationConstants,
    build_observation,
    feature_names,
)
from scalerl.environment.queue import QueueStepResult, RequestQueue
from scalerl.environment.replicas import ReplicaPool
from scalerl.environment.reward import (
    RewardWeights,
    compute_reward,
)
from scalerl.environment.startup import FIXED_V1, draw_tri_point_multiplier, startup_rng
from scalerl.workloads.trace import WorkloadReplay, WorkloadTrace

__all__ = [
    "CONTROL_PLANE_KEYS",
    "HOLD",
    "SCALE_DOWN",
    "SCALE_UP",
    "AutoscalingEnv",
    "Observation",
    "PHYSICAL_ONLY_KEYS",
    "TelemetrySnapshot",
]

# Control-plane facts a controller always knows currently; everything else in a
# step ``info`` is a monitoring measurement subject to telemetry delay.
CONTROL_PLANE_KEYS = frozenset(
    {
        "tick",
        "time_seconds",
        "requested_action",
        "requested_replica_target",
        "applied_replica_change",
        "active_replicas",
        "pending_replicas",
        "terminating_replicas",
    }
)

# Physical-truth diagnostics that describe hidden future lifecycle state (sampled
# startup realizations, #81): kept in step ``info`` for evaluation, but never in
# telemetry snapshots, ``decision_info``, or the observation.
PHYSICAL_ONLY_KEYS = frozenset(
    {
        "startup_delay_model",
        "startup_realized_delays_seconds",
        "startup_multipliers",
        "physical_pending_by_ticks",
    }
)

Observation = NDArray[np.float32]


@dataclass(frozen=True, slots=True)
class TelemetrySnapshot:
    """Measurements of one completed tick, as monitoring would report them."""

    tick: int
    measurements: Mapping[str, Any]  # the tick's non-control-plane ``info`` fields

    @property
    def request_rate(self) -> float:
        return float(self.measurements["request_rate"])

    @property
    def queued_requests(self) -> float:
        return float(self.measurements["queued_requests"])

    @property
    def utilization(self) -> float:
        return float(self.measurements["utilization"])

    @property
    def p95_latency_seconds(self) -> float:
        return float(self.measurements["p95_latency_seconds"])

    @property
    def infrastructure_cost(self) -> float:
        return float(self.measurements["infrastructure_cost"])


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
        self._startup_model = config.dynamics.startup_delay_model
        self._startup_rng = startup_rng(config.dynamics.startup_delay_seed)
        self._pool = ReplicaPool(
            config.replicas,
            startup_multiplier=(
                None if self._startup_model == FIXED_V1 else self._draw_startup_multiplier
            ),
        )

        pending_buckets = len(self._pending_buckets())
        self._actions = ActionContract.from_config(config)
        self.action_space = spaces.Discrete(self._actions.action_count)
        self._history_ticks = config.observation.traffic_history_ticks
        self._delay_ticks = config.dynamics.telemetry_delay_ticks
        self._jitter = config.dynamics.capacity_jitter_fraction
        # Newest first; just enough for the delayed window the observation reads.
        self._telemetry: deque[TelemetrySnapshot] = deque(
            maxlen=self._delay_ticks + self._history_ticks
        )
        self._dynamics_rng = np.random.default_rng(config.dynamics.dynamics_seed)
        self._observation_constants = ObservationConstants.from_config(config)
        if self._observation_constants.pending_buckets != pending_buckets:
            raise AssertionError("observation constants disagree with the replica lifecycle")
        self._observation_features = feature_names(self._observation_constants)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(len(self._observation_features),), dtype=np.float32
        )
        self._queue = RequestQueue(config.replicas, config.timing)
        self._needs_reset = True

    @property
    def episode_ticks(self) -> int:
        """Return the number of steps in one episode."""
        return self._episode_ticks

    @property
    def action_contract(self) -> ActionContract:
        """The versioned action contract (codes -> requested replica targets)."""
        return self._actions

    @property
    def action_semantics(self) -> str:
        return self._actions.semantics

    @property
    def telemetry_delay_ticks(self) -> int:
        return self._delay_ticks

    @property
    def telemetry_buffer_size(self) -> int:
        """Maximum completed-tick snapshots kept: delay + traffic history."""
        return self._delay_ticks + self._history_ticks

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
        self._telemetry.clear()
        # Restart the dynamics realization: same seed, same multiplier sequence.
        self._dynamics_rng = np.random.default_rng(self.config.dynamics.dynamics_seed)
        self._startup_rng = startup_rng(self.config.dynamics.startup_delay_seed)
        self._needs_reset = False
        return self._observation(), self.replica_counts | {"tick": 0, "time_seconds": 0.0}

    def step(
        self, action: np.int64 | int
    ) -> tuple[Observation, SupportsFloat, bool, bool, dict[str, Any]]:
        if self._needs_reset:
            raise RuntimeError("episode has ended or not started; call reset() before step()")
        requested = self._actions.validate_code(action)

        # 1. apply the action: move committed capacity to the requested target in
        # one step; new replicas start pending and follow the startup lifecycle
        committed = self._pool.desired_count
        target = self._actions.target_for(requested, committed)
        started: tuple[Any, ...] = ()
        if target > committed:
            applied = self._pool.scale_up(target - committed)
            started = self._pool.last_started
        elif target < committed:
            applied = -self._pool.scale_down(committed - target)
        else:
            applied = 0
        tick_replicas = self.replica_counts

        # 2-4. serve this tick's demand with the currently active replicas at the
        # tick's realized capacity (one draw per tick, whatever the action)
        tick = self._clock.step_count
        request_rate = self._replay.next_demand()
        multiplier = self._capacity_multiplier()
        queue_result = self._queue.step(request_rate, self._pool.active_count, multiplier)
        metrics = compute_tick_metrics(
            queue_result,
            active_replicas=self._pool.active_count,
            pending_replicas=self._pool.pending_count,
            config=self.config,
            capacity_multiplier=multiplier,
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

        # 8. info describes what physically happened; telemetry records it as a
        # completed measurement for (possibly delayed) controller views
        truncated = self._replay.is_exhausted
        self._needs_reset = truncated
        per_replica = self.config.replicas.service_capacity_rps * multiplier

        info: dict[str, Any] = {
            "tick": tick,
            "time_seconds": self._clock.time_seconds,
            "request_rate": request_rate,
            "requested_action": requested,
            "requested_replica_target": target,
            "applied_replica_change": applied,
            **tick_replicas,
            **_queue_info(queue_result),
            "utilization": metrics.utilization,
            "p95_latency_seconds": metrics.p95_latency_seconds,
            "sla_violated": metrics.sla_violated,
            "infrastructure_cost": metrics.infrastructure_cost,
            "capacity_multiplier": multiplier,
            "effective_service_capacity_rps_per_replica": per_replica,
            "effective_total_service_capacity_rps": tick_replicas["active_replicas"] * per_replica,
            "reward_components": {
                "latency_penalty": breakdown.latency_penalty,
                "cost_penalty": breakdown.cost_penalty,
                "sla_penalty": breakdown.sla_penalty,
                "queue_penalty": breakdown.queue_penalty,
                "churn_penalty": breakdown.churn_penalty,
            },
            "reward": breakdown.reward,
        }
        if self._startup_model != FIXED_V1:
            interval = self.config.timing.control_interval_seconds
            info |= {
                "startup_delay_model": self._startup_model,
                "startup_realized_delays_seconds": [r.realized_delay_seconds for r in started],
                "startup_multipliers": [r.multiplier for r in started],
                "physical_pending_by_ticks": list(
                    self._pool.physical_pending_by_ticks_until_active(interval)
                ),
            }
        measurements = {
            k: v
            for k, v in info.items()
            if k not in CONTROL_PLANE_KEYS and k not in PHYSICAL_ONLY_KEYS
        }
        self._telemetry.appendleft(TelemetrySnapshot(tick, MappingProxyType(measurements)))
        return self._observation(), breakdown.reward, False, truncated, info

    def _draw_startup_multiplier(self) -> float:
        """One per-replica startup multiplier from the dedicated startup RNG."""
        return draw_tri_point_multiplier(self._startup_rng)

    def _capacity_multiplier(self) -> float:
        if self._jitter == 0:
            return 1.0  # nominal: exact, and no RNG draw
        return float(self._dynamics_rng.uniform(1.0 - self._jitter, 1.0 + self._jitter))

    def visible_telemetry(self) -> tuple[TelemetrySnapshot, ...]:
        """Completed-tick snapshots a controller can see now, newest first.

        With delay ``d`` the newest visible snapshot is the completed tick
        ``d`` ticks before the latest one; empty until such a tick exists.
        """
        snapshots = tuple(self._telemetry)
        return snapshots[self._delay_ticks : self._delay_ticks + self._history_ticks]

    def decision_info(self, info: Mapping[str, Any]) -> dict[str, Any]:
        """Controller-facing view of the latest step (or reset) ``info``.

        Control-plane facts (tick, time, requested/applied action) come from
        ``info``; replica counts are the current ones; load/queue/latency/cost
        measurements come from the telemetry visible under the configured delay
        (``telemetry_tick`` names their tick). With no delay this equals ``info``
        plus current replica counts. Without a visible measurement yet, no
        measurement fields are present, so controllers see "no sample".
        """
        view = {key: value for key, value in info.items() if key in CONTROL_PLANE_KEYS}
        visible = self.visible_telemetry()
        if visible:
            view |= dict(visible[0].measurements)
            view["telemetry_tick"] = visible[0].tick
        return view | self.replica_counts

    def _observation(self) -> Observation:
        visible = self.visible_telemetry()
        latest = (
            Measurement(
                utilization=visible[0].utilization,
                queued_requests=visible[0].queued_requests,
                p95_latency_seconds=visible[0].p95_latency_seconds,
                infrastructure_cost=visible[0].infrastructure_cost,
            )
            if visible
            else None
        )
        return build_observation(
            self._observation_constants,
            request_rates_newest_first=[snapshot.request_rate for snapshot in visible],
            latest=latest,
            active_replicas=self._pool.active_count,
            pending_by_ticks=self._pending_buckets(),
            completed_ticks=self._clock.step_count,
            episode_ticks=self._episode_ticks,
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
