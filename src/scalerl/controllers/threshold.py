"""Reactive threshold (target-tracking) autoscaler on modeled utilization."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from scalerl.environment.actions import HOLD, SCALE_DOWN, SCALE_UP, ActionContract
from scalerl.environment.gym_env import Observation

DecisionReason = Literal[
    "no_sample",
    "above_max",
    "below_min",
    "above_high",
    "below_low",
    "at_max",
    "at_min",
    "cooldown",
    "within_band",
]


@dataclass(frozen=True, slots=True)
class ThresholdDecision:
    """Why the controller chose its latest action.

    ``cooldown_remaining`` counts future decisions still blocked by cooldown
    after this one (0 when no cooldown is pending). ``action`` is the emitted
    environment action code; ``target_replicas`` is the committed capacity it
    requests (set only when the controller was given an action contract).
    ``desired_replicas`` is the committed capacity (active + pending) the
    decision started from.
    """

    utilization: float | None
    desired_replicas: int
    action: int
    reason: DecisionReason
    cooldown_remaining: int = 0
    target_replicas: int | None = None


class ThresholdController:
    """Scale one replica at a time when utilization leaves ``[low, high]``.

    Utilization is ``info["utilization"]`` from the last completed tick. Reset
    ``info`` has none, so the first decision holds rather than reading the
    zero-filled reset observation as 0% load. Desired capacity is
    ``active + pending`` so replicas still starting up are not re-requested.

    Capacity outside ``[min_replicas, max_replicas]`` (possible when these
    differ from the environment's bounds) is first stepped back inside,
    regardless of utilization; only then are thresholds applied.

    **Cooldown** (``cooldown_ticks``, default 0 = off, symmetric for up and
    down) stops threshold-driven scaling from thrashing. It starts only after a
    step whose ``info["applied_replica_change"]`` is non-zero; a request the
    environment clipped at a bound (applied change 0) starts nothing. With
    ``cooldown_ticks = N`` the next N decisions hold with reason ``"cooldown"``
    and decision N + 1 applies thresholds again. Each completed tick's change
    (identified by ``info["tick"]``) is consumed once, so re-reading the same
    ``info`` never restarts a cooldown. Every decision while a cooldown is
    pending uses one of its N slots.

    Decision priority: no utilization sample yet (hold), then capacity outside
    the controller bounds (always corrected, even during cooldown), then
    cooldown, then thresholds.

    Thresholds are crossed strictly: utilization equal to a threshold holds.
    ``0 <= low_threshold < high_threshold < 1``; a ``high_threshold`` of 1
    is rejected because utilization saturates at exactly 1, so scale-up could
    never trigger. ``low_threshold == 0`` is allowed and disables scale-down.

    Apart from cooldown, decisions depend only on the current inputs;
    ``last_decision`` is read-only diagnostics.

    **Action contracts (#79).** The control law is the same under every
    contract: one more replica, one fewer, or hold. Without ``action_contract``
    (or with ``delta-v1``) the controller emits the historical codes 0/1/2
    unchanged. Under ``desired-replicas-v1`` the same decision is encoded as
    the target ``committed + 1`` / ``committed - 1`` / ``committed`` (clipped to
    the environment's bounds); it is not a proportional (HPA-style) rule.
    """

    def __init__(
        self,
        *,
        low_threshold: float,
        high_threshold: float,
        min_replicas: int,
        max_replicas: int,
        cooldown_ticks: int = 0,
        action_contract: ActionContract | None = None,
    ) -> None:
        for name, value in (("low_threshold", low_threshold), ("high_threshold", high_threshold)):
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise TypeError(f"{name} must be a number")
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if not 0 <= low_threshold < high_threshold < 1:
            raise ValueError("thresholds must satisfy 0 <= low_threshold < high_threshold < 1")
        for name, count in (("min_replicas", min_replicas), ("max_replicas", max_replicas)):
            if isinstance(count, bool) or not isinstance(count, int):
                raise TypeError(f"{name} must be an integer")
        if min_replicas < 1:
            raise ValueError("min_replicas must be at least 1")
        if max_replicas < min_replicas:
            raise ValueError("max_replicas must be at least min_replicas")
        if isinstance(cooldown_ticks, bool) or not isinstance(cooldown_ticks, int):
            raise TypeError("cooldown_ticks must be an integer")
        if cooldown_ticks < 0:
            raise ValueError("cooldown_ticks must be non-negative")

        self._low = float(low_threshold)
        self._high = float(high_threshold)
        self._min_replicas = min_replicas
        self._max_replicas = max_replicas
        self._cooldown_ticks = cooldown_ticks
        self._contract = action_contract
        self._cooldown_remaining = 0
        self._consumed_tick: Any = None
        self._last_decision: ThresholdDecision | None = None

    @property
    def low_threshold(self) -> float:
        return self._low

    @property
    def high_threshold(self) -> float:
        return self._high

    @property
    def min_replicas(self) -> int:
        return self._min_replicas

    @property
    def max_replicas(self) -> int:
        return self._max_replicas

    @property
    def cooldown_ticks(self) -> int:
        return self._cooldown_ticks

    @property
    def action_contract(self) -> ActionContract | None:
        return self._contract

    @property
    def last_decision(self) -> ThresholdDecision | None:
        """Return the most recent decision since reset, if any."""
        return self._last_decision

    def reset(self, seed: int | None = None) -> None:
        """Clear diagnostics and cooldown state; ``seed`` is unused (deterministic)."""
        self._cooldown_remaining = 0
        self._consumed_tick = None
        self._last_decision = None

    def act(self, observation: Observation, info: Mapping[str, Any]) -> int:
        utilization: float | None = info.get("utilization")
        desired = info["active_replicas"] + info["pending_replicas"]
        self._start_cooldown_after_applied_change(info)
        cooling = self._cooldown_remaining > 0

        reason: DecisionReason
        if utilization is None:
            action, reason = HOLD, "no_sample"
        elif desired > self._max_replicas:
            action, reason = SCALE_DOWN, "above_max"
        elif desired < self._min_replicas:
            action, reason = SCALE_UP, "below_min"
        elif cooling:
            action, reason = HOLD, "cooldown"
        elif utilization > self._high:
            if desired < self._max_replicas:
                action, reason = SCALE_UP, "above_high"
            else:
                action, reason = HOLD, "at_max"
        elif utilization < self._low:
            if desired > self._min_replicas:
                action, reason = SCALE_DOWN, "below_low"
            else:
                action, reason = HOLD, "at_min"
        else:
            action, reason = HOLD, "within_band"

        if cooling:
            self._cooldown_remaining -= 1
        target: int | None = None
        if self._contract is not None:
            action = self._contract.code_for_step(action, desired)
            target = self._contract.target_for(action, desired)
        self._last_decision = ThresholdDecision(
            utilization, desired, action, reason, self._cooldown_remaining, target
        )
        return action

    def _start_cooldown_after_applied_change(self, info: Mapping[str, Any]) -> None:
        """Start cooldown if the last completed step really changed the fleet."""
        if self._cooldown_ticks == 0 or "applied_replica_change" not in info:
            return
        tick = info.get("tick")
        if tick is not None and tick == self._consumed_tick:
            return  # this tick's change was already counted
        self._consumed_tick = tick
        if info["applied_replica_change"] != 0:
            self._cooldown_remaining = self._cooldown_ticks
