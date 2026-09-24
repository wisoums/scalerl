"""Reactive threshold (target-tracking) autoscaler on modeled utilization."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from scalerl.environment.gym_env import HOLD, SCALE_DOWN, SCALE_UP, Observation

DecisionReason = Literal["no_sample", "above_high", "below_low", "at_max", "at_min", "within_band"]


@dataclass(frozen=True, slots=True)
class ThresholdDecision:
    """Why the controller chose its latest action."""

    utilization: float | None
    desired_replicas: int
    action: int
    reason: DecisionReason


class ThresholdController:
    """Scale one replica at a time when utilization leaves ``[low, high]``.

    Utilization is ``info["utilization"]`` from the last completed tick. Reset
    ``info`` has none, so the first decision holds rather than reading the
    zero-filled reset observation as 0% load. Desired capacity is
    ``active + pending`` so replicas still starting up are not re-requested.

    Thresholds are crossed strictly: utilization equal to a threshold holds.
    ``0 <= low_threshold < high_threshold < 1``; a ``high_threshold`` of 1
    is rejected because utilization saturates at exactly 1, so scale-up could
    never trigger. ``low_threshold == 0`` is allowed and disables scale-down.

    Decisions depend only on the current inputs; ``last_decision`` is
    read-only diagnostics.
    """

    def __init__(
        self,
        *,
        low_threshold: float,
        high_threshold: float,
        min_replicas: int,
        max_replicas: int,
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

        self._low = float(low_threshold)
        self._high = float(high_threshold)
        self._min_replicas = min_replicas
        self._max_replicas = max_replicas
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
    def last_decision(self) -> ThresholdDecision | None:
        """Return the most recent decision since reset, if any."""
        return self._last_decision

    def reset(self, seed: int | None = None) -> None:
        """Clear diagnostics; the controller is deterministic, so ``seed`` is unused."""
        self._last_decision = None

    def act(self, observation: Observation, info: Mapping[str, Any]) -> int:
        utilization: float | None = info.get("utilization")
        desired = info["active_replicas"] + info["pending_replicas"]

        reason: DecisionReason
        if utilization is None:
            action, reason = HOLD, "no_sample"
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

        self._last_decision = ThresholdDecision(utilization, desired, action, reason)
        return action
