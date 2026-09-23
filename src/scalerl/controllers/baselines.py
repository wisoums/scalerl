"""Random and static-capacity reference controllers."""

from __future__ import annotations

import random
from collections.abc import Mapping
from typing import Any

from scalerl.environment.config import ReplicaConfig
from scalerl.environment.gym_env import HOLD, SCALE_DOWN, SCALE_UP, Observation


class RandomController:
    """Uniformly random actions from a local seeded RNG.

    ``reset(seed)`` restarts the action sequence for that seed;
    ``reset(None)`` continues the current sequence. Global random state is
    never touched.
    """

    def __init__(self, seed: int | None = None) -> None:
        _check_seed(seed)
        self._rng = random.Random(seed)

    def reset(self, seed: int | None = None) -> None:
        _check_seed(seed)
        if seed is not None:
            self._rng = random.Random(seed)

    def act(self, observation: Observation, info: Mapping[str, Any]) -> int:
        return self._rng.choice((SCALE_DOWN, HOLD, SCALE_UP))


class StaticController:
    """Converge to a fixed number of replicas, then hold regardless of traffic.

    Desired capacity is ``active + pending`` from ``info``, so replicas still
    starting up already count toward the target.
    """

    def __init__(self, target_replicas: int, replicas: ReplicaConfig) -> None:
        if isinstance(target_replicas, bool) or not isinstance(target_replicas, int):
            raise TypeError("target_replicas must be an integer")
        if not replicas.min_replicas <= target_replicas <= replicas.max_replicas:
            raise ValueError(
                f"target_replicas {target_replicas} is outside replica bounds "
                f"[{replicas.min_replicas}, {replicas.max_replicas}]"
            )
        self._target = target_replicas

    @property
    def target_replicas(self) -> int:
        """Return the replica count this controller maintains."""
        return self._target

    def reset(self, seed: int | None = None) -> None:
        """No per-episode state; ``seed`` is accepted for interface compatibility."""

    def act(self, observation: Observation, info: Mapping[str, Any]) -> int:
        desired = info["active_replicas"] + info["pending_replicas"]
        if desired < self._target:
            return SCALE_UP
        if desired > self._target:
            return SCALE_DOWN
        return HOLD


def _check_seed(seed: int | None) -> None:
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
        raise TypeError("seed must be an integer or None")
