"""Random and static-capacity reference controllers."""

from __future__ import annotations

import random
from collections.abc import Mapping
from typing import Any

from scalerl.environment.actions import HOLD, SCALE_DOWN, SCALE_UP, ActionContract
from scalerl.environment.config import DELTA_V1, ReplicaConfig
from scalerl.environment.gym_env import Observation


class RandomController:
    """Uniformly random actions from a local seeded RNG (a sanity reference only).

    ``reset(seed)`` restarts the action sequence for that seed;
    ``reset(None)`` continues the current sequence. Global random state is
    never touched.

    Without an action contract (or under ``delta-v1``) it draws uniformly from
    codes 0/1/2 exactly as before #79. Under ``desired-replicas-v1`` it draws a
    uniformly random *target* code, which behaves very differently from random
    ±1 steps; the two are not comparable and never inform the action-contract
    decision.
    """

    def __init__(
        self, seed: int | None = None, *, action_contract: ActionContract | None = None
    ) -> None:
        _check_seed(seed)
        self._rng = random.Random(seed)
        self._contract = action_contract

    def reset(self, seed: int | None = None) -> None:
        _check_seed(seed)
        if seed is not None:
            self._rng = random.Random(seed)

    def act(self, observation: Observation, info: Mapping[str, Any]) -> int:
        if self._contract is None or self._contract.semantics == DELTA_V1:
            return self._rng.choice((SCALE_DOWN, HOLD, SCALE_UP))
        return self._rng.randrange(self._contract.action_count)


class StaticController:
    """Converge to a fixed number of replicas, then hold regardless of traffic.

    Desired capacity is ``active + pending`` from ``info``, so replicas still
    starting up already count toward the target. Under ``desired-replicas-v1``
    the fixed target is requested directly (one decision); otherwise one
    replica per tick, as before #79.
    """

    def __init__(
        self,
        target_replicas: int,
        replicas: ReplicaConfig,
        *,
        action_contract: ActionContract | None = None,
    ) -> None:
        if isinstance(target_replicas, bool) or not isinstance(target_replicas, int):
            raise TypeError("target_replicas must be an integer")
        if not replicas.min_replicas <= target_replicas <= replicas.max_replicas:
            raise ValueError(
                f"target_replicas {target_replicas} is outside replica bounds "
                f"[{replicas.min_replicas}, {replicas.max_replicas}]"
            )
        self._target = target_replicas
        self._contract = action_contract

    @property
    def target_replicas(self) -> int:
        """Return the replica count this controller maintains."""
        return self._target

    def reset(self, seed: int | None = None) -> None:
        """No per-episode state; ``seed`` is accepted for interface compatibility."""

    def act(self, observation: Observation, info: Mapping[str, Any]) -> int:
        desired = info["active_replicas"] + info["pending_replicas"]
        if self._contract is not None and self._contract.semantics != DELTA_V1:
            return self._contract.code_for_target(self._target, desired)
        if desired < self._target:
            return SCALE_UP
        if desired > self._target:
            return SCALE_DOWN
        return HOLD


def _check_seed(seed: int | None) -> None:
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
        raise TypeError("seed must be an integer or None")
