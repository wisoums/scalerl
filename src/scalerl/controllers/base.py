"""Common controller interface and a minimal episode runner."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable

from scalerl.environment.gym_env import AutoscalingEnv, Observation


@runtime_checkable
class Controller(Protocol):
    """Chooses one ``AutoscalingEnv`` action (0 down, 1 hold, 2 up) per tick.

    ``info`` is the dict returned by the latest ``reset()`` or ``step()``.
    Rule-based controllers may read raw values from it; learned-policy
    adapters can use only the observation.
    """

    def reset(self, seed: int | None = None) -> None:
        """Reset per-episode state; a seed makes stochastic controllers reproducible."""
        ...

    def act(self, observation: Observation, info: Mapping[str, Any]) -> int:
        """Return the action for the next tick."""
        ...


def run_episode(
    env: AutoscalingEnv, controller: Controller, *, seed: int | None = None
) -> list[dict[str, Any]]:
    """Run one full episode and return the ``info`` dict of every step.

    The same ``seed`` resets both the environment and the controller.
    """
    observation, info = env.reset(seed=seed)
    controller.reset(seed=seed)

    infos = []
    done = False
    while not done:
        action = controller.act(observation, info)
        observation, _, terminated, truncated, info = env.step(action)
        infos.append(info)
        done = terminated or truncated
    return infos
