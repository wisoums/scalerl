"""Common controller interface and a minimal episode runner."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable

from scalerl.environment.gym_env import AutoscalingEnv, Observation


@runtime_checkable
class Controller(Protocol):
    """Chooses one ``AutoscalingEnv`` action code per tick.

    The code's meaning is the environment's versioned action contract
    (``env.action_contract``, #79): under ``delta-v1`` 0/1/2 = down/hold/up;
    under ``desired-replicas-v1`` code ``c`` requests ``min_replicas + c``
    replicas. A controller must emit codes for the contract of the environment
    it runs in (rule-based controllers take an ``action_contract``; learned
    policies are bound to theirs by the model compatibility check).

    ``info`` is the latest ``reset()``/``step()`` info with replica counts
    replaced by the current, decision-time counts (see :func:`decision_info`).
    Rule-based controllers may read raw values from it; learned-policy
    adapters can use only the observation.
    """

    def reset(self, seed: int | None = None) -> None:
        """Reset per-episode state; a seed makes stochastic controllers reproducible."""
        ...

    def act(self, observation: Observation, info: Mapping[str, Any]) -> int:
        """Return the action for the next tick."""
        ...


def decision_info(env: AutoscalingEnv, info: Mapping[str, Any]) -> dict[str, Any]:
    """Return what a controller may know now, given the latest ``reset()``/``step()`` info.

    Replica counts are current (a step's ``info`` reports the replicas that
    served that tick; pending replicas may have activated at its end).
    Measurements (load, queue, latency, cost) are the ones visible under the
    environment's telemetry delay (#65); with no delay they are ``info``'s own.
    See :meth:`AutoscalingEnv.decision_info`.
    """
    return env.decision_info(info)


def run_episode(
    env: AutoscalingEnv, controller: Controller, *, seed: int | None = None
) -> list[dict[str, Any]]:
    """Run one full episode and return the raw ``info`` dict of every step.

    The same ``seed`` resets both the environment and the controller. The
    controller sees :func:`decision_info` (delayed telemetry, current replica
    counts); the returned infos are the unmodified physical step infos.
    """
    observation, info = env.reset(seed=seed)
    controller.reset(seed=seed)

    infos = []
    done = False
    while not done:
        action = controller.act(observation, decision_info(env, info))
        observation, _, terminated, truncated, info = env.step(action)
        infos.append(info)
        done = terminated or truncated
    return infos
