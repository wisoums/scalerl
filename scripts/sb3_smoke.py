"""Tiny Stable-Baselines3 learning smoke on the real ScaleRL environment (CPU only).

Infrastructure check, not an experiment, for both deep-RL algorithms (DQN #15,
PPO #16): it proves that ScaleRL is a valid Gymnasium environment SB3 can train
on, that PyTorch runs forward/backward passes (DQN's Q-network; PPO's actor and
critic), that each trained policy can drive the environment, and that each
model survives a save/load round-trip. It makes no claim about quality,
reward design, convergence, or hyperparameters, and has no performance
assertions.

No MLflow, services, Azure data, or GPU. Run it anywhere ScaleRL is installed:

    python scripts/sb3_smoke.py
    docker compose run --rm -T --no-deps trainer python - < scripts/sb3_smoke.py

The training pipelines themselves are ``scalerl.training.dqn`` / ``.ppo``.
"""

from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.env_checker import check_env

from scalerl.environment import AutoscalingEnv, ReplicaConfig, SimulatorConfig, TimingConfig
from scalerl.workloads import ramp_workload

INTERVAL_SECONDS = 30.0
EPISODE_TICKS = 16
TOTAL_TIMESTEPS = 64  # DQN: four short episodes, every step trains; PPO: two 32-step rollouts
PPO_ROLLOUT = 32


def check(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"FAIL: {message}")
    print(f"ok   {message}")


def make_env() -> AutoscalingEnv:
    duration = INTERVAL_SECONDS * EPISODE_TICKS
    config = SimulatorConfig(
        timing=TimingConfig(
            control_interval_seconds=INTERVAL_SECONDS, episode_duration_seconds=duration
        ),
        replicas=ReplicaConfig(min_replicas=1, max_replicas=6, initial_replicas=2),
    )
    trace = ramp_workload(
        duration_seconds=duration,
        control_interval_seconds=INTERVAL_SECONDS,
        start_rate=40.0,
        end_rate=240.0,
    )
    return AutoscalingEnv(config, trace)


def snapshot(module: torch.nn.Module) -> list[torch.Tensor]:
    return [parameter.detach().clone() for parameter in module.parameters()]


def changed(before: list[torch.Tensor], module: torch.nn.Module) -> bool:
    return any(
        not torch.equal(old, new) for old, new in zip(before, module.parameters(), strict=True)
    )


def drive_and_round_trip(name: str, model: BaseAlgorithm, algorithm: type[BaseAlgorithm]) -> None:
    """The trained policy drives the env, and survives save/load with the same action."""
    env = make_env()
    observation, _ = env.reset(seed=0)
    action, _ = model.predict(observation, deterministic=True)
    check(env.action_space.contains(int(action)), f"{name} predicts a valid action ({action})")
    _, reward, _, _, info = env.step(int(action))
    check(bool(np.isfinite(reward)), f"{name}'s action drives the env (reward {reward:.3f})")
    check(info["requested_action"] == int(action), f"the env applied {name}'s action")

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / f"{name.lower()}-smoke.zip"
        model.save(path)
        loaded = algorithm.load(path, env=make_env(), device="cpu")
    reloaded_action, _ = loaded.predict(observation, deterministic=True)
    check(int(reloaded_action) == int(action), f"saved/loaded {name} predicts the same action")


def dqn_smoke() -> None:
    env = make_env()
    model = DQN(
        "MlpPolicy",
        env,
        seed=0,
        device="cpu",
        learning_starts=0,
        buffer_size=256,
        batch_size=16,
        train_freq=1,
        gradient_steps=1,
        target_update_interval=16,
        exploration_fraction=0.5,
        policy_kwargs={"net_arch": [32, 32]},
        verbose=0,
    )
    check(model.device.type == "cpu", "DQN runs on CPU")
    before = snapshot(model.q_net)
    model.learn(total_timesteps=TOTAL_TIMESTEPS)
    updates = int(model._n_updates)
    check(
        model.num_timesteps == TOTAL_TIMESTEPS, f"DQN learned for {model.num_timesteps} timesteps"
    )
    check(updates > 0, f"DQN ran {updates} gradient updates")
    check(changed(before, model.q_net), "DQN Q-network weights changed")
    drive_and_round_trip("DQN", model, DQN)


def ppo_smoke() -> None:
    model = PPO(
        "MlpPolicy",
        make_env(),
        seed=0,
        device="cpu",
        n_steps=PPO_ROLLOUT,
        batch_size=16,
        n_epochs=1,
        policy_kwargs={"net_arch": {"pi": [32, 32], "vf": [32, 32]}},
        verbose=0,
    )
    check(model.device.type == "cpu", "PPO runs on CPU")
    actor = snapshot(model.policy.mlp_extractor.policy_net)
    critic = snapshot(model.policy.mlp_extractor.value_net)
    model.learn(total_timesteps=TOTAL_TIMESTEPS)
    check(
        model.num_timesteps == TOTAL_TIMESTEPS, f"PPO learned for {model.num_timesteps} timesteps"
    )
    check(changed(actor, model.policy.mlp_extractor.policy_net), "PPO actor weights changed")
    check(changed(critic, model.policy.mlp_extractor.value_net), "PPO critic weights changed")
    drive_and_round_trip("PPO", model, PPO)


def main() -> int:
    started = time.perf_counter()
    torch.set_num_threads(1)
    env = make_env()
    check_env(env, warn=True)
    check(True, f"ScaleRL passes SB3's Gymnasium env checker (obs {env.observation_space.shape})")
    dqn_smoke()
    ppo_smoke()

    print(f"sb3 smoke: passed in {time.perf_counter() - started:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
