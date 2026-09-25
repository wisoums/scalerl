"""Tiny Stable-Baselines3 learning smoke on the real ScaleRL environment (CPU only).

Infrastructure check, not an experiment: it proves that ScaleRL is a valid
Gymnasium environment SB3 can train on, that PyTorch runs forward/backward
passes, that the trained policy can drive the environment, and that the model
survives a save/load round-trip. It makes no claim about DQN quality, reward
design, convergence, or hyperparameters, and has no performance assertions.

No MLflow, services, Azure data, or GPU. Run it anywhere ScaleRL is installed:

    python scripts/sb3_smoke.py
    docker compose run --rm -T --no-deps trainer python - < scripts/sb3_smoke.py

The DQN pipeline itself is Issue #15.
"""

from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env

from scalerl.environment import AutoscalingEnv, ReplicaConfig, SimulatorConfig, TimingConfig
from scalerl.workloads import ramp_workload

INTERVAL_SECONDS = 30.0
EPISODE_TICKS = 16
TOTAL_TIMESTEPS = 64  # four short episodes; every step after the first trains


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


def q_parameters(model: DQN) -> list[torch.Tensor]:
    return [parameter.detach().clone() for parameter in model.q_net.parameters()]


def main() -> int:
    started = time.perf_counter()
    torch.set_num_threads(1)
    env = make_env()
    check_env(env, warn=True)
    check(True, f"ScaleRL passes SB3's Gymnasium env checker (obs {env.observation_space.shape})")

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
    check(model.device.type == "cpu", "model runs on CPU")
    before = q_parameters(model)
    model.learn(total_timesteps=TOTAL_TIMESTEPS)
    after = q_parameters(model)
    updates = int(model._n_updates)
    check(model.num_timesteps == TOTAL_TIMESTEPS, f"learned for {model.num_timesteps} timesteps")
    check(updates > 0, f"{updates} gradient updates ran")
    check(
        any(not torch.equal(old, new) for old, new in zip(before, after, strict=True)),
        "Q-network weights changed (backward pass + optimizer step)",
    )

    observation, _ = env.reset(seed=0)
    action, _ = model.predict(observation, deterministic=True)
    check(env.action_space.contains(int(action)), f"policy predicts a valid action ({action})")
    _, reward, _, _, info = env.step(int(action))
    check(bool(np.isfinite(reward)), f"the action drives the env (reward {reward:.3f})")
    check(info["requested_action"] == int(action), "the env applied the predicted action")

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "dqn-smoke.zip"
        model.save(path)
        loaded = DQN.load(path, env=make_env(), device="cpu")
    reloaded_action, _ = loaded.predict(observation, deterministic=True)
    check(int(reloaded_action) == int(action), "saved/loaded model predicts the same action")

    print(f"sb3 smoke: passed in {time.perf_counter() - started:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
