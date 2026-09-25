"""Tests for PPO training (#16) and the algorithm-generic SB3 bundle layer.

Tiny budgets only (≤ 256 timesteps, 16x16 networks); no performance assertions.
"""

import json
import warnings
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import torch
from mlflow import MlflowClient
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecNormalize

from scalerl.benchmarks import build_workload, load_benchmark_manifest
from scalerl.controllers import Controller
from scalerl.environment import (
    AutoscalingEnv,
    ObservationConfig,
    ReplicaConfig,
    SimulatorConfig,
    SlaConfig,
)
from scalerl.evaluation import EpisodeMetrics, evaluate_controller_episode
from scalerl.mlops import EnvironmentCompatibility
from scalerl.rl import (
    ModelMetadata,
    SB3Controller,
    load_sb3_controller,
    read_model_bundle,
    save_model_bundle,
)
from scalerl.training import common
from scalerl.training import ppo as training
from scalerl.training.common import MODEL_SOURCE_TAG
from scalerl.training.dqn import DQNHyperparameters, build_dqn
from scalerl.training.ppo import (
    DEFAULT_CHECKPOINT_INTERVAL,
    DEFAULT_TIMESTEPS,
    PPO_CONFIG_VERSION,
    PPOHyperparameters,
    PPOTrainingResult,
    build_ppo,
    load_hyperparameters,
    train_ppo,
)

TINY = PPOHyperparameters(n_steps=64, batch_size=32, n_epochs=2, net_arch=(16, 16))
TRAIN = "syn-train-spike"
VALIDATION = ("syn-val-bursty", "syn-val-steady-high")


def env_for(workload_id: str = TRAIN, config: SimulatorConfig | None = None) -> AutoscalingEnv:
    entry = load_benchmark_manifest().get(workload_id)
    return AutoscalingEnv(config or SimulatorConfig(), build_workload(entry))


def metadata(algorithm: str, hyperparameters: Any) -> ModelMetadata:
    return ModelMetadata(
        algorithm=algorithm,  # type: ignore[arg-type]
        config_version=f"{algorithm}-v1",
        benchmark_version="v1",
        training_workload_id=TRAIN,
        training_workload_split="train",
        seed=0,
        total_timesteps=64,
        hyperparameters=hyperparameters.as_params(),
        scalerl_version="test",
    )


def bundle_of(model: Any, algorithm: str, hyperparameters: Any, directory: Path) -> Path:
    compatibility = EnvironmentCompatibility.from_env(env_for(), "v1")
    return save_model_bundle(
        model,
        directory,
        metadata=metadata(algorithm, hyperparameters),
        compatibility=compatibility,
    )


@pytest.fixture(scope="module")
def ppo_bundle(tmp_path_factory: pytest.TempPathFactory) -> tuple[PPO, Path]:
    model = build_ppo(Monitor(env_for()), TINY, seed=0)
    model.learn(total_timesteps=64)
    return model, bundle_of(model, "ppo", TINY, tmp_path_factory.mktemp("ppo"))


@pytest.fixture(scope="module")
def dqn_bundle(tmp_path_factory: pytest.TempPathFactory) -> Path:
    hp = DQNHyperparameters(buffer_size=500, learning_starts=16, batch_size=16, net_arch=(16, 16))
    model = build_dqn(Monitor(env_for()), hp, seed=0)
    model.learn(total_timesteps=64)
    return bundle_of(model, "dqn", hp, tmp_path_factory.mktemp("dqn"))


@pytest.fixture
def tracking_uri(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.setenv("MLFLOW_DISABLE_AGENT_HINT", "1")
    yield f"sqlite:///{tmp_path / 'mlflow.db'}"


# --- configuration ----------------------------------------------------------------------------


def test_ppo_v1_defaults_and_budget_are_exact() -> None:
    defaults = PPOHyperparameters()

    assert PPO_CONFIG_VERSION == "ppo-v1"
    assert (defaults.n_steps, defaults.batch_size, defaults.n_epochs) == (2048, 64, 10)
    assert (defaults.learning_rate, defaults.gamma, defaults.gae_lambda) == (3e-4, 0.99, 0.95)
    assert (defaults.clip_range, defaults.ent_coef, defaults.vf_coef) == (0.2, 0.0, 0.5)
    assert defaults.max_grad_norm == 0.5 and defaults.normalize_advantage is True
    assert DEFAULT_TIMESTEPS == 204_800 == 100 * defaults.n_steps
    assert defaults.n_steps % defaults.batch_size == 0
    assert DEFAULT_CHECKPOINT_INTERVAL % defaults.n_steps == 0
    # Separate actor and critic networks of the same widths.
    assert defaults.sb3_kwargs()["policy_kwargs"] == {"net_arch": {"pi": [64, 64], "vf": [64, 64]}}


@pytest.mark.parametrize(
    "overrides",
    [
        {"learning_rate": 0.0},
        {"n_steps": 1},
        {"batch_size": 1},
        {"gamma": 0.0},
        {"gae_lambda": 1.5},
        {"clip_range": 0.0},
        {"ent_coef": -0.01},
        {"net_arch": ()},
        {"net_arch": (0, 8)},
        {"n_steps": 64, "batch_size": 128},  # minibatch larger than the rollout
        {"n_steps": 100, "batch_size": 64},  # would truncate the last minibatch
        {"buffer_size": 1000},  # DQN settings do not exist for PPO
        {"learning_starts": 10},
    ],
)
def test_invalid_ppo_hyperparameters_are_rejected(overrides: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        PPOHyperparameters(**overrides)


def test_ppo_is_built_on_the_raw_env_without_vecnormalize() -> None:
    env = env_for()
    model = build_ppo(Monitor(env), TINY, seed=0)

    wrapped = model.get_env()
    assert not isinstance(wrapped, VecNormalize)
    # The policy sees exactly AutoscalingEnv's observation space and reward.
    assert model.observation_space == env.observation_space
    assert model.action_space.n == 3  # type: ignore[attr-defined]
    observation, _ = env.reset(seed=0)
    assert wrapped is not None
    vec_observation = wrapped.reset()
    assert vec_observation[0] == pytest.approx(observation)


def test_ppo_training_raises_no_sb3_warnings() -> None:
    model = build_ppo(Monitor(env_for()), TINY, seed=0)
    with warnings.catch_warnings():
        warnings.filterwarnings("error", module="stable_baselines3.*")
        model.learn(total_timesteps=128)
    assert model.num_timesteps == 128


# --- generic SB3 bundles: DQN and PPO -------------------------------------------------------------


@pytest.mark.parametrize("algorithm", ["dqn", "ppo"])
def test_model_metadata_accepts_both_algorithms(algorithm: str) -> None:
    hp = DQNHyperparameters() if algorithm == "dqn" else PPOHyperparameters()
    assert metadata(algorithm, hp).algorithm == algorithm
    with pytest.raises(ValueError):
        metadata("a2c", hp)


def test_generic_loader_loads_ppo_and_dqn_bundles(
    ppo_bundle: tuple[PPO, Path], dqn_bundle: Path
) -> None:
    env = env_for("syn-val-bursty")

    ppo_controller = load_sb3_controller(ppo_bundle[1], env)
    dqn_controller = load_sb3_controller(dqn_bundle, env)

    assert isinstance(ppo_controller.model, PPO)
    assert isinstance(dqn_controller.model, DQN)
    for controller in (ppo_controller, dqn_controller):
        assert isinstance(controller, Controller)
        metrics = evaluate_controller_episode(env, controller).metrics
        assert isinstance(metrics, EpisodeMetrics)
    assert read_model_bundle(ppo_bundle[1])[0].algorithm == "ppo"
    assert read_model_bundle(dqn_bundle)[0].algorithm == "dqn"


def test_existing_dqn_bundle_format_still_loads(dqn_bundle: Path, tmp_path: Path) -> None:
    # A bundle written before PPO existed has exactly these three files and fields.
    payload = json.loads((dqn_bundle / "metadata.json").read_text())
    assert payload["algorithm"] == "dqn" and payload["bundle_version"] == "scalerl-sb3-bundle-v1"
    copy = tmp_path / "old-dqn"
    copy.mkdir()
    for name in ("model.zip", "compatibility.json", "metadata.json"):
        (copy / name).write_bytes((dqn_bundle / name).read_bytes())

    assert isinstance(load_sb3_controller(copy, env_for()).model, DQN)


def test_ppo_adapter_is_deterministic_and_ignores_info(ppo_bundle: tuple[PPO, Path]) -> None:
    controller = SB3Controller(ppo_bundle[0])
    env = env_for()
    observation, info = env.reset(seed=0)
    poisoned = {**info, "request_rate": 1e9, "queued_requests": 1e9, "future": [1e9] * 5}

    actions = [controller.act(observation, variant) for variant in (info, {}, poisoned) * 3]

    assert len(set(actions)) == 1 and type(actions[0]) is int and actions[0] in (0, 1, 2)


@pytest.mark.parametrize(
    ("field", "config"),
    [
        (
            "traffic_history_ticks",
            SimulatorConfig(observation=ObservationConfig(traffic_history_ticks=6)),
        ),
        ("startup_delay_seconds", SimulatorConfig(replicas=ReplicaConfig(startup_delay_seconds=0))),
        ("max_replicas", SimulatorConfig(replicas=ReplicaConfig(max_replicas=12))),
        ("service_capacity_rps", SimulatorConfig(replicas=ReplicaConfig(service_capacity_rps=10))),
        ("cost_per_hour", SimulatorConfig(replicas=ReplicaConfig(cost_per_hour=1.0))),
        ("latency_target_seconds", SimulatorConfig(sla=SlaConfig(latency_target_seconds=1.0))),
    ],
)
def test_incompatible_environments_reject_ppo_bundles(
    ppo_bundle: tuple[PPO, Path], field: str, config: SimulatorConfig
) -> None:
    with pytest.raises(ValueError, match=f"incompatible environment: .*{field}"):
        load_sb3_controller(ppo_bundle[1], env_for("syn-val-bursty", config))


# --- guardrails -------------------------------------------------------------------------------


@pytest.fixture
def no_training(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("training started before the guardrail")

    monkeypatch.setattr(training, "build_ppo", refuse)
    monkeypatch.setattr(common, "build_workloads", refuse)


@pytest.mark.parametrize(
    ("workload", "validation", "message"),
    [
        ("syn-test-spike-hard", VALIDATION, "PPO trains on train workloads only.*held-out test"),
        ("azure-test-993600", VALIDATION, "held-out test"),
        (TRAIN, ("syn-test-seasonal-shifted",), "PPO validation.*held-out test"),
        (TRAIN, ("syn-train-bursty",), "a train workload"),
        ("syn-val-bursty", VALIDATION, "a validation workload"),
    ],
)
def test_wrong_splits_are_rejected_before_training(
    workload: str,
    validation: tuple[str, ...],
    message: str,
    no_training: None,
    tracking_uri: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        train_ppo(
            workload_id=workload,
            validation_workload_ids=validation,
            hyperparameters=TINY,
            timesteps=128,
            tracking_uri=tracking_uri,
        )


@pytest.mark.parametrize(("timesteps", "trained"), [(1, 64), (100, 128), (200, 256)])
def test_budgets_sb3_would_overshoot_are_refused_before_training(
    timesteps: int, trained: int, no_training: None, tracking_uri: str
) -> None:
    with pytest.raises(ValueError, match=f"multiple of n_steps \\(64\\).*{trained} timesteps"):
        train_ppo(
            workload_id=TRAIN,
            validation_workload_ids=VALIDATION,
            hyperparameters=TINY,
            timesteps=timesteps,
            tracking_uri=tracking_uri,
        )


def test_default_budget_needs_whole_2048_rollouts(no_training: None, tracking_uri: str) -> None:
    with pytest.raises(ValueError, match="multiple of n_steps \\(2048\\).*200704"):
        train_ppo(workload_id=TRAIN, validation_workload_ids=VALIDATION, timesteps=200_000)


def test_checkpoint_interval_must_fall_on_rollouts(no_training: None, tracking_uri: str) -> None:
    with pytest.raises(ValueError, match="checkpoint interval \\(100\\).*n_steps \\(64\\)"):
        train_ppo(
            workload_id=TRAIN,
            validation_workload_ids=VALIDATION,
            hyperparameters=TINY,
            timesteps=128,
            checkpoint_interval=100,
            tracking_uri=tracking_uri,
        )


# --- tracked training -------------------------------------------------------------------------


def run_tiny(tracking_uri: str, **kwargs: Any) -> PPOTrainingResult:
    options: dict[str, Any] = {
        "workload_id": TRAIN,
        "validation_workload_ids": VALIDATION,
        "hyperparameters": TINY,
        "timesteps": 256,
        "seed": 0,
        "tracking_uri": tracking_uri,
        "log_interval": 64,
        "checkpoint_interval": 128,
    }
    return train_ppo(**{**options, **kwargs})


def test_training_run_is_tracked_with_model_checkpoints_and_validation(
    tracking_uri: str, tmp_path: Path
) -> None:
    result = run_tiny(tracking_uri)
    client = MlflowClient(tracking_uri)
    run = client.get_run(result.training_run_id)
    tags, params, metrics = run.data.tags, run.data.params, run.data.metrics

    assert run.info.status == "FINISHED"
    assert (tags["scalerl.run_kind"], tags["scalerl.controller"]) == ("train", "ppo")
    assert run.info.run_name == f"train-ppo-{TRAIN}"
    assert (tags["scalerl.workload_id"], tags["scalerl.workload_split"]) == (TRAIN, "train")
    assert params["training_steps"] == "256" and metrics["training_timesteps"] == 256
    assert params["hp.algorithm"] == "sb3-ppo"
    assert params["hp.ppo_config_version"] == PPO_CONFIG_VERSION
    assert params["hp.observation_normalization"] == "env-v1"
    assert params["hp.reward_normalization"] == "none"
    assert params["hp.normalize_advantage"] == "true"
    assert params["hp.n_steps"] == "64" and params["hp.net_arch"] == "[16, 16]"
    assert "hp.buffer_size" not in params and "hp.exploration_fraction" not in params
    assert result.rollouts == 4

    for name in (
        "train/policy_gradient_loss",
        "train/value_loss",
        "train/entropy_loss",
        "train/approx_kl",
        "train/clip_fraction",
        "train/clip_range",
        "train/explained_variance",
        "train/learning_rate",
        "train/loss",
        "train/n_updates",
        "train/episodes",
    ):
        assert client.get_metric_history(result.training_run_id, name), name
    updates = client.get_metric_history(result.training_run_id, "train/n_updates")
    # The final update (4 rollouts × 2 epochs) is logged at the final step.
    assert max(point.value for point in updates) == 8
    assert "train/exploration_rate" not in metrics  # no epsilon exploration in PPO

    artifacts = {item.path for item in client.list_artifacts(result.training_run_id, "model")}
    assert artifacts == {"model/model.zip", "model/compatibility.json", "model/metadata.json"}
    checkpoints = {i.path for i in client.list_artifacts(result.training_run_id, "checkpoints")}
    assert checkpoints == {
        "checkpoints/step-0000128.zip",
        "checkpoints/step-0000256.zip",
        "checkpoints/manifest.json",
    }
    assert [c["step"] for c in result.checkpoints] == [128, 256]
    manifest = json.loads(
        Path(
            client.download_artifacts(
                result.training_run_id, "checkpoints/manifest.json", str(tmp_path)
            )
        ).read_text()
    )
    assert manifest["model_of_record"] == "model/" and manifest["interval_timesteps"] == 128

    bundle = Path(client.download_artifacts(result.training_run_id, "model", str(tmp_path)))
    metadata_, _ = read_model_bundle(bundle)
    assert (metadata_.algorithm, metadata_.config_version) == ("ppo", PPO_CONFIG_VERSION)
    assert metadata_.total_timesteps == 256 and metadata_.training_run_id == result.training_run_id
    checkpoint = client.download_artifacts(
        result.training_run_id, "checkpoints/step-0000128.zip", str(tmp_path)
    )
    assert PPO.load(checkpoint, device="cpu").num_timesteps == 128

    for workload, run_id in zip(VALIDATION, result.validation_run_ids, strict=True):
        validation = client.get_run(run_id)
        assert validation.info.status == "FINISHED"
        assert validation.info.run_name == f"evaluate-ppo-{workload}"
        assert validation.data.tags["scalerl.controller"] == "ppo"
        assert validation.data.tags["scalerl.run_kind"] == "evaluate"
        assert validation.data.tags["scalerl.workload_split"] == "validation"
        assert validation.data.tags[MODEL_SOURCE_TAG] == result.training_run_id
        assert set(EpisodeMetrics.__dataclass_fields__) <= set(validation.data.metrics)


def test_checkpoints_can_be_disabled(tracking_uri: str) -> None:
    result = run_tiny(tracking_uri, checkpoint_interval=None, timesteps=128)
    client = MlflowClient(tracking_uri)

    assert result.checkpoints == () and result.checkpoint_interval is None
    assert client.list_artifacts(result.training_run_id, "checkpoints") == []


def test_actor_and_critic_learn(tracking_uri: str) -> None:
    model = build_ppo(Monitor(env_for()), TINY, seed=0)
    actor = [p.detach().clone() for p in model.policy.mlp_extractor.policy_net.parameters()]
    critic = [p.detach().clone() for p in model.policy.mlp_extractor.value_net.parameters()]

    model.learn(total_timesteps=128)

    after_actor = list(model.policy.mlp_extractor.policy_net.parameters())
    after_critic = list(model.policy.mlp_extractor.value_net.parameters())
    assert any(not torch.equal(a, b) for a, b in zip(actor, after_actor, strict=True))
    assert any(not torch.equal(a, b) for a, b in zip(critic, after_critic, strict=True))


def test_result_json_round_trips_and_reloads_hyperparameters(
    tracking_uri: str, tmp_path: Path
) -> None:
    result = run_tiny(tracking_uri, validation_workload_ids=("syn-val-bursty",))
    path = result.save(tmp_path / "ppo.json")

    assert PPOTrainingResult.load(path) == result
    payload = json.loads(path.read_text())
    assert payload["model_artifact_uri"] == f"runs:/{result.training_run_id}/model"
    assert payload["observation_normalization"] == "env-v1"
    assert path.stat().st_size < 20_000
    hp_file = tmp_path / "hp.json"
    hp_file.write_text(result.hyperparameters.model_dump_json())
    assert load_hyperparameters(hp_file) == (TINY, "file:hp.json")


def test_same_seed_reproduces_the_policy_on_this_machine(tracking_uri: str) -> None:
    first = run_tiny(tracking_uri, checkpoint_interval=None)
    second = run_tiny(tracking_uri, checkpoint_interval=None)

    assert first.validation_metrics == second.validation_metrics


def test_cli_trains_and_writes_the_result(tracking_uri: str, tmp_path: Path) -> None:
    output = tmp_path / "result.json"
    hp_file = tmp_path / "hp.json"
    hp_file.write_text(TINY.model_dump_json())

    exit_code = training.main(
        [
            "--workload",
            TRAIN,
            "--validation-workload",
            "syn-val-bursty",
            "--timesteps",
            "128",
            "--hyperparameters",
            str(hp_file),
            "--hp",
            "ent_coef=0.01",
            "--checkpoint-interval",
            "0",
            "--tracking-uri",
            tracking_uri,
            "--output",
            str(output),
        ]
    )

    result = PPOTrainingResult.load(output)
    assert exit_code == 0
    assert result.hyperparameter_source == "file:hp.json+overrides"
    assert result.hyperparameters.ent_coef == 0.01
    assert result.checkpoints == ()
