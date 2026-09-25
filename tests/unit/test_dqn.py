"""Tests for the SB3 controller adapter, model bundles, and DQN training (#15).

Tiny budgets only (≤256 timesteps, 16x16 networks); no performance assertions.
"""

import json
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from mlflow import MlflowClient
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor

from scalerl.benchmarks import build_workload, load_benchmark_manifest
from scalerl.controllers import Controller, run_episode
from scalerl.environment import (
    AutoscalingEnv,
    ObservationConfig,
    ReplicaConfig,
    SimulatorConfig,
    SlaConfig,
    TimingConfig,
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
from scalerl.training import dqn as training
from scalerl.training.dqn import (
    DEFAULT_TIMESTEPS,
    DQN_CONFIG_VERSION,
    MODEL_SOURCE_TAG,
    DQNHyperparameters,
    DQNTrainingResult,
    apply_overrides,
    build_dqn,
    load_hyperparameters,
    train_dqn,
)

TINY = DQNHyperparameters(
    buffer_size=500,
    learning_starts=32,
    batch_size=16,
    target_update_interval=64,
    net_arch=(16, 16),
)
TRAIN = "syn-train-spike"
VALIDATION = ("syn-val-bursty", "syn-val-steady-high")


def env_for(workload_id: str = TRAIN, config: SimulatorConfig | None = None) -> AutoscalingEnv:
    entry = load_benchmark_manifest().get(workload_id)
    return AutoscalingEnv(config or SimulatorConfig(), build_workload(entry))


def tiny_model(env: AutoscalingEnv, timesteps: int = 64, seed: int = 0) -> DQN:
    model = build_dqn(Monitor(env), TINY, seed=seed)
    model.learn(total_timesteps=timesteps)
    return model


def save_bundle(model: DQN, env: AutoscalingEnv, directory: Path) -> Path:
    metadata = ModelMetadata(
        algorithm="dqn",
        config_version=DQN_CONFIG_VERSION,
        benchmark_version="v1",
        training_workload_id=TRAIN,
        training_workload_split="train",
        seed=0,
        total_timesteps=64,
        hyperparameters=TINY.as_params(),
        scalerl_version="test",
    )
    compatibility = EnvironmentCompatibility.from_env(env, "v1")
    return save_model_bundle(model, directory, metadata=metadata, compatibility=compatibility)


@pytest.fixture(scope="module")
def trained(tmp_path_factory: pytest.TempPathFactory) -> tuple[DQN, Path]:
    env = env_for()
    model = tiny_model(env)
    return model, save_bundle(model, env, tmp_path_factory.mktemp("bundle"))


@pytest.fixture
def tracking_uri(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.setenv("MLFLOW_DISABLE_AGENT_HINT", "1")
    yield f"sqlite:///{tmp_path / 'mlflow.db'}"


# --- hyperparameters ----------------------------------------------------------------------


def test_v1_defaults_are_a_full_training_config_not_the_ci_smoke() -> None:
    defaults = DQNHyperparameters()

    assert DEFAULT_TIMESTEPS == 200_000
    assert DQN_CONFIG_VERSION == "dqn-v1"
    assert defaults.buffer_size == 100_000 and defaults.learning_starts == 1_000
    assert defaults.net_arch == (64, 64)
    assert defaults.sb3_kwargs()["policy_kwargs"] == {"net_arch": [64, 64]}


@pytest.mark.parametrize(
    "overrides",
    [
        {"learning_rate": 0.0},
        {"batch_size": 0},
        {"gamma": 1.5},
        {"net_arch": ()},
        {"net_arch": (0, 8)},
        {"exploration_initial_eps": 0.1, "exploration_final_eps": 0.2},
        {"tau": 0.5},  # not an exposed setting
    ],
)
def test_invalid_hyperparameters_are_rejected(overrides: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        DQNHyperparameters(**overrides)


def test_overrides_and_json_round_trip(tmp_path: Path) -> None:
    updated = apply_overrides(TINY, ["learning_rate=0.001", "net_arch=32,32", "gamma=0.95"])
    assert (updated.learning_rate, updated.net_arch, updated.gamma) == (0.001, (32, 32), 0.95)
    with pytest.raises(ValueError, match="invalid override"):
        apply_overrides(TINY, ["tau=1"])

    path = tmp_path / "hp.json"
    path.write_text(updated.model_dump_json())
    assert load_hyperparameters(path) == (updated, "file:hp.json")


# --- adapter ------------------------------------------------------------------------------


def test_adapter_is_a_controller_returning_python_ints(trained: tuple[DQN, Path]) -> None:
    controller = SB3Controller(trained[0])
    env = env_for()
    observation, info = env.reset(seed=0)

    action = controller.act(observation, info)

    assert isinstance(controller, Controller)
    assert type(action) is int and action in (0, 1, 2)


def test_adapter_predicts_deterministically() -> None:
    calls: list[dict[str, Any]] = []

    class Recorder:
        action_space = AutoscalingEnv(
            SimulatorConfig(), build_workload(load_benchmark_manifest().get(TRAIN))
        ).action_space

        def predict(self, observation: Any, **kwargs: Any) -> tuple[np.ndarray, None]:
            calls.append(kwargs)
            return np.array(2), None

    controller = SB3Controller(Recorder())  # type: ignore[arg-type]

    assert controller.act(np.zeros(12, dtype=np.float32), {}) == 2
    assert calls == [{"deterministic": True}]


def test_adapter_ignores_info(trained: tuple[DQN, Path]) -> None:
    controller = SB3Controller(trained[0])
    env = env_for()
    observation, info = env.reset(seed=0)
    poisoned = {
        **info,
        "request_rate": 1e9,
        "queued_requests": 1e9,
        "future_request_rates": [1e9] * 10,
    }

    actions = {controller.act(observation, variant) for variant in (info, {}, poisoned)}

    assert len(actions) == 1


def test_reset_never_changes_the_model(trained: tuple[DQN, Path]) -> None:
    model = trained[0]
    controller = SB3Controller(model)
    before = [p.detach().clone() for p in model.q_net.parameters()]

    controller.reset(seed=123)
    controller.reset()

    after = list(model.q_net.parameters())
    assert all(bool((a == b).all()) for a, b in zip(before, after, strict=True))


def test_adapter_uses_only_the_observation_through_the_standard_runner(
    trained: tuple[DQN, Path],
) -> None:
    env = env_for()
    controller = SB3Controller(trained[0])

    infos = run_episode(env, controller, seed=0)

    assert len(infos) == env.episode_ticks
    assert all(info["requested_action"] in (0, 1, 2) for info in infos)


# --- bundle and compatibility -----------------------------------------------------------------


def test_bundle_round_trip_evaluates_with_the_shared_metrics(trained: tuple[DQN, Path]) -> None:
    _, bundle = trained
    metadata, compatibility = read_model_bundle(bundle)
    env = env_for("syn-val-bursty")

    controller = load_sb3_controller(bundle, env)
    first = evaluate_controller_episode(env, controller)
    second = evaluate_controller_episode(env, controller)

    assert {path.name for path in bundle.iterdir()} == {
        "model.zip",
        "compatibility.json",
        "metadata.json",
    }
    assert (metadata.algorithm, metadata.training_workload_split) == ("dqn", "train")
    assert compatibility == EnvironmentCompatibility.from_env(env, "v1")
    assert isinstance(first.metrics, EpisodeMetrics)
    assert first.metrics == second.metrics  # deterministic evaluation


@pytest.mark.parametrize(
    ("field", "config"),
    [
        (
            "traffic_history_ticks",
            SimulatorConfig(observation=ObservationConfig(traffic_history_ticks=6)),
        ),
        ("startup_delay_seconds", SimulatorConfig(replicas=ReplicaConfig(startup_delay_seconds=0))),
        (
            "control_interval_seconds",
            SimulatorConfig(
                timing=TimingConfig(control_interval_seconds=60, episode_duration_seconds=3600)
            ),
        ),
        ("max_replicas", SimulatorConfig(replicas=ReplicaConfig(max_replicas=12))),
        ("service_capacity_rps", SimulatorConfig(replicas=ReplicaConfig(service_capacity_rps=25))),
        ("latency_target_seconds", SimulatorConfig(sla=SlaConfig(latency_target_seconds=1.0))),
    ],
)
def test_incompatible_environments_are_rejected_before_loading(
    trained: tuple[DQN, Path], field: str, config: SimulatorConfig
) -> None:
    entry = load_benchmark_manifest().get("syn-val-bursty")
    trace = build_workload(entry)
    if config.timing.control_interval_seconds != trace.control_interval_seconds:
        trace = trace.__class__(trace.request_rates[::2], config.timing.control_interval_seconds)
    env = AutoscalingEnv(config, trace)

    with pytest.raises(ValueError, match=f"incompatible environment: .*{field}"):
        load_sb3_controller(trained[1], env)


def test_same_shape_different_meaning_is_still_rejected(trained: tuple[DQN, Path]) -> None:
    # Same observation shape as training, but capacity (a normalizer) differs.
    env = env_for(config=SimulatorConfig(replicas=ReplicaConfig(service_capacity_rps=10)))
    assert env.observation_space.shape == env_for().observation_space.shape

    with pytest.raises(ValueError, match="service_capacity_rps"):
        load_sb3_controller(trained[1], env)


def test_incomplete_bundles_are_rejected(trained: tuple[DQN, Path], tmp_path: Path) -> None:
    (tmp_path / "model.zip").write_bytes((trained[1] / "model.zip").read_bytes())

    with pytest.raises(FileNotFoundError, match="compatibility.json"):
        load_sb3_controller(tmp_path, env_for())


# --- guardrails -------------------------------------------------------------------------------


@pytest.fixture
def no_training(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("training started before the workload guardrail")

    monkeypatch.setattr(training, "build_dqn", refuse)
    monkeypatch.setattr(training, "build_workloads", refuse)


@pytest.mark.parametrize(
    ("workload", "message"),
    [
        ("syn-test-spike-hard", "held-out test"),
        ("azure-test-993600", "held-out test"),
        ("syn-val-bursty", "a validation workload"),
    ],
)
def test_training_workload_must_be_train(
    workload: str, message: str, no_training: None, tracking_uri: str
) -> None:
    with pytest.raises(ValueError, match=message):
        train_dqn(
            workload_id=workload, validation_workload_ids=VALIDATION, tracking_uri=tracking_uri
        )


@pytest.mark.parametrize(
    ("validation", "message"),
    [
        (("syn-test-seasonal-shifted",), "held-out test"),
        (("syn-val-bursty", "azure-test-1166400"), "held-out test"),
        (("syn-train-bursty",), "a train workload"),
        ((), "at least one validation"),
    ],
)
def test_validation_workloads_must_be_validation(
    validation: tuple[str, ...], message: str, no_training: None, tracking_uri: str
) -> None:
    with pytest.raises(ValueError, match=message):
        train_dqn(workload_id=TRAIN, validation_workload_ids=validation, tracking_uri=tracking_uri)


@pytest.mark.parametrize(("timesteps", "trained"), [(1, 4), (6, 8), (97, 100)])
def test_budgets_sb3_would_overshoot_are_refused_before_training(
    timesteps: int, trained: int, no_training: None, tracking_uri: str
) -> None:
    # SB3 finishes whole train_freq rollouts: the recorded budget must be exact.
    with pytest.raises(ValueError, match=f"multiple of train_freq \\(4\\).*{trained} timesteps"):
        train_dqn(
            workload_id=TRAIN,
            validation_workload_ids=VALIDATION,
            timesteps=timesteps,
            tracking_uri=tracking_uri,
        )


def test_budgets_follow_the_configured_train_freq(no_training: None, tracking_uri: str) -> None:
    with pytest.raises(ValueError, match="multiple of train_freq \\(8\\)"):
        train_dqn(
            workload_id=TRAIN,
            validation_workload_ids=VALIDATION,
            hyperparameters=TINY.model_copy(update={"train_freq": 8}),
            timesteps=100,
            tracking_uri=tracking_uri,
        )
    with pytest.raises(ValueError, match="at least 1"):
        train_dqn(workload_id=TRAIN, validation_workload_ids=VALIDATION, timesteps=0)


def test_default_validation_set_is_the_synthetic_validation_workloads() -> None:
    assert training.default_validation_workload_ids() == (
        "syn-val-steady-high",
        "syn-val-ramp-down",
        "syn-val-bursty",
    )


# --- tracked training ------------------------------------------------------------------------


def run_tiny_training(tracking_uri: str, **kwargs: Any) -> DQNTrainingResult:
    return train_dqn(
        workload_id=TRAIN,
        validation_workload_ids=VALIDATION,
        hyperparameters=TINY,
        timesteps=256,
        seed=0,
        tracking_uri=tracking_uri,
        log_interval=64,
        **kwargs,
    )


def test_training_run_is_tracked_with_model_and_validation_lineage(
    tracking_uri: str, tmp_path: Path
) -> None:
    result = run_tiny_training(tracking_uri)
    client = MlflowClient(tracking_uri)
    run = client.get_run(result.training_run_id)
    tags, params, metrics = run.data.tags, run.data.params, run.data.metrics

    assert run.info.status == "FINISHED"
    assert (tags["scalerl.run_kind"], tags["scalerl.controller"]) == ("train", "dqn")
    assert (tags["scalerl.workload_id"], tags["scalerl.workload_split"]) == (TRAIN, "train")
    assert params["training_steps"] == "256"
    assert params["seed"] == "0"
    assert params["hp.dqn_config_version"] == DQN_CONFIG_VERSION
    assert params["hp.learning_rate"] == "0.0005"
    assert params["hp.net_arch"] == "[16, 16]"
    assert params["compat.traffic_history_ticks"] == "4"
    assert metrics["training_timesteps"] == 256
    assert result.timesteps == 256  # requested == recorded == actually trained
    for name in ("train/loss", "train/exploration_rate", "train/n_updates", "train/episodes"):
        history = client.get_metric_history(result.training_run_id, name)
        assert history and all(point.step % 64 == 0 for point in history), name
    for key in ("sla_violation_rate", "normalized_cost", "queue_pressure", "churn_rate"):
        assert metrics[f"validation.{key}"] == pytest.approx(result.validation_metrics[key])

    artifacts = {item.path for item in client.list_artifacts(result.training_run_id, "model")}
    assert artifacts == {"model/model.zip", "model/compatibility.json", "model/metadata.json"}
    downloaded = Path(client.download_artifacts(result.training_run_id, "model", str(tmp_path)))
    metadata, _ = read_model_bundle(downloaded)
    assert metadata.training_run_id == result.training_run_id
    assert metadata.total_timesteps == int(params["training_steps"]) == result.timesteps
    # The downloaded bundle is directly usable through the standard harness.
    controller = load_sb3_controller(downloaded, env_for("syn-val-bursty"))
    assert evaluate_controller_episode(env_for("syn-val-bursty"), controller).metrics

    assert len(result.validation_run_ids) == 2
    for workload, run_id in zip(VALIDATION, result.validation_run_ids, strict=True):
        validation = client.get_run(run_id)
        assert validation.info.status == "FINISHED"
        assert validation.data.tags[MODEL_SOURCE_TAG] == result.training_run_id
        assert validation.data.tags["scalerl.run_kind"] == "evaluate"
        assert validation.data.tags["scalerl.workload_id"] == workload
        assert validation.data.tags["scalerl.workload_split"] == "validation"
        assert set(EpisodeMetrics.__dataclass_fields__) <= set(validation.data.metrics)


def test_result_json_round_trips_without_model_bytes(tracking_uri: str, tmp_path: Path) -> None:
    result = run_tiny_training(tracking_uri)
    path = result.save(tmp_path / "outputs" / "dqn.json")

    loaded = DQNTrainingResult.load(path)
    payload = json.loads(path.read_text())

    assert loaded == result
    assert payload["model_artifact_uri"] == f"runs:/{result.training_run_id}/model"
    assert payload["training_workload_id"] == TRAIN
    assert payload["compatibility"]["observation_shape"] == [12]
    assert path.stat().st_size < 20_000


def test_same_seed_reproduces_the_policy_on_this_machine(tracking_uri: str) -> None:
    # Same seed and config on the same platform/library versions; no cross-platform claim.
    first = run_tiny_training(tracking_uri)
    second = run_tiny_training(tracking_uri)

    assert first.validation_metrics == second.validation_metrics
    assert first.training_episodes == second.training_episodes


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
            "96",
            "--hyperparameters",
            str(hp_file),
            "--hp",
            "gamma=0.95",
            "--log-interval",
            "32",
            "--tracking-uri",
            tracking_uri,
            "--output",
            str(output),
        ]
    )

    result = DQNTrainingResult.load(output)
    assert exit_code == 0
    assert result.hyperparameter_source == "file:hp.json+overrides"
    assert result.hyperparameters.gamma == 0.95
    assert result.timesteps == 96


def test_training_modules_do_not_import_optional_services() -> None:
    code = (
        "import sys\n"
        "import scalerl, scalerl.rl, scalerl.training.dqn, scalerl.tuning.dqn\n"
        "assert not {'mlflow', 'optuna', 'streamlit'} & set(sys.modules)\n"
        "import scalerl.environment\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
    base = "import sys, scalerl\nassert 'stable_baselines3' not in sys.modules\n"
    subprocess.run([sys.executable, "-c", base], check=True)
