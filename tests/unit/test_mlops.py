"""Tests for MLflow run tracking and reproducible run metadata.

Every test uses a temporary SQLite store and artifact directory; no MLflow
server or network is involved.
"""

import json
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from mlflow import MlflowClient
from pydantic import ValidationError

import scalerl
from scalerl.benchmarks import build_workload, load_benchmark_manifest
from scalerl.environment import (
    AutoscalingEnv,
    ReplicaConfig,
    RewardWeights,
    SimulatorConfig,
    TimingConfig,
)
from scalerl.mlops import (
    INSTALL_HINT,
    EnvironmentCompatibility,
    RunSpec,
    software_metadata,
    start_tracked_run,
)

V1_TIMING = TimingConfig(control_interval_seconds=30.0, episode_duration_seconds=3600.0)


@pytest.fixture
def tracking_uri(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    """An isolated SQLite store; artifacts go to ``tmp_path/mlruns``."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.setenv("MLFLOW_DISABLE_AGENT_HINT", "1")
    yield f"sqlite:///{tmp_path / 'mlflow.db'}"


def make_spec(**overrides: Any) -> RunSpec:
    params: dict[str, Any] = {
        "run_kind": "train",
        "controller": "threshold",
        "workload_id": "syn-train-spike",
        "workload_split": "train",
        "simulator_config": SimulatorConfig(),
        "simulator_config_source": "default",
        "seed": 42,
    }
    return RunSpec(**{**params, **overrides})


def calibrated_config() -> SimulatorConfig:
    return SimulatorConfig(
        timing=V1_TIMING,
        replicas=ReplicaConfig(max_replicas=8, service_capacity_rps=2.0, startup_delay_seconds=60),
    )


def download_json(client: MlflowClient, run_id: str, path: str, tmp_path: Path) -> Any:
    local = client.download_artifacts(run_id, path, str(tmp_path / "downloads"))
    return json.loads(Path(local).read_text())


# --- run spec: leakage and provenance ----------------------------------------


@pytest.mark.parametrize(
    ("run_kind", "workload_id", "split"),
    [
        ("train", "syn-train-spike", "train"),
        ("tune", "syn-val-bursty", "validation"),
        ("evaluate", "azure-test-993600", "test"),
        ("evaluate", "syn-val-steady-high", "validation"),
    ],
)
def test_allowed_run_kind_and_split_combinations(
    run_kind: str, workload_id: str, split: str
) -> None:
    spec = make_spec(run_kind=run_kind, workload_id=workload_id, workload_split=split)

    assert (spec.run_kind, spec.workload_split) == (run_kind, split)


@pytest.mark.parametrize("run_kind", ["train", "tune"])
@pytest.mark.parametrize("workload_id", ["azure-test-993600", "syn-test-spike-hard"])
def test_train_and_tune_runs_cannot_target_test_workloads(run_kind: str, workload_id: str) -> None:
    with pytest.raises(ValidationError, match="may not use held-out test workload"):
        make_spec(run_kind=run_kind, workload_id=workload_id, workload_split="test")


def test_declared_split_must_match_the_benchmark() -> None:
    # Mislabeling a test workload as validation cannot bypass the guardrail.
    with pytest.raises(ValidationError, match="is in split 'test', not 'validation'"):
        make_spec(run_kind="tune", workload_id="azure-test-993600", workload_split="validation")


def test_unknown_workload_is_rejected() -> None:
    with pytest.raises(ValidationError, match="is not in benchmark v1"):
        make_spec(workload_id="syn-train-imaginary")


def test_train_validation_calibration_is_accepted() -> None:
    spec = make_spec(
        simulator_config=calibrated_config(),
        simulator_config_source="calibrated_train_validation",
        calibration_workload_ids=("azure-train-129600", "azure-val-734400"),
        calibration_note="capacity sized from train/validation Azure peaks",
    )

    assert spec.calibration_workload_ids == ("azure-train-129600", "azure-val-734400")


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (
            {"calibration_workload_ids": ("azure-train-129600", "azure-test-993600")},
            "'azure-test-993600' is a held-out test workload",
        ),
        ({"calibration_workload_ids": ("azure-train-9",)}, "is not in the benchmark"),
        ({"calibration_workload_ids": ()}, "must list calibration_workload_ids"),
    ],
)
def test_calibration_provenance_is_validated(overrides: dict[str, Any], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        make_spec(
            simulator_config=calibrated_config(),
            simulator_config_source="calibrated_train_validation",
            **overrides,
        )


def test_calibration_ids_require_calibrated_source() -> None:
    with pytest.raises(ValidationError, match="require simulator_config_source"):
        make_spec(
            simulator_config=calibrated_config(),
            simulator_config_source="predeclared",
            calibration_workload_ids=("azure-train-129600",),
        )


def test_default_source_requires_the_default_config() -> None:
    with pytest.raises(ValidationError, match="requires the default SimulatorConfig"):
        make_spec(simulator_config=calibrated_config(), simulator_config_source="default")


@pytest.mark.parametrize(
    "overrides",
    [
        {"run_kind": "deploy"},
        {"controller": ""},
        {"benchmark_version": "v2"},
        {"hyperparameters": {"lr": float("nan")}},
        {"training_steps": -1},
        {"unexpected": 1},
    ],
)
def test_invalid_run_specs_are_rejected(overrides: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        make_spec(**overrides)


# --- compatibility and software metadata -------------------------------------


def test_compatibility_comes_from_the_real_environment() -> None:
    config = calibrated_config()
    trace = build_workload(load_benchmark_manifest().get("syn-train-spike"))

    from_env = EnvironmentCompatibility.from_env(AutoscalingEnv(config, trace), "v1")

    assert from_env == EnvironmentCompatibility.from_config(config, "v1")
    assert from_env.observation_shape == (12,)  # 4 history + 6 state + ceil(60 / 30)
    assert from_env.traffic_history_ticks == 4
    assert from_env.action_count == 3
    assert (from_env.max_replicas, from_env.service_capacity_rps) == (8, 2.0)


def test_incompatible_environments_are_reported() -> None:
    trained = EnvironmentCompatibility.from_config(SimulatorConfig(timing=V1_TIMING))
    current = EnvironmentCompatibility.from_config(
        SimulatorConfig(timing=V1_TIMING, replicas=ReplicaConfig(startup_delay_seconds=90.0))
    )

    assert trained.mismatches(trained) == {}
    trained.require_compatible(trained)
    assert trained.mismatches(current) == {
        "observation_shape": ((12,), (13,)),
        "startup_delay_seconds": (60.0, 90.0),
    }
    with pytest.raises(ValueError, match=r"observation_shape: \(12,\) != \(13,\)"):
        trained.require_compatible(current)


# Fields AutoscalingEnv does not use to build observations; everything else in
# SimulatorConfig must be part of the compatibility contract.
NON_OBSERVATION_FIELDS = {"min_replicas", "initial_replicas"}

# A valid alternative value for every observation-defining field.
OBSERVATION_FIELD_CHANGES: dict[str, dict[str, Any]] = {
    "control_interval_seconds": {"timing": {"control_interval_seconds": 60.0}},
    "episode_duration_seconds": {"timing": {"episode_duration_seconds": 7200.0}},
    "startup_delay_seconds": {"replicas": {"startup_delay_seconds": 30.0}},
    "max_replicas": {"replicas": {"max_replicas": 12}},
    "service_capacity_rps": {"replicas": {"service_capacity_rps": 2.0}},
    "cost_per_hour": {"replicas": {"cost_per_hour": 0.0}},
    "latency_target_seconds": {"sla": {"latency_target_seconds": 0.25}},
    "traffic_history_ticks": {"observation": {"traffic_history_ticks": 1}},
}


def config_with(changes: dict[str, dict[str, Any]]) -> SimulatorConfig:
    base = SimulatorConfig(timing=V1_TIMING).model_dump()
    for section, values in changes.items():
        base[section] = {**base[section], **values}
    return SimulatorConfig.model_validate(base)


def test_every_simulator_field_is_classified_for_compatibility() -> None:
    config_fields = {
        field for section in SimulatorConfig().model_dump().values() for field in section
    }
    contract_fields = set(EnvironmentCompatibility.model_fields)

    # A new SimulatorConfig field must be added to the contract or deliberately excluded.
    assert config_fields - NON_OBSERVATION_FIELDS <= contract_fields
    assert config_fields - NON_OBSERVATION_FIELDS == set(OBSERVATION_FIELD_CHANGES)


@pytest.mark.parametrize("field", sorted(OBSERVATION_FIELD_CHANGES))
def test_changing_an_observation_field_breaks_compatibility(field: str) -> None:
    trained = EnvironmentCompatibility.from_config(config_with({}))
    current = EnvironmentCompatibility.from_config(config_with(OBSERVATION_FIELD_CHANGES[field]))

    assert field in trained.mismatches(current)
    with pytest.raises(ValueError, match="incompatible environment"):
        trained.require_compatible(current)


@pytest.mark.parametrize(
    "changes",
    [
        {"sla": {"latency_target_seconds": 0.25}},
        {"timing": {"episode_duration_seconds": 7200.0}},
    ],
    ids=["latency_target_only", "episode_duration_only"],
)
def test_same_shape_but_different_feature_semantics_is_incompatible(
    changes: dict[str, dict[str, Any]],
) -> None:
    trained = EnvironmentCompatibility.from_config(config_with({}))
    current = EnvironmentCompatibility.from_config(config_with(changes))

    assert trained.observation_shape == current.observation_shape
    with pytest.raises(ValueError, match="incompatible environment"):
        trained.require_compatible(current)


def test_start_state_fields_do_not_affect_compatibility() -> None:
    trained = EnvironmentCompatibility.from_config(config_with({}))
    current = EnvironmentCompatibility.from_config(
        config_with({"replicas": {"min_replicas": 2, "initial_replicas": 3}})
    )

    trained.require_compatible(current)


def test_git_sha_prefers_override_then_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GITHUB_SHA", "from-ci")
    assert software_metadata("explicit")["git_sha"] == "explicit"
    assert software_metadata()["git_sha"] == "from-ci"


def test_git_sha_falls_back_to_unknown_without_git(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GITHUB_SHA", raising=False)
    monkeypatch.delenv("SCALERL_GIT_SHA", raising=False)
    monkeypatch.setenv("PATH", "")  # git is unavailable, as in some wheel installs

    metadata = software_metadata()

    assert (metadata["git_sha"], metadata["git_dirty"]) == ("unknown", None)
    assert metadata["scalerl_version"] == scalerl.__version__
    assert metadata["packages"]["mlflow"] != "not_installed"


def test_git_sha_is_read_from_the_checkout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GITHUB_SHA", raising=False)
    monkeypatch.delenv("SCALERL_GIT_SHA", raising=False)
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(scalerl.__file__).parent,
        capture_output=True,
        text=True,
        check=False,
    )
    if head.returncode != 0:
        pytest.skip("not running from a Git checkout")

    metadata = software_metadata()

    assert metadata["git_sha"] == head.stdout.strip()
    assert isinstance(metadata["git_dirty"], bool)


# --- tracked run lifecycle ---------------------------------------------------


def test_successful_run_is_finished(tracking_uri: str) -> None:
    with start_tracked_run(make_spec(), tracking_uri=tracking_uri) as run:
        run_id = run.run_id

    tracked = MlflowClient(tracking_uri).get_run(run_id)
    assert tracked.info.status == "FINISHED"
    assert tracked.info.run_name == "train-threshold-syn-train-spike"


def test_failing_run_is_marked_failed_and_the_error_propagates(tracking_uri: str) -> None:
    with pytest.raises(RuntimeError, match="training diverged"):
        with start_tracked_run(make_spec(), tracking_uri=tracking_uri) as run:
            run_id = run.run_id
            raise RuntimeError("training diverged")

    assert MlflowClient(tracking_uri).get_run(run_id).info.status == "FAILED"


def test_interrupted_run_is_marked_killed(tracking_uri: str) -> None:
    with pytest.raises(KeyboardInterrupt):
        with start_tracked_run(make_spec(), tracking_uri=tracking_uri) as run:
            run_id = run.run_id
            raise KeyboardInterrupt

    assert MlflowClient(tracking_uri).get_run(run_id).info.status == "KILLED"


def test_runs_use_the_requested_experiment(tracking_uri: str) -> None:
    with start_tracked_run(make_spec(), tracking_uri=tracking_uri, experiment_name="azure-v1"):
        pass
    with start_tracked_run(make_spec(), tracking_uri=tracking_uri, experiment_name="azure-v1"):
        pass

    client = MlflowClient(tracking_uri)
    experiment = client.get_experiment_by_name("azure-v1")
    assert experiment is not None
    assert len(client.search_runs([experiment.experiment_id])) == 2


def test_racing_workers_join_the_same_experiment(
    tracking_uri: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    with start_tracked_run(make_spec(), tracking_uri=tracking_uri, experiment_name="race"):
        pass
    real_lookup = MlflowClient.get_experiment_by_name
    lookups = []

    def stale_first_lookup(self: MlflowClient, name: str) -> Any:
        # The first lookup misses, as if another worker created it just after.
        lookups.append(name)
        return None if len(lookups) == 1 else real_lookup(self, name)

    monkeypatch.setattr(MlflowClient, "get_experiment_by_name", stale_first_lookup)
    with start_tracked_run(make_spec(), tracking_uri=tracking_uri, experiment_name="race") as run:
        run_id = run.run_id
    monkeypatch.undo()

    client = MlflowClient(tracking_uri)
    experiment = client.get_experiment_by_name("race")
    assert experiment is not None
    assert client.get_run(run_id).info.experiment_id == experiment.experiment_id
    assert client.get_run(run_id).info.status == "FINISHED"
    assert len(client.search_experiments(filter_string="name = 'race'")) == 1


def test_other_experiment_creation_errors_propagate(
    tracking_uri: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from mlflow.exceptions import MlflowException

    def refuse(self: MlflowClient, name: str, *args: Any, **kwargs: Any) -> str:
        raise MlflowException("backend unavailable", error_code="INTERNAL_ERROR")

    monkeypatch.setattr(MlflowClient, "create_experiment", refuse)

    with pytest.raises(MlflowException, match="backend unavailable"):
        with start_tracked_run(make_spec(), tracking_uri=tracking_uri, experiment_name="new"):
            pass


def test_tracking_uri_comes_from_the_environment(
    tracking_uri: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking_uri)

    with start_tracked_run(make_spec()) as run:
        run_id = run.run_id

    assert MlflowClient(tracking_uri).get_run(run_id).info.status == "FINISHED"


def test_explicit_tracking_uri_overrides_the_environment(
    tracking_uri: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    other = f"sqlite:///{tmp_path / 'other.db'}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", other)

    with start_tracked_run(make_spec(), tracking_uri=tracking_uri) as run:
        run_id = run.run_id

    assert MlflowClient(tracking_uri).get_run(run_id).info.status == "FINISHED"
    assert MlflowClient(other).search_experiments(filter_string="name = 'scalerl'") == []


# --- logged metadata ---------------------------------------------------------


def calibrated_spec() -> RunSpec:
    return make_spec(
        run_kind="tune",
        controller="threshold",
        workload_id="azure-val-734400",
        workload_split="validation",
        simulator_config=calibrated_config(),
        simulator_config_source="calibrated_train_validation",
        calibration_workload_ids=("azure-train-129600", "azure-val-734400"),
        calibration_note="capacity sized from train/validation Azure peaks",
        reward_weights=RewardWeights(latency=2.0, churn=0.3),
        seed=7,
        evaluation_seeds=(100, 101),
        hyperparameters={"low_threshold": 0.3, "high_threshold": 0.8, "grid": {"cooldown": [0, 2]}},
        training_episodes=5,
    )


def test_run_records_lineage_params_and_tags(tracking_uri: str) -> None:
    with start_tracked_run(calibrated_spec(), tracking_uri=tracking_uri, git_sha="abc123") as run:
        run_id = run.run_id

    data = MlflowClient(tracking_uri).get_run(run_id).data
    assert data.tags["scalerl.run_kind"] == "tune"
    assert data.tags["scalerl.controller"] == "threshold"
    assert data.tags["scalerl.workload_id"] == "azure-val-734400"
    assert data.tags["scalerl.workload_split"] == "validation"
    assert data.tags["scalerl.benchmark_version"] == "v1"
    assert data.tags["scalerl.simulator_config_source"] == "calibrated_train_validation"
    assert data.tags["scalerl.git_sha"] == data.tags["mlflow.source.git.commit"] == "abc123"
    assert data.tags["scalerl.version"] == scalerl.__version__

    params = data.params
    assert params["seed"] == "7"
    assert params["evaluation_seeds"] == "[100, 101]"
    assert params["training_episodes"] == "5"
    assert params["calibration_workload_ids"] == '["azure-train-129600", "azure-val-734400"]'
    assert params["calibration_note"] == "capacity sized from train/validation Azure peaks"
    assert params["sim.timing.control_interval_seconds"] == "30.0"
    assert params["sim.timing.episode_duration_seconds"] == "3600.0"
    assert params["sim.replicas.min_replicas"] == "1"
    assert params["sim.replicas.initial_replicas"] == "1"
    assert params["sim.replicas.max_replicas"] == "8"
    assert params["sim.replicas.startup_delay_seconds"] == "60.0"
    assert params["sim.replicas.service_capacity_rps"] == "2.0"
    assert params["sim.replicas.cost_per_hour"] == "0.1"
    assert params["sim.sla.latency_target_seconds"] == "0.5"
    assert params["reward.latency"] == "2.0"
    assert params["reward.churn"] == "0.3"
    assert params["hp.low_threshold"] == "0.3"
    assert params["hp.grid.cooldown"] == "[0, 2]"
    assert params["compat.observation_shape"] == "[12]"
    assert params["compat.traffic_history_ticks"] == "4"
    assert params["sim.observation.traffic_history_ticks"] == "4"
    assert params["compat.action_count"] == "3"
    # Values the caller did not provide are not logged.
    assert "training_steps" not in params


def test_run_logs_exact_resolved_artifacts(tracking_uri: str, tmp_path: Path) -> None:
    spec = calibrated_spec()
    with start_tracked_run(spec, tracking_uri=tracking_uri, git_sha="abc123") as run:
        run_id = run.run_id
    client = MlflowClient(tracking_uri)

    resolved = download_json(client, run_id, "scalerl/resolved_run.json", tmp_path)
    assert resolved["run_id"] == run_id
    assert RunSpec.model_validate_json(json.dumps(resolved["run_spec"])) == spec
    assert resolved["software"]["git_sha"] == "abc123"

    config = download_json(client, run_id, "scalerl/simulator_config.json", tmp_path)
    assert SimulatorConfig.model_validate_json(json.dumps(config)) == spec.simulator_config

    weights = download_json(client, run_id, "scalerl/reward_weights.json", tmp_path)
    assert RewardWeights.model_validate_json(json.dumps(weights)) == spec.reward_weights

    compatibility = download_json(client, run_id, "scalerl/compatibility.json", tmp_path)
    assert EnvironmentCompatibility.model_validate_json(
        json.dumps(compatibility)
    ) == EnvironmentCompatibility.from_config(spec.simulator_config, "v1")

    software = download_json(client, run_id, "scalerl/software.json", tmp_path)
    assert software["python_version"] == sys.version.split()[0]
    assert {"mlflow", "gymnasium", "numpy", "pydantic", "stable-baselines3", "torch"} <= set(
        software["packages"]
    )


# --- metrics and caller artifacts --------------------------------------------


def test_scalar_and_stepped_metrics_are_retrievable(tracking_uri: str) -> None:
    with start_tracked_run(make_spec(), tracking_uri=tracking_uri) as run:
        run.log_metrics({"episode_reward": -12.5, "sla_violation_rate": 0.25})
        for step, loss in enumerate([0.9, 0.5, 0.2]):
            run.log_metric("train_loss", loss, step=step)
        run_id = run.run_id

    client = MlflowClient(tracking_uri)
    metrics = client.get_run(run_id).data.metrics
    assert metrics == {"episode_reward": -12.5, "sla_violation_rate": 0.25, "train_loss": 0.2}
    history = client.get_metric_history(run_id, "train_loss")
    assert [(m.step, m.value) for m in history] == [(0, 0.9), (1, 0.5), (2, 0.2)]


def test_nothing_is_logged_that_the_caller_did_not_compute(tracking_uri: str) -> None:
    with start_tracked_run(make_spec(), tracking_uri=tracking_uri) as run:
        run_id = run.run_id

    assert MlflowClient(tracking_uri).get_run(run_id).data.metrics == {}


@pytest.mark.parametrize(
    ("value", "error"),
    [(float("nan"), ValueError), (float("inf"), ValueError), (None, TypeError), (True, TypeError)],
)
def test_invalid_metric_values_are_rejected(
    tracking_uri: str, value: object, error: type[Exception]
) -> None:
    with start_tracked_run(make_spec(), tracking_uri=tracking_uri) as run:
        with pytest.raises(error):
            run.log_metric("infrastructure_cost", value)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="step must be a non-negative integer"):
            run.log_metric("infrastructure_cost", 1.0, step=-1)


def test_caller_tags_are_logged(tracking_uri: str) -> None:
    with start_tracked_run(make_spec(), tracking_uri=tracking_uri) as run:
        run.set_tag("scalerl.optuna.study", "threshold-v1")
        with pytest.raises(TypeError, match="must be a string"):
            run.set_tag("scalerl.optuna.trial", 7)  # type: ignore[arg-type]
        run_id = run.run_id

    tags = MlflowClient(tracking_uri).get_run(run_id).data.tags
    assert tags["scalerl.optuna.study"] == "threshold-v1"
    assert "scalerl.optuna.trial" not in tags


def test_caller_artifacts_are_logged(tracking_uri: str, tmp_path: Path) -> None:
    evaluation = tmp_path / "evaluation.csv"
    evaluation.write_text("tick,reward\n0,-1.0\n")

    with start_tracked_run(make_spec(), tracking_uri=tracking_uri) as run:
        run.log_artifact_dict("evaluation.json", {"mean_reward": -1.0})
        run.log_artifact(evaluation, artifact_path="evaluation")
        with pytest.raises(ValueError, match="plain .json file name"):
            run.log_artifact_dict("../escape.json", {})
        with pytest.raises(FileNotFoundError):
            run.log_artifact(tmp_path / "missing.pt")
        run_id = run.run_id

    client = MlflowClient(tracking_uri)
    assert download_json(client, run_id, "evaluation.json", tmp_path) == {"mean_reward": -1.0}
    assert [a.path for a in client.list_artifacts(run_id, "evaluation")] == [
        "evaluation/evaluation.csv"
    ]


# --- optional dependency -----------------------------------------------------


def test_core_packages_do_not_import_mlflow() -> None:
    code = (
        "import sys\n"
        "import scalerl, scalerl.environment, scalerl.controllers, scalerl.workloads\n"
        "import scalerl.benchmarks, scalerl.mlops\n"
        "assert 'mlflow' not in sys.modules, 'core import pulled in mlflow'\n"
    )

    subprocess.run([sys.executable, "-c", code], check=True)


def test_tracking_without_mlflow_gives_an_install_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "mlflow", None)  # simulate a missing optional extra

    with pytest.raises(ImportError, match=r'pip install -e "\.\[mlops\]"'):
        with start_tracked_run(make_spec()):
            pass
    assert "mlops" in INSTALL_HINT
