"""Tests for threshold tuning (#13) on the shared Optuna + MLflow infrastructure."""

import itertools
import os
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import optuna
import pytest
from mlflow import MlflowClient

import scalerl.tuning.threshold as threshold_tuning
from scalerl.benchmarks import AzureWorkload, load_benchmark_manifest
from scalerl.controllers import ThresholdController
from scalerl.environment import AutoscalingEnv, ReplicaConfig, SimulatorConfig, TimingConfig
from scalerl.tuning import RUN_IDS_ATTR, StudySpec
from scalerl.tuning.threshold import (
    SELECTED_TRIAL_ATTR,
    SELECTION_KEYS,
    THRESHOLD_GRID,
    ThresholdTuningResult,
    TrialAggregate,
    default_tuning_workload_ids,
    evaluate_episode,
    run_threshold_study,
    select_trial,
    threshold_study_spec,
)
from scalerl.workloads import WorkloadTrace

AZURE_FIXTURE = Path(__file__).parent.parent / "fixtures" / "azure_functions_2021_small.csv"
WORKLOADS = ("syn-train-spike", "syn-val-bursty")
EXPECTED_GRID = set(itertools.product([0.6, 0.7, 0.8], [0.2, 0.3], [3, 5, 10]))


@dataclass
class StudyRun:
    result: ThresholdTuningResult
    storage: str
    tracking_uri: str
    specs: list[StudySpec]
    directory: Path

    def study(self) -> optuna.Study:
        return optuna.load_study(study_name=self.result.study_name, storage=self.storage)


@pytest.fixture(scope="module")
def study_run(tmp_path_factory: pytest.TempPathFactory) -> Iterator[StudyRun]:
    """One real grid study over a train and a validation workload, shared by tests."""
    directory = tmp_path_factory.mktemp("threshold-study")
    previous = Path.cwd()
    os.chdir(directory)  # MLflow's SQLite store writes artifacts under ./mlruns
    os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
    specs: list[StudySpec] = []
    real_run_study = threshold_tuning.run_study

    def spying_run_study(spec: StudySpec, objective: Any, **kwargs: Any) -> Any:
        specs.append(spec)
        return real_run_study(spec, objective, **kwargs)

    patch = pytest.MonkeyPatch()
    patch.setattr(threshold_tuning, "run_study", spying_run_study)
    storage = f"sqlite:///{directory / 'optuna.db'}"
    tracking_uri = f"sqlite:///{directory / 'mlflow.db'}"
    try:
        result = run_threshold_study(
            workload_ids=WORKLOADS, storage=storage, tracking_uri=tracking_uri
        )
        yield StudyRun(result, storage, tracking_uri, specs, directory)
    finally:
        patch.undo()
        os.chdir(previous)


def params_tuple(params: dict[str, Any]) -> tuple[Any, ...]:
    return (params["high_threshold"], params["low_threshold"], params["cooldown_ticks"])


# --- search space and workloads --------------------------------------------------------


def test_grid_is_the_exact_eighteen_configurations() -> None:
    spec = threshold_study_spec(WORKLOADS)

    assert spec.sampler == "grid"
    assert spec.grid == THRESHOLD_GRID
    assert spec.grid_size == 18
    assert set(itertools.product(*THRESHOLD_GRID.values())) == EXPECTED_GRID


def test_default_workloads_are_the_synthetic_train_and_validation_suite() -> None:
    manifest = load_benchmark_manifest()
    expected = tuple(entry.id for entry in manifest.tuning if not isinstance(entry, AzureWorkload))

    ids = default_tuning_workload_ids()

    assert ids == expected
    splits = {manifest.get(workload_id).split for workload_id in ids}
    assert splits == {"train", "validation"}


@pytest.mark.parametrize(
    "held_out", ["syn-test-spike-hard", "azure-test-993600", "azure-test-1166400"]
)
def test_held_out_workloads_are_rejected(held_out: str) -> None:
    with pytest.raises(ValueError, match="held-out test workload"):
        run_threshold_study(workload_ids=("syn-train-spike", held_out))


def test_selection_needs_a_validation_workload() -> None:
    with pytest.raises(ValueError, match="at least one validation workload"):
        run_threshold_study(workload_ids=("syn-train-spike", "syn-train-bursty"))


def test_azure_workloads_need_the_local_trace() -> None:
    with pytest.raises(ValueError, match="pass azure_csv_path"):
        run_threshold_study(workload_ids=("syn-train-spike", "azure-val-734400"))


def test_explicit_azure_validation_workload_uses_the_local_trace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MLFLOW_DISABLE_AGENT_HINT", "1")

    result = run_threshold_study(
        workload_ids=("syn-train-spike", "azure-val-734400"),
        azure_csv_path=AZURE_FIXTURE,
        tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}",
    )

    assert result.validation_workload_ids == ("azure-val-734400",)
    assert result.trial_count == 18
    assert result.optuna_storage == "in-memory"


# --- the study ------------------------------------------------------------------------


def test_study_runs_through_the_shared_grid_infrastructure(study_run: StudyRun) -> None:
    assert len(study_run.specs) == 1
    spec = study_run.specs[0]
    assert (spec.sampler, spec.grid, spec.direction) == ("grid", THRESHOLD_GRID, "minimize")
    assert spec.objective_name == "threshold-sla-first"
    assert spec.tuning_workload_ids == WORKLOADS


def test_every_configuration_is_evaluated_exactly_once(study_run: StudyRun) -> None:
    trials = study_run.study().trials

    combinations = [params_tuple(trial.params) for trial in trials]
    assert len(combinations) == 18
    assert set(combinations) == EXPECTED_GRID
    assert all(trial.state == optuna.trial.TrialState.COMPLETE for trial in trials)


def test_trials_record_validation_aggregates(study_run: StudyRun) -> None:
    for trial in study_run.study().trials:
        for key in SELECTION_KEYS:
            assert 0.0 <= trial.user_attrs[key] <= 1.0
        assert trial.value == trial.user_attrs["validation_sla_violation_rate"]
        assert "train_sla_violation_rate" in trial.user_attrs
        assert "validation_episode_reward" in trial.user_attrs


def test_each_workload_evaluation_is_its_own_mlflow_run(study_run: StudyRun) -> None:
    client = MlflowClient(study_run.tracking_uri)

    for trial in study_run.study().trials:
        run_ids = trial.user_attrs[RUN_IDS_ATTR]
        assert len(run_ids) == len(WORKLOADS)
        for run_id, workload_id in zip(run_ids, WORKLOADS, strict=True):
            run = client.get_run(run_id)
            params, tags, metrics = run.data.params, run.data.tags, run.data.metrics
            assert run.info.status == "FINISHED"
            assert tags["scalerl.workload_id"] == workload_id
            assert tags["scalerl.workload_split"] == workload_id.split("-")[1].replace(
                "val", "validation"
            )
            assert tags["scalerl.run_kind"] == "tune"
            assert tags["scalerl.controller"] == "threshold"
            assert tags["scalerl.optuna.trial_state"] == "COMPLETE"
            assert params["hp.low_threshold"] == str(trial.params["low_threshold"])
            assert params["hp.high_threshold"] == str(trial.params["high_threshold"])
            assert params["hp.cooldown_ticks"] == str(trial.params["cooldown_ticks"])
            assert params["hp.optuna.study"] == "threshold-v1"
            assert params["hp.optuna.trial"] == str(trial.number)
            for metric in (
                "infrastructure_cost",
                "normalized_cost",
                "sla_violation_rate",
                "mean_p95_latency_seconds",
                "max_p95_latency_seconds",
                "mean_queue_depth",
                "max_queue_depth",
                "queue_pressure",
                "scaling_actions",
                "churn_rate",
                "episode_reward",
            ):
                assert metric in metrics
            assert 0.0 <= metrics["normalized_cost"] <= 1.0
            assert 0.0 <= metrics["queue_pressure"] <= 1.0


def test_result_records_the_selected_configuration(study_run: StudyRun) -> None:
    study = study_run.study()
    result = study_run.result
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    expected = select_trial(
        [
            TrialAggregate(t.number, t.params, *(t.user_attrs[key] for key in SELECTION_KEYS))
            for t in completed
        ]
    )

    assert result.selected_trial_number == expected.number
    assert result.selected_params == dict(expected.params)
    assert study.user_attrs[SELECTED_TRIAL_ATTR] == result.selected_trial_number
    assert study.user_attrs["selected_params"] == result.selected_params
    assert study.user_attrs["selection_rule"] == "threshold-sla-first-v1"
    assert result.trial_count == 18
    assert result.validation_workload_ids == ("syn-val-bursty",)
    assert result.benchmark_version == "v1"
    assert result.optuna_storage == study_run.storage
    selected = next(t for t in completed if t.number == result.selected_trial_number)
    assert result.selected_mlflow_run_ids == tuple(selected.user_attrs[RUN_IDS_ATTR])
    assert set(result.selected_validation_metrics) >= set(SELECTION_KEYS)


def test_result_json_round_trips(study_run: StudyRun, tmp_path: Path) -> None:
    path = study_run.result.save(tmp_path / "nested" / "threshold-v1.json")

    assert ThresholdTuningResult.load(path) == study_run.result


def test_rerunning_the_study_adds_no_duplicate_configurations(study_run: StudyRun) -> None:
    previous = Path.cwd()
    os.chdir(study_run.directory)
    try:
        again = run_threshold_study(
            workload_ids=WORKLOADS,
            storage=study_run.storage,
            tracking_uri=study_run.tracking_uri,
        )
    finally:
        os.chdir(previous)

    combinations = [params_tuple(t.params) for t in study_run.study().trials]
    assert len(combinations) == len(set(combinations)) == 18
    assert again.selected_trial_number == study_run.result.selected_trial_number


# --- selection rule -------------------------------------------------------------------


def aggregate(
    number: int,
    sla: float,
    cost: float,
    queue: float = 0.1,
    churn: float = 0.1,
    params: tuple[float, float, int] = (0.8, 0.2, 5),
) -> TrialAggregate:
    high, low, cooldown = params
    return TrialAggregate(
        number,
        {"high_threshold": high, "low_threshold": low, "cooldown_ticks": cooldown},
        validation_sla_violation_rate=sla,
        validation_normalized_cost=cost,
        validation_queue_pressure=queue,
        validation_churn_rate=churn,
    )


def test_sla_is_the_primary_criterion() -> None:
    a, b = aggregate(0, sla=0.02, cost=0.8), aggregate(1, sla=0.01, cost=0.9)

    assert select_trial([a, b]) is b


def test_cost_breaks_sla_ties() -> None:
    b, c = aggregate(1, sla=0.01, cost=0.9), aggregate(2, sla=0.01, cost=0.5)

    assert select_trial([b, c]) is c


def test_queue_then_churn_break_remaining_ties() -> None:
    busier = aggregate(0, sla=0.01, cost=0.5, queue=0.3)
    calmer = aggregate(1, sla=0.01, cost=0.5, queue=0.1, churn=0.4)
    steadier = aggregate(2, sla=0.01, cost=0.5, queue=0.1, churn=0.1)

    assert select_trial([busier, calmer]) is calmer
    assert select_trial([busier, calmer, steadier]) is steadier


def test_parameters_break_exact_ties_deterministically() -> None:
    tied = [
        aggregate(0, 0.01, 0.5, params=(0.8, 0.2, 3)),
        aggregate(1, 0.01, 0.5, params=(0.6, 0.3, 10)),
        aggregate(2, 0.01, 0.5, params=(0.6, 0.2, 5)),
    ]

    assert select_trial(tied).number == 2  # smallest (high, low, cooldown)
    assert select_trial(list(reversed(tied))).number == 2


def test_selection_needs_completed_trials() -> None:
    with pytest.raises(ValueError, match="no completed trials"):
        select_trial([])


# --- episode metrics ------------------------------------------------------------------


def test_bound_clipped_requests_are_not_churn() -> None:
    config = SimulatorConfig(
        timing=TimingConfig(control_interval_seconds=30.0, episode_duration_seconds=600.0),
        replicas=ReplicaConfig(min_replicas=1, max_replicas=2, initial_replicas=1),
    )
    env = AutoscalingEnv(config, WorkloadTrace([500.0] * 20, control_interval_seconds=30.0))
    # The controller allows 6 replicas but the environment caps at 2.
    controller = ThresholdController(
        low_threshold=0.3, high_threshold=0.8, min_replicas=1, max_replicas=6
    )

    metrics = evaluate_episode(env, controller)

    assert metrics.scaling_actions == 1  # only the first request changed the fleet
    assert metrics.churn_rate == 1 / 20
    assert metrics.normalized_cost == pytest.approx(39 / 40)  # 1 + 19 x 2 of 20 x 2
    assert 0.0 < metrics.queue_pressure < 1.0
    assert metrics.sla_violation_rate > 0.9
