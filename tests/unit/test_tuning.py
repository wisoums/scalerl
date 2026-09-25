"""Tests for Optuna study infrastructure and its MLflow/benchmark integration.

Studies use in-memory or temporary SQLite Optuna storage; MLflow uses a
temporary SQLite store. Nothing contacts a network service.
"""

import itertools
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import optuna
import pytest
from mlflow import MlflowClient
from pydantic import ValidationError

from scalerl.benchmarks import build_workload, load_benchmark_manifest
from scalerl.controllers import StaticController, run_episode
from scalerl.environment import AutoscalingEnv, SimulatorConfig
from scalerl.mlops import RunSpec
from scalerl.tuning import (
    INSTALL_HINT,
    RUN_IDS_ATTR,
    StudySpec,
    TrialContext,
    require_tuning_workloads,
    run_study,
    safe_storage_label,
)

THRESHOLD_GRID: dict[str, list[Any]] = {
    "high_threshold": [0.6, 0.7, 0.8],
    "low_threshold": [0.2, 0.3],
    "cooldown_ticks": [3, 5, 10],
}


def make_spec(**overrides: Any) -> StudySpec:
    params: dict[str, Any] = {
        "name": "demo",
        "objective_name": "demo-quadratic",
        "objective_version": "v1",
        "search_space_version": "v1",
        "direction": "minimize",
        "tuning_workload_ids": ("syn-train-spike",),
        "sampler": "tpe",
        "sampler_seed": 42,
        "n_trials": 6,
    }
    return StudySpec(**{**params, **overrides})


def grid_spec(**overrides: Any) -> StudySpec:
    params: dict[str, Any] = {
        "name": "threshold-grid",
        "objective_name": "demo-grid",
        "sampler": "grid",
        "grid": THRESHOLD_GRID,
        "n_trials": None,
    }
    return make_spec(**{**params, **overrides})


def quadratic(context: TrialContext) -> float:
    x = context.trial.suggest_float("x", -10.0, 10.0)
    return (x - 2.0) ** 2


def grid_objective(context: TrialContext) -> float:
    params = {
        name: context.trial.suggest_categorical(name, values)
        for name, values in THRESHOLD_GRID.items()
    }
    return params["high_threshold"] - params["low_threshold"] + params["cooldown_ticks"] / 100


def trial_params(study: optuna.Study) -> list[dict[str, Any]]:
    return [trial.params for trial in study.trials]


def tune_spec(workload_id: str, split: str = "train", **overrides: Any) -> RunSpec:
    params: dict[str, Any] = {
        "run_kind": "tune",
        "controller": "static",
        "workload_id": workload_id,
        "workload_split": split,
        "simulator_config": SimulatorConfig(),
        "simulator_config_source": "default",
        "seed": 0,
    }
    return RunSpec(**{**params, **overrides})


@pytest.fixture
def tracking_uri(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.setenv("MLFLOW_DISABLE_AGENT_HINT", "1")
    yield f"sqlite:///{tmp_path / 'mlflow.db'}"


@pytest.fixture
def optuna_storage(tmp_path: Path) -> str:
    return f"sqlite:///{tmp_path / 'optuna.db'}"


# --- study spec and workload guardrail -------------------------------------------


def test_train_and_validation_workloads_are_accepted_for_tuning() -> None:
    entries = require_tuning_workloads(["syn-train-spike", "syn-val-bursty", "azure-val-734400"])

    assert [entry.split for entry in entries] == ["train", "validation", "validation"]
    assert set(entries) <= set(load_benchmark_manifest().tuning)


@pytest.mark.parametrize("workload_id", ["azure-test-993600", "syn-test-spike-hard"])
def test_held_out_workloads_are_rejected_for_tuning(workload_id: str) -> None:
    with pytest.raises(ValueError, match="held-out test workload"):
        require_tuning_workloads([workload_id])
    with pytest.raises(ValidationError, match="held-out test workload"):
        make_spec(tuning_workload_ids=("syn-train-spike", workload_id))


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"tuning_workload_ids": ("syn-train-imaginary",)}, "not in benchmark v1"),
        ({"tuning_workload_ids": ()}, "at least 1 item"),
        ({"tuning_workload_ids": ("syn-train-spike", "syn-train-spike")}, "must be unique"),
        ({"sampler": "grid"}, "needs a non-empty grid"),
        ({"sampler": "grid", "grid": {"x": []}}, "needs a non-empty grid"),
        ({"grid": {"x": [1, 2]}}, "only used by the grid sampler"),
        ({"n_trials": None}, "needs n_trials or timeout_seconds"),
        ({"direction": "sideways"}, "direction"),
        ({"sampler": "cmaes"}, "sampler"),
        ({"n_jobs": 0}, "n_jobs"),
        ({"unexpected": 1}, "Extra inputs are not permitted"),
    ],
)
def test_invalid_study_specs_are_rejected(overrides: dict[str, Any], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        make_spec(**overrides)


def test_grid_size_matches_the_threshold_grid() -> None:
    assert grid_spec().grid_size == 3 * 2 * 3 == 18
    assert make_spec().grid_size is None


@pytest.mark.parametrize(
    ("storage", "label"),
    [
        (None, "in-memory"),
        ("sqlite:///optuna.db", "sqlite:///optuna.db"),
        ("postgresql://user:secret@db.local:5432/optuna", "postgresql://db.local:5432/optuna"),
    ],
)
def test_storage_labels_never_include_credentials(storage: str | None, label: str) -> None:
    assert safe_storage_label(storage) == label


# --- optional dependency ------------------------------------------------------------


def test_core_and_tuning_specs_do_not_import_optuna() -> None:
    code = (
        "import sys\n"
        "import scalerl, scalerl.environment, scalerl.controllers, scalerl.mlops\n"
        "import scalerl.tuning\n"
        "from scalerl.tuning import StudySpec\n"
        "assert 'optuna' not in sys.modules, 'core import pulled in optuna'\n"
    )

    subprocess.run([sys.executable, "-c", code], check=True)


def test_running_a_study_without_optuna_gives_an_install_hint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "optuna", None)

    with pytest.raises(ImportError, match=r'pip install -e "\.\[tuning\]"'):
        run_study(make_spec(), quadratic)
    assert "tuning" in INSTALL_HINT


# --- samplers ------------------------------------------------------------------------


def test_seeded_sequential_studies_are_reproducible() -> None:
    first = run_study(make_spec(name="a"), quadratic)
    second = run_study(make_spec(name="b"), quadratic)
    other_seed = run_study(make_spec(name="c", sampler_seed=7), quadratic)

    assert len(first.trials) == 6
    assert trial_params(first) == trial_params(second)
    assert [t.value for t in first.trials] == [t.value for t in second.trials]
    assert trial_params(first) != trial_params(other_seed)


def test_random_sampler_is_seeded() -> None:
    spec = make_spec(sampler="random", n_trials=4)

    assert trial_params(run_study(spec, quadratic)) == trial_params(run_study(spec, quadratic))


def test_grid_sampler_runs_every_threshold_combination_exactly_once() -> None:
    study = run_study(grid_spec(), grid_objective)

    combinations = [
        (t.params["high_threshold"], t.params["low_threshold"], t.params["cooldown_ticks"])
        for t in study.trials
    ]
    expected = set(itertools.product([0.6, 0.7, 0.8], [0.2, 0.3], [3, 5, 10]))
    assert len(study.trials) == 18
    assert len(set(combinations)) == 18
    assert set(combinations) == expected
    assert all(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)
    assert study.best_params == {"high_threshold": 0.6, "low_threshold": 0.3, "cooldown_ticks": 3}


# --- storage and resume ---------------------------------------------------------------


def test_studies_resume_from_persistent_storage(optuna_storage: str) -> None:
    first = run_study(make_spec(storage=optuna_storage, n_trials=5), quadratic)
    finished = [(t.number, t.params, t.value) for t in first.trials]

    resumed = run_study(make_spec(storage=optuna_storage, n_trials=8), quadratic)

    assert [t.number for t in resumed.trials] == list(range(8))
    assert [(t.number, t.params, t.value) for t in resumed.trials[:5]] == finished

    again = run_study(make_spec(storage=optuna_storage, n_trials=8), quadratic)
    assert len(again.trials) == 8  # budget already met: nothing re-run


def test_interrupted_grid_study_resumes_without_repeating_combinations(
    optuna_storage: str,
) -> None:
    run_study(grid_spec(storage=optuna_storage, n_trials=7), grid_objective)

    study = run_study(grid_spec(storage=optuna_storage), grid_objective)

    combinations = {tuple(t.params.values()) for t in study.trials}
    assert len(study.trials) == len(combinations) == 18


def test_resuming_with_a_different_definition_is_refused(optuna_storage: str) -> None:
    run_study(make_spec(storage=optuna_storage, n_trials=2), quadratic)

    with pytest.raises(ValueError, match=r"different definition \(objective_version\)"):
        run_study(make_spec(storage=optuna_storage, objective_version="v2"), quadratic)


def test_parallel_execution_is_explicit_and_works() -> None:
    study = run_study(make_spec(sampler="random", n_trials=4, n_jobs=2), quadratic)

    assert len(study.trials) == 4
    assert make_spec().n_jobs == 1  # sequential by default


# --- MLflow linkage --------------------------------------------------------------------


def evaluate_static(context: TrialContext, workload_id: str, split: str, uri: str) -> float:
    """A real workload evaluation: one tracked MLflow run per workload."""
    target = context.trial.suggest_int("target_replicas", 1, 4)
    entry = load_benchmark_manifest().get(workload_id)
    config = SimulatorConfig(
        timing=SimulatorConfig().timing.model_copy(
            update={"control_interval_seconds": 30.0, "episode_duration_seconds": 3600.0}
        )
    )
    env = AutoscalingEnv(config, build_workload(entry))
    run_kind = "train" if split == "train" else "tune"
    spec = tune_spec(workload_id, split, run_kind=run_kind, controller="static")
    with context.track(spec, tracking_uri=uri) as run:
        infos = run_episode(env, StaticController(target, config.replicas), seed=0)
        cost = sum(info["infrastructure_cost"] for info in infos)
        violations = sum(info["sla_violated"] for info in infos) / len(infos)
        run.log_metrics({"infrastructure_cost": cost, "sla_violation_rate": violations})
    return cost + 10 * violations


def test_single_workload_trials_get_one_mlflow_run_each(tracking_uri: str) -> None:
    spec = make_spec(sampler="random", n_trials=3)

    study = run_study(
        spec, lambda context: evaluate_static(context, "syn-train-spike", "train", tracking_uri)
    )

    client = MlflowClient(tracking_uri)
    for trial in study.trials:
        run_ids = trial.user_attrs[RUN_IDS_ATTR]
        assert len(run_ids) == 1
        run = client.get_run(run_ids[0])
        assert run.info.status == "FINISHED"
        assert run.data.tags["scalerl.optuna.study"] == "demo"
        assert run.data.tags["scalerl.optuna.trial"] == str(trial.number)
        assert run.data.tags["scalerl.workload_id"] == "syn-train-spike"
        params = run.data.params
        assert params["hp.optuna.study"] == "demo"
        assert params["hp.optuna.trial"] == str(trial.number)
        assert params["hp.optuna.sampler"] == "random"
        assert params["hp.optuna.sampler_seed"] == "42"
        assert params["hp.optuna.objective_name"] == "demo-quadratic"
        assert params["hp.optuna.search_space_version"] == "v1"
        assert params["hp.optuna.storage"] == "in-memory"
        assert params["hp.optuna.params.target_replicas"] == str(trial.params["target_replicas"])
        assert "infrastructure_cost" in run.data.metrics


def test_multi_workload_trials_keep_one_mlflow_run_per_workload(tracking_uri: str) -> None:
    workloads = [
        ("syn-train-spike", "train"),
        ("syn-val-bursty", "validation"),
        ("syn-val-ramp-down", "validation"),
    ]
    spec = make_spec(
        sampler="random", n_trials=2, tuning_workload_ids=tuple(w for w, _ in workloads)
    )

    def objective(context: TrialContext) -> float:
        scores = [evaluate_static(context, w, split, tracking_uri) for w, split in workloads]
        return sum(scores) / len(scores)

    study = run_study(spec, objective)

    client = MlflowClient(tracking_uri)
    all_run_ids: list[str] = []
    for trial in study.trials:
        run_ids = trial.user_attrs[RUN_IDS_ATTR]
        assert len(run_ids) == 3
        runs = [client.get_run(run_id) for run_id in run_ids]
        assert [r.data.tags["scalerl.workload_id"] for r in runs] == [w for w, _ in workloads]
        assert [r.data.tags["scalerl.workload_split"] for r in runs] == [s for _, s in workloads]
        assert {r.data.tags["scalerl.optuna.trial"] for r in runs} == {str(trial.number)}
        all_run_ids += run_ids
    assert len(set(all_run_ids)) == 6


def test_track_rejects_runs_outside_the_study(tracking_uri: str) -> None:
    seen: list[str] = []

    def objective(context: TrialContext) -> float:
        with pytest.raises(ValueError, match="not one of study 'demo'"):
            with context.track(
                tune_spec("syn-val-bursty", "validation"), tracking_uri=tracking_uri
            ):
                pass
        evaluate = tune_spec("syn-train-spike", run_kind="evaluate")
        with pytest.raises(ValueError, match="only track train/tune runs"):
            with context.track(evaluate, tracking_uri=tracking_uri):
                pass
        seen.append("checked")
        return 0.0

    run_study(make_spec(n_trials=1), objective)

    assert seen == ["checked"]


# --- trial states ------------------------------------------------------------------------


def test_failing_evaluation_fails_both_the_trial_and_its_run(tracking_uri: str) -> None:
    storage_spec = make_spec(n_trials=1)
    captured: list[optuna.Study] = []

    def objective(context: TrialContext) -> float:
        captured.append(context.trial.study)
        with context.track(tune_spec("syn-train-spike"), tracking_uri=tracking_uri):
            raise RuntimeError("simulator exploded")

    with pytest.raises(RuntimeError, match="simulator exploded"):
        run_study(storage_spec, objective)

    trial = captured[0].trials[0]
    assert trial.state == optuna.trial.TrialState.FAIL
    run_ids = trial.user_attrs[RUN_IDS_ATTR]
    assert len(run_ids) == 1  # the link survives the failure
    assert MlflowClient(tracking_uri).get_run(run_ids[0]).info.status == "FAILED"


def test_failures_can_be_caught_so_the_study_continues(optuna_storage: str) -> None:
    calls: list[int] = []

    def flaky(context: TrialContext) -> float:
        calls.append(context.number)
        if context.number == 1:
            raise RuntimeError("one bad configuration")
        return quadratic(context)

    study = run_study(make_spec(storage=optuna_storage, n_trials=4), flaky, catch=(RuntimeError,))

    states = [t.state for t in study.trials]
    assert states.count(optuna.trial.TrialState.FAIL) == 1
    assert states.count(optuna.trial.TrialState.COMPLETE) == 3
    assert calls == [0, 1, 2, 3]


def test_pruned_trials_stay_distinguishable(tracking_uri: str) -> None:
    captured: list[optuna.Study] = []

    def objective(context: TrialContext) -> float:
        captured.append(context.trial.study)
        with context.track(tune_spec("syn-train-spike"), tracking_uri=tracking_uri) as run:
            run.log_metric("validation_cost", 12.0, step=0)
            context.report(12.0, step=0)
            context.prune()
        return 0.0

    run_study(make_spec(n_trials=1, pruner="median"), objective)

    trial = captured[0].trials[0]
    assert trial.state == optuna.trial.TrialState.PRUNED
    run = MlflowClient(tracking_uri).get_run(trial.user_attrs[RUN_IDS_ATTR][0])
    assert run.info.status == "FINISHED"
    assert run.data.tags["scalerl.optuna.trial_state"] == "PRUNED"
