"""Tests for the seeded Optuna DQN study (#15): tiny budgets, no performance assertions."""

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import optuna
import pytest
from mlflow import MlflowClient

from scalerl.training.dqn import MODEL_SOURCE_TAG, DQNHyperparameters, load_hyperparameters
from scalerl.tuning import RUN_IDS_ATTR, TRIAL_STATE_TAG
from scalerl.tuning import dqn as tuning
from scalerl.tuning.dqn import (
    OBJECTIVE_NAME,
    SEARCH_SPACE_VERSION,
    DQNTuningResult,
    TrialAggregate,
    run_dqn_study,
    select_trial,
    suggest_hyperparameters,
)

TRAIN = "syn-train-spike"
VALIDATION = ("syn-val-bursty",)


def tiny_suggest(trial: optuna.Trial) -> DQNHyperparameters:
    """The real v1 search space, shrunk to test-sized networks and buffers."""
    sampled = suggest_hyperparameters(trial)
    return sampled.model_copy(
        update={"buffer_size": 500, "learning_starts": 32, "net_arch": (16, 16)}
    )


@pytest.fixture
def tracking_uri(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.setenv("MLFLOW_DISABLE_AGENT_HINT", "1")
    yield f"sqlite:///{tmp_path / 'mlflow.db'}"


def tiny_study(tracking_uri: str, **kwargs: Any) -> DQNTuningResult:
    options: dict[str, Any] = {
        "training_workload_id": TRAIN,
        "validation_workload_ids": VALIDATION,
        "n_trials": 2,
        "timesteps": 96,
        "tracking_uri": tracking_uri,
        "log_interval": 48,
        "suggest": tiny_suggest,
    }
    return run_dqn_study(**{**options, **kwargs})


def test_search_space_v1_is_the_documented_one() -> None:
    trial = optuna.trial.FixedTrial(
        {
            "net_arch": "128x128",
            "learning_rate": 3e-4,
            "gamma": 0.98,
            "batch_size": 128,
            "target_update_interval": 250,
            "exploration_fraction": 0.3,
            "exploration_final_eps": 0.02,
        }
    )

    sampled = suggest_hyperparameters(trial)

    assert SEARCH_SPACE_VERSION == "dqn-search-v1"
    assert sampled.net_arch == (128, 128)
    assert (sampled.learning_rate, sampled.gamma, sampled.batch_size) == (3e-4, 0.98, 128)
    # Unsearched settings keep the dqn-v1 defaults.
    defaults = DQNHyperparameters()
    assert (sampled.buffer_size, sampled.learning_starts, sampled.train_freq) == (
        defaults.buffer_size,
        defaults.learning_starts,
        defaults.train_freq,
    )


def test_selection_is_sla_first_then_cost_queue_churn_then_trial_number() -> None:
    def aggregate(number: int, sla: float, cost: float, queue: float, churn: float) -> Any:
        return TrialAggregate(number, sla, cost, queue, churn)

    candidates = [
        aggregate(0, 0.2, 0.1, 0.1, 0.1),
        aggregate(1, 0.1, 0.9, 0.9, 0.9),  # best SLA wins despite cost
        aggregate(2, 0.1, 0.9, 0.9, 0.9),  # exact tie: lower trial number wins
    ]
    assert select_trial(candidates).number == 1
    assert (
        select_trial([aggregate(3, 0.1, 0.5, 0.2, 0.0), aggregate(4, 0.1, 0.4, 0.9, 0.9)]).number
        == 4
    )
    assert select_trial(list(reversed(candidates))).number == 1
    with pytest.raises(ValueError, match="no completed trials"):
        select_trial([])


def test_tiny_study_tracks_trials_and_selects_deterministically(
    tracking_uri: str, tmp_path: Path
) -> None:
    result = tiny_study(tracking_uri)
    client = MlflowClient(tracking_uri)

    assert result.trial_count == 2
    assert (result.objective_name, result.search_space_version) == (
        OBJECTIVE_NAME,
        SEARCH_SPACE_VERSION,
    )
    assert (result.sampler, result.sampler_seed) == ("tpe", 42)
    assert result.validation_workload_ids == VALIDATION
    assert "validation_sla_violation_rate" in result.selected_validation_metrics
    assert set(result.selected_params) == {
        "net_arch",
        "learning_rate",
        "gamma",
        "batch_size",
        "target_update_interval",
        "exploration_fraction",
        "exploration_final_eps",
    }
    # One training run + one validation run per trial, all `tune` runs of the trial.
    assert len(result.selected_mlflow_run_ids) == 2
    training_run = client.get_run(result.selected_training_run_id)
    assert training_run.data.tags["scalerl.run_kind"] == "tune"
    assert training_run.data.tags["scalerl.controller"] == "dqn"
    assert training_run.data.tags[TRIAL_STATE_TAG] == "COMPLETE"
    assert training_run.data.params["hp.optuna.search_space_version"] == SEARCH_SPACE_VERSION
    assert training_run.data.params["hp.optuna.sampler"] == "tpe"
    assert {a.path for a in client.list_artifacts(training_run.info.run_id, "model")} >= {
        "model/model.zip",
        "model/compatibility.json",
    }
    validation_run = client.get_run(result.selected_mlflow_run_ids[1])
    assert validation_run.data.tags["scalerl.run_kind"] == "tune"
    assert validation_run.data.tags[MODEL_SOURCE_TAG] == result.selected_training_run_id
    assert "sla_violation_rate" in validation_run.data.metrics

    # The selection can be retrained through the training CLI's --hyperparameters.
    path = result.save(tmp_path / "dqn-tuning.json")
    hyperparameters, source = load_hyperparameters(path)
    assert hyperparameters == result.selected_hyperparameters
    assert source == f"optuna:{result.study_name}#trial{result.selected_trial_number}"
    assert DQNTuningResult.load(path) == result


def test_resumed_study_runs_only_the_remaining_budget(tracking_uri: str, tmp_path: Path) -> None:
    storage = f"sqlite:///{tmp_path / 'optuna.db'}"
    first = tiny_study(tracking_uri, n_trials=1, storage=storage)
    second = tiny_study(tracking_uri, n_trials=2, storage=storage)

    study = optuna.load_study(study_name="dqn-v1", storage=storage)
    assert (first.trial_count, second.trial_count, len(study.trials)) == (1, 2, 2)
    assert all(trial.user_attrs[RUN_IDS_ATTR] for trial in study.trials)
    assert study.trials[0].params == first.selected_params


@pytest.mark.parametrize(
    ("train", "validation", "message"),
    [
        ("syn-test-spike-hard", VALIDATION, "held-out test"),
        (TRAIN, ("azure-test-993600",), "held-out test"),
        ("syn-val-bursty", VALIDATION, "a validation workload"),
        (TRAIN, ("syn-train-bursty",), "a train workload"),
    ],
)
def test_held_out_and_wrong_split_workloads_are_rejected_before_training(
    train: str,
    validation: tuple[str, ...],
    message: str,
    tracking_uri: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def refuse(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("tuning started before the workload guardrail")

    monkeypatch.setattr(tuning, "build_workloads", refuse)
    monkeypatch.setattr(tuning, "run_study", refuse)

    with pytest.raises(ValueError, match=message):
        tiny_study(tracking_uri, training_workload_id=train, validation_workload_ids=validation)
