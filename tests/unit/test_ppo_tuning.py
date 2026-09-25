"""Tests for the seeded Optuna PPO study (#16): tiny budgets, no performance assertions."""

import itertools
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import optuna
import pytest
from mlflow import MlflowClient

from scalerl.training.common import MODEL_SOURCE_TAG
from scalerl.training.ppo import PPOHyperparameters, load_hyperparameters
from scalerl.tuning import TRIAL_STATE_TAG
from scalerl.tuning import sb3 as sb3_tuning
from scalerl.tuning.ppo import (
    BATCH_SIZES,
    BUDGET_MULTIPLE,
    N_STEPS,
    OBJECTIVE_NAME,
    SEARCH_SPACE_VERSION,
    PPOTuningResult,
    run_ppo_study,
    suggest_hyperparameters,
)

TRAIN = "syn-train-spike"
VALIDATION = ("syn-val-bursty",)
TINY_ROLLOUT = 64


def tiny_suggest(trial: optuna.Trial) -> PPOHyperparameters:
    """The real v1 search space, shrunk to test-sized rollouts and networks."""
    sampled = suggest_hyperparameters(trial)
    return sampled.model_copy(
        update={"n_steps": TINY_ROLLOUT, "batch_size": 32, "n_epochs": 1, "net_arch": (16, 16)}
    )


@pytest.fixture
def tracking_uri(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.setenv("MLFLOW_DISABLE_AGENT_HINT", "1")
    yield f"sqlite:///{tmp_path / 'mlflow.db'}"


def tiny_study(tracking_uri: str, **kwargs: Any) -> PPOTuningResult:
    options: dict[str, Any] = {
        "training_workload_id": TRAIN,
        "validation_workload_ids": VALIDATION,
        "n_trials": 2,
        "timesteps": 128,
        "tracking_uri": tracking_uri,
        "log_interval": 64,
        "suggest": tiny_suggest,
        "rollout_sizes": (TINY_ROLLOUT,),
    }
    return run_ppo_study(**{**options, **kwargs})


def test_every_search_space_combination_is_a_valid_config() -> None:
    assert SEARCH_SPACE_VERSION == "ppo-search-v1"
    for n_steps, batch_size in itertools.product(N_STEPS, BATCH_SIZES):
        assert n_steps % batch_size == 0
        PPOHyperparameters(n_steps=n_steps, batch_size=batch_size)
    assert BUDGET_MULTIPLE == 2048
    assert 204_800 % BUDGET_MULTIPLE == 0  # the default budget fits every rollout size


def test_search_space_v1_sample_keeps_unsearched_defaults() -> None:
    trial = optuna.trial.FixedTrial(
        {
            "net_arch": "128x128",
            "learning_rate": 1e-4,
            "gamma": 0.98,
            "gae_lambda": 0.9,
            "clip_range": 0.1,
            "ent_coef": 0.01,
            "n_steps": 512,
            "batch_size": 256,
            "n_epochs": 5,
        }
    )

    sampled = suggest_hyperparameters(trial)

    assert (sampled.n_steps, sampled.batch_size, sampled.net_arch) == (512, 256, (128, 128))
    defaults = PPOHyperparameters()
    assert (sampled.vf_coef, sampled.max_grad_norm, sampled.normalize_advantage) == (
        defaults.vf_coef,
        defaults.max_grad_norm,
        defaults.normalize_advantage,
    )


def test_tiny_study_tracks_trials_and_selects(tracking_uri: str, tmp_path: Path) -> None:
    result = tiny_study(tracking_uri)
    client = MlflowClient(tracking_uri)

    assert result.trial_count == 2
    assert (result.objective_name, result.search_space_version) == (
        OBJECTIVE_NAME,
        SEARCH_SPACE_VERSION,
    )
    assert OBJECTIVE_NAME == "ppo-sla-first"
    assert (result.sampler, result.sampler_seed) == ("tpe", 42)
    assert "validation_sla_violation_rate" in result.selected_validation_metrics
    assert {"learning_rate", "gae_lambda", "clip_range", "n_steps"} <= set(result.selected_params)
    assert len(result.selected_mlflow_run_ids) == 2  # training + one validation run

    training_run = client.get_run(result.selected_training_run_id)
    assert training_run.data.tags["scalerl.run_kind"] == "tune"
    assert training_run.data.tags["scalerl.controller"] == "ppo"
    assert training_run.data.tags[TRIAL_STATE_TAG] == "COMPLETE"
    assert training_run.data.params["hp.optuna.search_space_version"] == SEARCH_SPACE_VERSION
    assert training_run.data.params["hp.optuna.objective_name"] == OBJECTIVE_NAME
    assert client.list_artifacts(training_run.info.run_id, "checkpoints") == []
    validation_run = client.get_run(result.selected_mlflow_run_ids[1])
    assert validation_run.data.tags["scalerl.run_kind"] == "tune"
    assert validation_run.data.tags[MODEL_SOURCE_TAG] == result.selected_training_run_id

    path = result.save(tmp_path / "ppo-tuning.json")
    hyperparameters, source = load_hyperparameters(path)
    assert hyperparameters == result.selected_hyperparameters
    assert source == f"optuna:{result.study_name}#trial{result.selected_trial_number}"
    assert PPOTuningResult.load(path) == result


def test_trial_budget_must_fit_every_proposable_rollout_size(
    tracking_uri: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("tuning started before the budget check")

    monkeypatch.setattr(sb3_tuning, "build_workloads", refuse)
    monkeypatch.setattr(sb3_tuning, "run_study", refuse)

    # 1024 fits n_steps 512 and 1024 but not 2048: some trials would overshoot.
    with pytest.raises(ValueError, match="multiple of n_steps \\(2048\\)"):
        run_ppo_study(
            training_workload_id=TRAIN,
            validation_workload_ids=VALIDATION,
            timesteps=1024,
            tracking_uri=tracking_uri,
        )


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

    monkeypatch.setattr(sb3_tuning, "build_workloads", refuse)
    monkeypatch.setattr(sb3_tuning, "run_study", refuse)

    with pytest.raises(ValueError, match=message):
        tiny_study(tracking_uri, training_workload_id=train, validation_workload_ids=validation)
