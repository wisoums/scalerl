"""Tests for controller-agnostic evaluation and the tracked predictive evaluation."""

import math
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from mlflow import MlflowClient

from scalerl.controllers import ForecastRecord, RandomController, StaticController
from scalerl.environment import AutoscalingEnv, SimulatorConfig
from scalerl.evaluation import (
    ForecastAccuracy,
    evaluate_controller_episode,
    score_forecasts,
    summarize_episode,
)
from scalerl.evaluation.predictive import evaluate_predictive
from scalerl.workloads import steady_workload


def infos_with(rates: dict[int, float]) -> list[dict[str, Any]]:
    return [{"tick": tick, "request_rate": rate} for tick, rate in rates.items()]


def record(target_tick: int, forecast: float, source_tick: int = 0) -> ForecastRecord:
    return ForecastRecord(source_tick, target_tick, forecast, sample_count=4)


# --- forecast accuracy -------------------------------------------------------------------


def test_forecast_errors_are_computed_on_matured_targets_only() -> None:
    infos = infos_with({0: 100.0, 1: 110.0, 2: 130.0})
    forecasts = [record(1, 120.0), record(2, 120.0), record(5, 999.0)]  # tick 5 never happened

    accuracy = score_forecasts(forecasts, infos)

    # errors: +10 and -10
    assert accuracy.forecast_count == 2
    assert accuracy.forecast_mae_rps == pytest.approx(10.0)
    assert accuracy.forecast_rmse_rps == pytest.approx(10.0)
    assert accuracy.forecast_bias_rps == pytest.approx(0.0)


def test_rmse_and_bias_weight_errors_correctly() -> None:
    infos = infos_with({1: 100.0, 2: 100.0, 3: 100.0})
    forecasts = [record(1, 100.0), record(2, 130.0), record(3, 140.0)]  # errors 0, 30, 40

    accuracy = score_forecasts(forecasts, infos)

    assert accuracy.forecast_mae_rps == pytest.approx(70.0 / 3)
    assert accuracy.forecast_rmse_rps == pytest.approx(math.sqrt((0 + 900 + 1600) / 3))
    assert accuracy.forecast_bias_rps == pytest.approx(70.0 / 3)


def test_no_matured_forecasts_are_reported_cleanly() -> None:
    accuracy = score_forecasts([record(9, 50.0)], infos_with({0: 10.0}))

    assert accuracy == ForecastAccuracy(0, None, None, None)
    assert accuracy.as_metrics() == {"forecast_count": 0.0}


# --- generic episode metrics ---------------------------------------------------------------


def test_any_controller_can_be_evaluated_with_the_shared_metrics() -> None:
    config = SimulatorConfig()
    trace = steady_workload(duration_seconds=3600, control_interval_seconds=30, rate=60.0)

    for controller in (RandomController(seed=0), StaticController(2, config.replicas)):
        evaluation = evaluate_controller_episode(AutoscalingEnv(config, trace), controller)
        assert len(evaluation.infos) == 120
        assert evaluation.metrics == summarize_episode(evaluation.infos, config)
        assert 0.0 <= evaluation.metrics.normalized_cost <= 1.0


def test_empty_episodes_cannot_be_summarized() -> None:
    with pytest.raises(ValueError, match="empty episode"):
        summarize_episode([], SimulatorConfig())


def test_evaluation_does_not_import_optional_services() -> None:
    code = (
        "import sys\n"
        "import scalerl.evaluation, scalerl.controllers\n"
        "assert not {'mlflow', 'streamlit', 'optuna'} & set(sys.modules)\n"
    )

    subprocess.run([sys.executable, "-c", code], check=True)


# --- tracked predictive evaluation -------------------------------------------------------------


@pytest.fixture
def tracking_uri(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.setenv("MLFLOW_DISABLE_AGENT_HINT", "1")
    yield f"sqlite:///{tmp_path / 'mlflow.db'}"


def test_each_workload_is_one_tracked_predictive_run(tracking_uri: str) -> None:
    workloads = ("syn-train-ramp-up", "syn-val-bursty")

    results = evaluate_predictive(workload_ids=workloads, tracking_uri=tracking_uri)

    client = MlflowClient(tracking_uri)
    assert [r.workload_id for r in results] == list(workloads)
    assert len({r.mlflow_run_id for r in results}) == 2
    for result, split in zip(results, ("train", "validation"), strict=True):
        run = client.get_run(result.mlflow_run_id)
        tags, params, metrics = run.data.tags, run.data.params, run.data.metrics
        assert run.info.status == "FINISHED"
        assert tags["scalerl.controller"] == "predictive"
        assert tags["scalerl.run_kind"] == "evaluate"
        assert tags["scalerl.workload_id"] == result.workload_id
        assert tags["scalerl.workload_split"] == split
        assert tags["scalerl.simulator_config_source"] == "default"
        assert params["hp.history_window_ticks"] == "4"
        assert params["hp.target_utilization"] == "0.8"
        assert params["hp.forecast_method"] == "linear-trend"
        assert params["hp.forecast_horizon_ticks"] == "3"
        assert params["hp.capacity_policy"] == "forecast-plus-backlog-v1"
        assert params["hp.backlog_recovery_ticks"] == "1"
        assert params["sim.replicas.startup_delay_seconds"] == "60.0"
        for name in (
            "forecast_count",
            "forecast_mae_rps",
            "forecast_rmse_rps",
            "forecast_bias_rps",
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
            assert name in metrics
        # Forecasts made in the last 3 ticks target ticks beyond the episode.
        assert metrics["forecast_count"] == 120 - 3
        assert metrics["forecast_mae_rps"] == pytest.approx(result.forecast.forecast_mae_rps)


@pytest.mark.parametrize("held_out", ["syn-test-spike-hard", "azure-test-993600"])
def test_development_evaluation_rejects_held_out_workloads(
    held_out: str, tracking_uri: str
) -> None:
    with pytest.raises(ValueError, match="held-out test workload"):
        evaluate_predictive(workload_ids=("syn-train-spike", held_out), tracking_uri=tracking_uri)

    assert (
        MlflowClient(tracking_uri).search_experiments(filter_string="name = 'scalerl-predictive'")
        == []
    )
