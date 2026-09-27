"""Tests for the proactive predictive baseline ``predictive-seasonal-v1`` (#80).

Synthetic fixtures and TRAIN workloads only; no validation or test demand is read.
"""

import json
import math
from typing import Any

import numpy as np
import pytest

from scalerl.benchmarks import build_workload, load_benchmark_manifest
from scalerl.controllers import (
    HistoricalDemandProfile,
    PredictiveController,
    ProactivePredictiveController,
    profile_level_factor,
    run_episode,
)
from scalerl.controllers import predictive as predictive_v1
from scalerl.controllers import proactive_predictive as seasonal
from scalerl.environment import (
    DELTA_V1,
    DESIRED_REPLICAS_V1,
    ActionConfig,
    ActionContract,
    AutoscalingEnv,
    SimulatorConfig,
)
from scalerl.evaluation.forecast import score_forecasts
from scalerl.workloads import WorkloadTrace

DESIRED = SimulatorConfig(action=ActionConfig(semantics=DESIRED_REPLICAS_V1))
CONTRACT = ActionContract.from_config(DESIRED)


def trace(rates: list[float]) -> WorkloadTrace:
    return WorkloadTrace(rates, 30.0)


def profile_of(*rates: list[float]) -> HistoricalDemandProfile:
    return HistoricalDemandProfile.from_traces({f"h{i}": trace(r) for i, r in enumerate(rates)})


def controller(profile: HistoricalDemandProfile | None = None) -> ProactivePredictiveController:
    return ProactivePredictiveController.from_config(DESIRED, profile=profile)


def feed(
    ctrl: ProactivePredictiveController,
    rates: list[float],
    *,
    active: int,
    pending: int = 0,
    queued: float = 0.0,
    start: int = 0,
) -> int:
    code = -1
    for offset, rate in enumerate(rates):
        code = ctrl.act(
            np.zeros(1),
            {
                "tick": start + offset,
                "request_rate": rate,
                "queued_requests": queued,
                "active_replicas": active,
                "pending_replicas": pending,
            },
        )
    return code


def target(code: int, committed: int) -> int:
    return CONTRACT.target_for(code, committed)


# --- identities ----------------------------------------------------------------------------------


def test_new_and_existing_identifiers() -> None:
    assert (seasonal.VARIANT, seasonal.FORECAST_METHOD, seasonal.CAPACITY_POLICY) == (
        "predictive-seasonal-v1",
        "historical-profile-plus-linear-v1",
        "proactive-scaleout-conservative-scalein-v1",
    )
    assert (predictive_v1.FORECAST_METHOD, predictive_v1.CAPACITY_POLICY) == (
        "linear-trend",
        "forecast-plus-backlog-v1",
    )
    assert (seasonal.HISTORY_WINDOW_TICKS, seasonal.TARGET_UTILIZATION) == (4, 0.8)
    assert seasonal.PROFILE_AGGREGATION == "median"


def test_predictive_v1_uses_the_extracted_helpers_identically() -> None:
    samples = ((3, 10.0), (4, 30.0), (5, 20.0), (6, 60.0))
    ctrl = PredictiveController.from_config(DESIRED)
    for tick, rate in samples:
        ctrl.act(np.zeros(1), {"tick": tick, "request_rate": rate, "queued_requests": 0.0,
                               "active_replicas": 1, "pending_replicas": 0})  # fmt: skip
    record = ctrl.forecasts[-1]
    assert record.forecast_rps == predictive_v1.linear_trend_forecast(samples, record.target_tick)
    assert predictive_v1.size_replicas(
        80.0, capacity_per_replica_rps=40.0, min_replicas=1, max_replicas=10
    ) == (2, 2)


# --- historical profile ----------------------------------------------------------------------


def test_profile_is_the_per_tick_median() -> None:
    profile = profile_of([1.0, 10.0, 5.0], [3.0, 30.0, 5.0], [2.0, 20.0, 500.0])
    assert profile.rates == (2.0, 20.0, 5.0)  # the 500 spike does not move the median
    assert profile.aggregation == "median" and profile.ticks == 3
    assert profile.rate_at(1) == 20.0 and profile.rate_at(3) is None and profile.rate_at(-1) is None


def test_profile_is_deterministic_and_serializes_stably() -> None:
    a = profile_of([1.0, 2.0], [3.0, 4.0], [5.0, 6.0])
    b = profile_of([1.0, 2.0], [3.0, 4.0], [5.0, 6.0])
    assert a == b and a.profile_id == b.profile_id and a.to_json() == b.to_json()
    assert HistoricalDemandProfile.from_json(a.to_json()) == a
    assert json.loads(a.to_json())["rates"] == [3.0, 4.0]
    assert profile_of([1.0, 2.0], [3.0, 4.5], [5.0, 6.0]).profile_id != a.profile_id


def test_profile_requires_aligned_equal_timing() -> None:
    with pytest.raises(ValueError, match="equal lengths"):
        profile_of([1.0, 2.0], [1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="control interval"):
        HistoricalDemandProfile.from_traces(
            {"a": WorkloadTrace([1.0], 30.0), "b": WorkloadTrace([1.0], 60.0)}
        )
    with pytest.raises(ValueError, match="at least one"):
        HistoricalDemandProfile.from_traces({})
    with pytest.raises(ValueError, match="control interval differs"):
        ProactivePredictiveController.from_config(
            DESIRED, profile=HistoricalDemandProfile.from_traces({"a": WorkloadTrace([1.0], 60.0)})
        )


@pytest.mark.parametrize(
    ("workload_id", "message"),
    [("syn-val-bursty", "validation"), ("syn-test-seasonal-shifted", "held-out test"),
     ("azure-val-734400", "validation"), ("azure-test-993600", "held-out test")],
)  # fmt: skip
def test_benchmark_profile_history_is_train_only(workload_id: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        HistoricalDemandProfile.from_benchmark(["syn-train-bursty", workload_id])


def test_benchmark_profile_from_train_workloads() -> None:
    profile = HistoricalDemandProfile.from_benchmark(["syn-train-seasonal", "syn-train-ramp-up"])
    assert profile.source_workload_ids == ("syn-train-seasonal", "syn-train-ramp-up")
    assert profile.ticks == 120 and profile.control_interval_seconds == 30.0


def test_profile_is_immutable_and_survives_resets() -> None:
    profile = profile_of([10.0] * 20, [10.0] * 20)
    before = profile.to_json()
    ctrl = controller(profile)
    feed(ctrl, [30.0, 40.0], active=2)
    ctrl.reset()
    feed(ctrl, [5.0], active=2)
    assert ctrl.profile is profile and profile.to_json() == before
    with pytest.raises(AttributeError):
        profile.rates = (0.0,)  # type: ignore[misc]


# --- level adaptation ------------------------------------------------------------------------


def test_level_factor_is_median_of_latest_four_ratios() -> None:
    profile = profile_of([10.0] * 10)
    samples = [(0, 1000.0), (1, 10.0), (2, 20.0), (3, 30.0), (4, 40.0)]
    factor, valid = profile_level_factor(samples, profile)
    assert (factor, valid) == (2.5, 4)  # ratios 1, 2, 3, 4 of ticks 1-4; tick 0 is too old


def test_level_factor_skips_zero_and_tiny_profile_values() -> None:
    profile = profile_of([0.0, 1e-9, 10.0, 20.0, 0.0])
    factor, valid = profile_level_factor([(0, 5.0), (1, 5.0), (2, 30.0), (3, 20.0)], profile)
    assert (factor, valid) == (2.0, 2)  # median(3.0, 1.0)
    assert profile_level_factor([(0, 5.0), (4, 5.0), (9, 5.0)], profile) == (1.0, 0)
    assert profile_level_factor([], profile) == (1.0, 0)


def test_adjusted_profile_forecast_uses_completed_samples_only() -> None:
    profile = profile_of([10.0] * 5 + [100.0] * 15)
    ctrl = controller(profile)
    feed(ctrl, [20.0, 20.0], active=1)  # ticks 0-1: observed at twice the profile level
    record = ctrl.forecasts[-1]
    assert record.target_tick == 4 and record.profile_level_factor == 2.0
    assert record.historical_profile_forecast_rps == 20.0  # profile[4] = 10 x 2
    feed(ctrl, [20.0, 20.0], active=1, start=2)
    record = ctrl.forecasts[-1]
    assert record.target_tick == 6 and record.historical_profile_forecast_rps == 200.0


# --- no future leakage ---------------------------------------------------------------------------


def _decisions_until(workload: WorkloadTrace, tick: int, profile: HistoricalDemandProfile) -> Any:
    env = AutoscalingEnv(DESIRED, workload)
    ctrl = controller(profile)
    observation, info = env.reset(seed=0)
    ctrl.reset(seed=0)
    actions = []
    while True:
        action = ctrl.act(observation, env.decision_info(info))
        actions.append(action)
        if info.get("tick") == tick:
            return actions, ctrl.forecasts, ctrl.last_decision
        observation, _, _, _, info = env.step(action)


def test_decisions_never_depend_on_future_current_demand() -> None:
    profile = profile_of([50.0] * 60 + [300.0] * 60, [60.0] * 60 + [250.0] * 60)
    common = [40.0 + 5.0 * (t % 7) for t in range(40)]
    calm = trace(common + [40.0] * 80)
    storm = trace(common + [900.0] * 80)
    assert calm.request_rates[:40] == storm.request_rates[:40]
    assert calm.request_rates[40:] != storm.request_rates[40:]
    assert _decisions_until(calm, 39, profile) == _decisions_until(storm, 39, profile)


def test_forecast_records_hold_no_actual_future_demand() -> None:
    fields = set(seasonal.ProactiveForecastRecord.__dataclass_fields__)
    assert not {f for f in fields if "actual" in f}
    ctrl = controller(profile_of([10.0] * 20))
    feed(ctrl, [10.0, 12.0], active=1)
    assert ctrl.forecasts[-1].target_tick == 1 + 3


def test_generator_parameters_are_not_an_input() -> None:
    import inspect

    source = inspect.getsource(seasonal)
    for hidden in ("amplitude", "period_seconds", "phase_seconds", "request_rates["):
        assert hidden not in source


# --- proactive scale-out ----------------------------------------------------------------------


def test_horizon_is_the_predictive_v1_startup_arithmetic() -> None:
    ctrl = controller()
    assert (
        ctrl.forecast_horizon_ticks
        == PredictiveController.from_config(DESIRED).forecast_horizon_ticks
        == 3
    )
    feed(ctrl, [10.0], active=1, start=7)
    record = ctrl.forecasts[-1]
    assert (record.source_tick, record.target_tick) == (7, 10)
    assert record.horizon_seconds == 90.0


def test_recurring_profile_scales_out_before_the_rise_is_observed() -> None:
    profile = profile_of([10.0] * 9 + [400.0] * 11, [10.0] * 9 + [380.0] * 11)
    with_profile, without = controller(profile), controller()
    rates = [10.0] * 7  # ticks 0-6 all look flat; the recurring rise is at tick 9
    code = feed(with_profile, rates, active=1)
    record = with_profile.forecasts[-1]
    assert record.target_tick == 9 and record.profile_used
    assert record.proactive_forecast_rps == max(record.linear_forecast_rps, 390.0) == 390.0
    assert target(code, 1) == 10  # ceil(390 / 40): the whole target requested before tick 9
    assert target(feed(without, rates, active=1), 1) == 1  # linear alone sees nothing coming


def test_linear_trend_fallback_without_profile() -> None:
    ctrl = controller()
    code = feed(ctrl, [40.0, 80.0, 120.0, 160.0], active=1)
    record = ctrl.forecasts[-1]
    assert record.historical_profile_forecast_rps is None and not record.profile_used
    assert record.proactive_forecast_rps == record.linear_forecast_rps == 280.0
    assert target(code, 1) == 7


def test_proactive_forecast_takes_the_larger_of_linear_and_profile() -> None:
    profile = profile_of([100.0] * 20)
    ctrl = controller(profile)
    feed(ctrl, [100.0, 150.0, 200.0, 250.0], active=1)  # rising faster than the profile
    record = ctrl.forecasts[-1]
    assert record.linear_forecast_rps > record.historical_profile_forecast_rps  # type: ignore[operator]
    assert record.proactive_forecast_rps == record.linear_forecast_rps


def test_profile_beyond_its_ticks_falls_back_to_linear() -> None:
    ctrl = controller(profile_of([10.0] * 5))
    feed(ctrl, [10.0, 10.0, 10.0], active=1)  # target tick 5 is outside the 5-tick profile
    assert not ctrl.forecasts[-1].profile_used


def test_scale_out_request_still_follows_startup_lifecycle() -> None:
    config = DESIRED.model_copy(
        update={"replicas": DESIRED.replicas.model_copy(update={"initial_replicas": 2})}
    )
    env = AutoscalingEnv(config, trace([400.0] * 120))
    ctrl = ProactivePredictiveController.from_config(config)
    observation, info = env.reset(seed=0)
    ctrl.act(observation, env.decision_info(info))
    observation, _, _, _, info = env.step(ctrl.act(observation, env.decision_info(info)))
    info = env.step(ctrl.act(observation, env.decision_info(info)))[4]
    assert info["applied_replica_change"] == 8 and info["pending_replicas"] == 8
    assert info["active_replicas"] == 2  # requested, not instantly active


# --- conservative scale-in --------------------------------------------------------------------


def test_never_scales_in_with_a_backlog() -> None:
    ctrl = controller()
    code = feed(ctrl, [10.0] * 4, active=5, queued=30.0)
    assert ctrl.last_decision is not None and ctrl.last_decision.reason == "backlog_hold"
    assert target(code, 5) == 5


def test_does_not_drop_below_observed_need() -> None:
    ctrl = controller()
    code = feed(ctrl, [400.0, 300.0, 200.0, 100.0], active=3)  # forecast ~0, observed 100 -> 3
    decision = ctrl.last_decision
    assert decision is not None and decision.observed_desired_replicas == 3
    assert ctrl.forecasts[-1].desired_replicas == 1
    assert target(code, 3) == 3 and decision.reason == "hold_for_observed_or_forecast"


def test_does_not_drop_below_forecast_need() -> None:
    ctrl = controller()
    code = feed(ctrl, [10.0, 20.0, 30.0, 40.0], active=3)  # observed 40 -> 1, forecast 70 -> 2
    assert ctrl.forecasts[-1].desired_replicas == 2
    assert target(code, 3) == 2


def test_scales_in_one_replica_at_a_time_when_both_are_low() -> None:
    ctrl = controller()
    code = feed(ctrl, [10.0] * 4, active=8)
    assert target(code, 8) == 7 and ctrl.last_decision is not None
    assert ctrl.last_decision.reason == "scale_in"


def test_scale_out_may_jump_several_replicas() -> None:
    ctrl = controller()
    assert target(feed(ctrl, [400.0] * 4, active=2), 2) == 10


def test_bounds_and_pending_capacity() -> None:
    assert target(feed(controller(), [5000.0] * 4, active=4), 4) == 10  # max
    ctrl = controller()
    assert target(feed(ctrl, [0.0] * 4, active=1), 1) == 1  # min: never below
    ctrl = controller()
    code = feed(ctrl, [10.0] * 4, active=2, pending=3)  # committed 5 includes pending
    assert target(code, 5) == 4


def test_episode_is_deterministic_and_resets_isolate_history() -> None:
    workload = build_workload(load_benchmark_manifest().get("syn-train-bursty"))
    ctrl = controller()
    first = run_episode(AutoscalingEnv(DESIRED, workload), ctrl, seed=0)
    records = ctrl.forecasts
    second = run_episode(AutoscalingEnv(DESIRED, workload), ctrl, seed=0)
    assert first == second and ctrl.forecasts == records
    assert all(1 <= info["requested_replica_target"] <= 10 for info in first)
    ups = [i["applied_replica_change"] for i in first if i["applied_replica_change"] > 0]
    downs = [i["applied_replica_change"] for i in first if i["applied_replica_change"] < 0]
    assert downs and min(downs) == -1  # scale-in never exceeds one replica
    assert ups and max(ups) > 1  # direct scale-out used


def test_delta_contract_is_still_encoded_through_action_contract() -> None:
    ctrl = ProactivePredictiveController.from_config(SimulatorConfig())
    assert ctrl.action_contract.semantics == DELTA_V1
    assert feed(ctrl, [400.0] * 4, active=2) == 2  # one step up under delta-v1


# --- forecast metrics -----------------------------------------------------------------------------


def test_forecast_scoring_mae_rmse_bias() -> None:
    from dataclasses import dataclass

    @dataclass
    class F:
        target_tick: int
        forecast_rps: float

    infos = [{"tick": t, "request_rate": r} for t, r in enumerate([10.0, 20.0, 30.0])]
    accuracy = score_forecasts([F(0, 12.0), F(1, 16.0), F(2, 30.0), F(3, 99.0)], infos)
    assert accuracy.forecast_count == 3  # tick 3 is outside the episode
    assert accuracy.forecast_mae_rps == pytest.approx(2.0)
    assert accuracy.forecast_rmse_rps == pytest.approx(math.sqrt(20 / 3))
    assert accuracy.forecast_bias_rps == pytest.approx(-2 / 3)  # forecast - actual
    empty = score_forecasts([F(9, 1.0)], infos)
    assert empty.forecast_count == 0 and empty.as_metrics() == {"forecast_count": 0.0}


def test_scoring_does_not_change_actions() -> None:
    workload = build_workload(load_benchmark_manifest().get("syn-train-spike"))
    ctrl = controller()
    before = run_episode(AutoscalingEnv(DESIRED, workload), ctrl, seed=0)
    score_forecasts(ctrl.forecasts, before)
    after = run_episode(AutoscalingEnv(DESIRED, workload), ctrl, seed=0)
    assert [i["requested_action"] for i in before] == [i["requested_action"] for i in after]
