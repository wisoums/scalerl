"""Tests for the Scenario Lab City View session layer and Streamlit app."""

import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from scalerl.benchmarks import build_workload, load_benchmark_manifest
from scalerl.controllers import RandomController, ThresholdController, decision_info
from scalerl.dashboard import (
    HELD_OUT_WARNING,
    ManagerSpec,
    Scenario,
    ScenarioInfo,
    ScenarioSession,
    azure_capacity_note,
    benchmark_choices,
    benchmark_scenario,
    custom_scenario,
    icon_row,
)
from scalerl.environment import ReplicaConfig, SimulatorConfig, TimingConfig
from scalerl.workloads import steady_workload

SCALE_DOWN, HOLD, SCALE_UP = 0, 1, 2
APP = Path(__file__).parents[2] / "src" / "scalerl" / "dashboard" / "app.py"
AZURE_FIXTURE = Path(__file__).parent.parent / "fixtures" / "azure_functions_2021_small.csv"
V1_TIMING = TimingConfig(control_interval_seconds=30.0, episode_duration_seconds=3600.0)


def v1_config(**replicas: Any) -> SimulatorConfig:
    params: dict[str, Any] = {"min_replicas": 1, "max_replicas": 6, "initial_replicas": 2}
    return SimulatorConfig(timing=V1_TIMING, replicas=ReplicaConfig(**{**params, **replicas}))


def make_session(kind: str = "manual", workload: str = "syn-train-spike", **manager: Any) -> Any:
    config = v1_config(**manager.pop("replicas", {}))
    return ScenarioSession(benchmark_scenario(workload), config, ManagerSpec(kind, **manager))  # type: ignore[arg-type]


def spy_on_env_step(session: ScenarioSession, monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    calls: list[Any] = []
    real_step = session.env.step

    def recording_step(action: int) -> Any:
        result = real_step(action)
        calls.append((action, result))
        return result

    monkeypatch.setattr(session.env, "step", recording_step)
    return calls


# --- scenarios ----------------------------------------------------------------


def test_benchmark_choices_hide_held_out_workloads_by_default() -> None:
    manifest = load_benchmark_manifest()

    default = benchmark_choices(manifest, include_test=False)
    revealed = benchmark_choices(manifest, include_test=True)

    assert {entry.split for entry in default} == {"train", "validation"}
    assert default == manifest.tuning
    assert revealed == manifest.tuning + manifest.test


def test_synthetic_benchmark_scenario_uses_the_benchmark_api() -> None:
    scenario = benchmark_scenario("syn-val-bursty")

    assert scenario.trace == build_workload(load_benchmark_manifest().get("syn-val-bursty"))
    assert scenario.info == ScenarioInfo("syn-val-bursty", "synthetic", "validation", 3600.0, 30.0)


def test_azure_benchmark_scenario_needs_an_existing_local_path(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="need the local path"):
        benchmark_scenario("azure-train-129600")
    with pytest.raises(ValueError, match="need the local path"):
        benchmark_scenario("azure-train-129600", azure_csv_path="  ")
    with pytest.raises(FileNotFoundError, match="Azure trace not found"):
        benchmark_scenario("azure-train-129600", azure_csv_path=tmp_path / "missing.txt")


def test_azure_benchmark_scenario_builds_from_the_fixture() -> None:
    scenario = benchmark_scenario("azure-val-734400", azure_csv_path=AZURE_FIXTURE)
    entry = load_benchmark_manifest().get("azure-val-734400")

    assert scenario.trace == build_workload(entry, azure_csv_path=AZURE_FIXTURE)
    assert (scenario.info.source, scenario.info.split) == ("azure", "validation")


@pytest.mark.parametrize(
    ("generator", "parameters"),
    [
        ("steady", {"rate": 40.0}),
        ("diurnal", {"baseline": 50.0, "amplitude": 20.0, "period_seconds": 600.0}),
        ("ramp", {"start_rate": 10.0, "end_rate": 90.0}),
        (
            "spike",
            {
                "baseline": 10.0,
                "spike_rate": 90.0,
                "spike_start_seconds": 300.0,
                "spike_duration_seconds": 60.0,
            },
        ),
        (
            "bursty",
            {
                "baseline": 10.0,
                "burst_rate": 50.0,
                "burst_probability": 0.2,
                "burst_duration_seconds": 60.0,
                "seed": 3,
            },
        ),
    ],
)
def test_custom_scenarios_use_the_existing_generators(
    generator: str, parameters: dict[str, Any]
) -> None:
    scenario = custom_scenario(
        generator, duration_seconds=1200.0, control_interval_seconds=30.0, parameters=parameters
    )

    assert len(scenario.trace) == 40
    assert scenario.info.split == "custom"
    timing = TimingConfig(control_interval_seconds=30.0, episode_duration_seconds=1200.0)
    session = ScenarioSession(scenario, SimulatorConfig(timing=timing), ManagerSpec("manual"))
    assert session.episode_ticks == 40


def test_custom_scenario_validation_comes_from_the_generators() -> None:
    with pytest.raises(ValueError, match="rate must be finite and non-negative"):
        custom_scenario(
            "steady", duration_seconds=600, control_interval_seconds=30, parameters={"rate": -1}
        )
    with pytest.raises(ValueError, match="unknown generator"):
        custom_scenario(
            "sawtooth", duration_seconds=600, control_interval_seconds=30, parameters={}
        )


# --- manual stepping ------------------------------------------------------------


def test_manual_actions_reach_the_real_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    session = make_session()
    calls = spy_on_env_step(session, monkeypatch)

    rows = [session.step_manual(action) for action in (SCALE_UP, HOLD, SCALE_DOWN)]

    assert [action for action, _ in calls] == [SCALE_UP, HOLD, SCALE_DOWN]
    assert [row["requested_action"] for row in rows] == [SCALE_UP, HOLD, SCALE_DOWN]
    assert [row["applied_replica_change"] for row in rows] == [1, 0, -1]
    assert session.tick == 3


def test_history_rows_are_the_environment_info(monkeypatch: pytest.MonkeyPatch) -> None:
    session = make_session()
    calls = spy_on_env_step(session, monkeypatch)

    for action in (SCALE_UP, SCALE_UP, HOLD, SCALE_DOWN):
        session.step_manual(action)

    for row, (_, (_, reward, _, _, info)) in zip(session.history, calls, strict=True):
        assert {key: row[key] for key in info} == info
        assert row["reward"] == reward
        assert row["sla_target_seconds"] == session.config.sla.latency_target_seconds


def test_cumulative_cost_sums_environment_tick_costs() -> None:
    session = make_session()
    for action in (SCALE_UP, SCALE_UP, HOLD, HOLD, SCALE_DOWN):
        session.step_manual(action)

    costs = [row["infrastructure_cost"] for row in session.history]
    assert session.cumulative_cost == sum(costs)
    assert [row["cumulative_cost"] for row in session.history] == list(np.cumsum(costs))


def test_fleet_is_the_next_decision_state_not_the_served_tick() -> None:
    session = make_session(replicas={"startup_delay_seconds": 30.0})  # activates after one tick

    row = session.step_manual(SCALE_UP)

    assert (row["active_replicas"], row["pending_replicas"]) == (2, 1)  # served the tick
    assert session.fleet == session.env.replica_counts
    assert (session.fleet["active_replicas"], session.fleet["pending_replicas"]) == (3, 0)
    assert (row["next_active_replicas"], row["next_pending_replicas"]) == (3, 0)


def test_manual_and_controller_steps_are_not_interchangeable() -> None:
    with pytest.raises(RuntimeError, match="use step_manual"):
        make_session().step_controller()
    with pytest.raises(RuntimeError, match="manager is driving"):
        make_session("random", seed=0).step_manual(SCALE_UP)


# --- controller stepping ----------------------------------------------------------


@pytest.mark.parametrize(
    ("kind", "manager"),
    [("random", {"seed": 5}), ("static", {"target_replicas": 4}), ("threshold", {})],
)
def test_controllers_decide_with_the_runner_decision_contract(
    kind: str, manager: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    session = make_session(kind, replicas={"startup_delay_seconds": 30.0}, **manager)
    controller = session.controller
    assert controller is not None
    seen: list[dict[str, Any]] = []
    expected: list[dict[str, Any]] = []
    real_act = controller.act

    def recording_act(observation: Any, info: Any) -> int:
        seen.append(dict(info))
        expected.append(decision_info(session.env, session.latest_info))
        return real_act(observation, info)

    monkeypatch.setattr(controller, "act", recording_act)
    steps = [session.step_controller() for _ in range(6)]

    assert seen == expected
    assert [row["requested_action"] for row in steps] == [
        row["requested_action"] for row in session.history
    ]


def test_static_controller_sees_replicas_that_activated_at_tick_end() -> None:
    session = make_session("static", target_replicas=3, replicas={"startup_delay_seconds": 30.0})

    actions = [session.step_controller()["requested_action"] for _ in range(4)]

    assert actions == [SCALE_UP, HOLD, HOLD, HOLD]


def test_threshold_diagnostics_are_the_controllers_own() -> None:
    session = make_session("threshold", workload="syn-train-steady-moderate")
    assert session.threshold_decision is None

    session.step_controller()
    session.step_controller()

    controller = session.controller
    assert isinstance(controller, ThresholdController)
    assert session.threshold_decision is controller.last_decision
    assert session.threshold_decision is not None
    assert session.threshold_decision.reason in {"above_high", "at_max", "within_band"}
    assert session.threshold_decision.action == session.history[-1]["requested_action"]


def test_non_threshold_sessions_have_no_threshold_diagnostics() -> None:
    session = make_session("random", seed=1)
    session.step_controller()

    assert session.threshold_decision is None


# --- reset, completion, and invalid builds ------------------------------------------


def test_reset_clears_history_cost_and_controller_state() -> None:
    session = make_session("random", seed=9)
    first = [session.step_controller()["requested_action"] for _ in range(10)]

    session.reset()

    assert (session.tick, session.history, session.cumulative_cost, session.done) == (
        0,
        [],
        0.0,
        False,
    )
    assert session.fleet["active_replicas"] == 2
    again = [session.step_controller()["requested_action"] for _ in range(10)]
    assert again == first  # the random controller restarted from its seed


def test_reset_clears_threshold_diagnostics() -> None:
    session = make_session("threshold")
    session.step_controller()

    session.reset()

    assert session.threshold_decision is None


def test_episode_completes_and_stops_stepping() -> None:
    session = make_session("threshold")

    session.run_to_end()

    assert session.done
    assert session.tick == session.episode_ticks == 120
    assert session.simulated_seconds == 3600.0
    with pytest.raises(RuntimeError, match="episode complete"):
        session.step_controller()
    manual = make_session()
    for _ in range(120):
        manual.step_manual(HOLD)
    with pytest.raises(RuntimeError, match="episode complete"):
        manual.step_manual(HOLD)


@pytest.mark.parametrize(
    "build",
    [
        lambda: ScenarioSession(
            benchmark_scenario("syn-train-spike"),
            v1_config(),
            ManagerSpec("static", target_replicas=9),
        ),
        lambda: ScenarioSession(
            benchmark_scenario("syn-train-spike"),
            v1_config(),
            ManagerSpec("threshold", low_threshold=0.9, high_threshold=0.5),
        ),
        lambda: ScenarioSession(
            custom_scenario(
                "steady", duration_seconds=600, control_interval_seconds=30, parameters={"rate": 5}
            ),
            v1_config(),
            ManagerSpec("manual"),
        ),
    ],
    ids=["static_target_out_of_bounds", "inverted_thresholds", "trace_config_mismatch"],
)
def test_invalid_builds_fail_without_touching_the_active_session(
    build: Callable[[], ScenarioSession],
) -> None:
    active = make_session()
    active.step_manual(SCALE_UP)
    before = (active.tick, active.history, active.cumulative_cost, active.fleet)

    with pytest.raises(ValueError):
        build()

    assert (active.tick, active.history, active.cumulative_cost, active.fleet) == before


# --- presentation helpers --------------------------------------------------------


def _azure_like_session(peak: float, capacity: float) -> ScenarioSession:
    trace = steady_workload(duration_seconds=3600, control_interval_seconds=30, rate=peak)
    info = ScenarioInfo("azure-train-129600", "azure", "train", 3600.0, 30.0)
    config = v1_config(service_capacity_rps=capacity)
    return ScenarioSession(Scenario(trace, info), config, ManagerSpec("manual"))


def test_azure_capacity_note_explains_oversized_capacity_without_changing_anything() -> None:
    session = _azure_like_session(peak=4.9, capacity=50.0)

    note = azure_capacity_note(session)

    assert note is not None
    assert "4.90 RPS" in note and "50 RPS" in note
    assert session.config.replicas.service_capacity_rps == 50.0
    assert max(session.trace.request_rates) == 4.9


def test_azure_capacity_note_is_absent_when_scaling_matters() -> None:
    assert azure_capacity_note(_azure_like_session(peak=4.9, capacity=2.0)) is None
    assert azure_capacity_note(make_session()) is None


@pytest.mark.parametrize(
    ("count", "expected"),
    [(0, ""), (3, "🚗🚗🚗"), (2.6, "🚗🚗🚗"), (180, "🚗" * 10 + " +170")],
)
def test_icon_row_caps_decorative_icons(count: float, expected: str) -> None:
    assert icon_row("🚗", count) == expected


def test_core_and_dashboard_session_do_not_import_streamlit() -> None:
    code = (
        "import sys\n"
        "import scalerl, scalerl.environment, scalerl.controllers, scalerl.dashboard\n"
        "assert 'streamlit' not in sys.modules, 'core import pulled in streamlit'\n"
    )

    subprocess.run([sys.executable, "-c", code], check=True)


# --- Streamlit app (headless AppTest) ------------------------------------------------


@pytest.fixture
def app() -> AppTest:
    return AppTest.from_file(str(APP), default_timeout=60).run()


def session_of(app: AppTest) -> ScenarioSession:
    session: ScenarioSession = app.session_state["scenario_session"]
    return session


def test_app_starts_with_a_default_synthetic_scenario(app: AppTest) -> None:
    assert not app.exception
    assert app.title[0].value == "🏙️ ScaleRL Scenario Lab"
    assert app.radio(key="source").value == "Benchmark v1"
    assert app.selectbox(key="manager").value == "manual"
    assert app.button(key="build")
    session = session_of(app)
    assert (session.tick, session.scenario.source, session.scenario.split) == (
        0,
        "synthetic",
        "train",
    )


def test_app_manual_step_and_reset(app: AppTest) -> None:
    app.button(key="step_up").click().run()

    assert not app.exception
    assert session_of(app).tick == 1
    assert app.metric[0].value == "1 / 120"

    app.button(key="reset_episode").click().run()

    assert session_of(app).tick == 0
    assert session_of(app).history == []


def test_app_history_charts_appear_after_the_first_tick(app: AppTest) -> None:
    assert app.get("vega_lite_chart") == []

    app.button(key="step_up").click().run()
    app.button(key="step_hold").click().run()

    # traffic, queue, scaling, replicas, latency vs SLA, cumulative cost
    assert len(app.get("vega_lite_chart")) == 6


def test_app_builds_and_steps_a_threshold_controller(app: AppTest) -> None:
    app.selectbox(key="manager").set_value("threshold").run()
    app.button(key="build").click().run()
    app.button(key="step").click().run()
    app.button(key="step").click().run()

    assert not app.exception
    session = session_of(app)
    assert (session.manager.kind, session.tick) == ("threshold", 2)
    assert any("reason" in markdown.value for markdown in app.markdown)


def test_app_episode_completion_disables_stepping(app: AppTest) -> None:
    app.selectbox(key="manager").set_value("static").run()
    app.button(key="build").click().run()
    app.button(key="run_to_end").click().run()

    assert not app.exception
    assert session_of(app).done
    assert any("Episode complete" in success.value for success in app.success)
    assert app.button(key="step").disabled


def test_app_custom_synthetic_scenario(app: AppTest) -> None:
    app.radio(key="source").set_value("Custom synthetic").run()
    app.selectbox(key="generator").set_value("ramp").run()
    app.button(key="build").click().run()

    assert not app.exception
    assert session_of(app).scenario.label == "custom ramp"


def test_app_held_out_scenarios_need_an_explicit_reveal(app: AppTest) -> None:
    options = app.selectbox(key="workload").options
    assert not any("HELD-OUT" in option for option in options)

    app.checkbox(key="show_test").check().run()

    assert any("HELD-OUT" in option for option in app.selectbox(key="workload").options)
    assert any(HELD_OUT_WARNING in warning.value for warning in app.warning)


def test_app_invalid_build_keeps_the_running_session(app: AppTest) -> None:
    app.button(key="step_up").click().run()
    running = session_of(app)

    app.number_input(key="min").set_value(5).run()
    app.number_input(key="max").set_value(2).run()
    app.button(key="build").click().run()

    assert not app.exception
    assert any("Scenario not built" in error.value for error in app.error)
    assert session_of(app) is running
    assert session_of(app).tick == 1


def test_app_azure_scenario_reports_a_missing_path(app: AppTest, tmp_path: Path) -> None:
    running = session_of(app)
    app.selectbox(key="workload").set_value("azure-train-129600").run()
    app.text_input(key="azure_path").input(str(tmp_path / "missing.txt")).run()
    app.button(key="build").click().run()

    assert any("Azure trace not found" in error.value for error in app.error)
    assert session_of(app) is running


def test_app_builds_an_azure_scenario_from_a_local_path(app: AppTest) -> None:
    app.selectbox(key="workload").set_value("azure-val-734400").run()
    app.text_input(key="azure_path").input(str(AZURE_FIXTURE)).run()
    app.button(key="build").click().run()

    assert not app.exception
    assert session_of(app).scenario.source == "azure"
    assert any("one replica" in info.value for info in app.info)


def test_random_controller_seeds_are_honored() -> None:
    session = make_session("random", seed=4)
    actions = [session.step_controller()["requested_action"] for _ in range(20)]

    reference = RandomController(seed=4)
    assert actions == [reference.act(np.zeros(9, dtype=np.float32), {}) for _ in range(20)]
