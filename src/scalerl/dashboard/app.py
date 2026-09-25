"""Streamlit Scenario Lab City View.

Launch with ``python -m scalerl.dashboard`` or
``streamlit run src/scalerl/dashboard/app.py`` (needs the ``dashboard`` extra).

Rendering and widgets only; the simulation runs in ``ScenarioSession``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pandas as pd
import streamlit as st

from scalerl.benchmarks import AzureWorkload, load_benchmark_manifest
from scalerl.dashboard.session import (
    ACTION_LABELS,
    CUSTOM_GENERATORS,
    DEFAULT_AZURE_PATH,
    HELD_OUT_WARNING,
    ManagerKind,
    ManagerSpec,
    Scenario,
    ScenarioSession,
    azure_capacity_note,
    benchmark_choices,
    benchmark_scenario,
    custom_scenario,
    icon_row,
)
from scalerl.environment import ReplicaConfig, SimulatorConfig, SlaConfig, TimingConfig
from scalerl.environment.gym_env import HOLD, SCALE_DOWN, SCALE_UP

SESSION_KEY = "scenario_session"
ERROR_KEY = "scenario_build_error"
MANAGERS: dict[ManagerKind, str] = {
    "manual": "🕹️ Manual",
    "random": "🎲 Random",
    "static": "📌 Static",
    "threshold": "🌡️ Threshold",
}
SPLIT_LABELS = {
    "train": "TRAIN",
    "validation": "VALIDATION",
    "test": "HELD-OUT TEST",
    "custom": "CUSTOM",
}
ACTION_DELTAS = {SCALE_DOWN: -1, HOLD: 0, SCALE_UP: 1}

SOURCE_HELP = (
    "Benchmark v1 uses the frozen, versioned workload suite from Issue #18 so experiments "
    "can be reproduced. Custom synthetic creates an exploratory traffic pattern from the "
    "parameters you choose and is not part of the frozen benchmark."
)
WORKLOAD_HELP = (
    "A workload is the request-rate pattern the simulated service receives over time. "
    "Benchmark workloads are preconfigured and frozen; selecting one does not change its "
    "traffic parameters."
)
WORKLOAD_SHAPE_HELP = {
    "steady": "Steady: a constant request rate for the whole episode.",
    "diurnal": "Seasonal/diurnal: demand rises and falls smoothly in a repeating pattern.",
    "ramp": "Ramp: demand gradually increases or decreases over the episode.",
    "spike": "Spike: normal traffic interrupted by a sudden temporary surge.",
    "bursty": "Bursty: random bursts around a baseline request rate.",
}
MANAGER_HELP = (
    "The manager/controller chooses one action each tick: scale down, hold, or scale up. "
    "Manual lets you choose; Random is a sanity-check baseline; Static targets a fixed fleet; "
    "Threshold reacts to utilization."
)

# Starting values for custom generators (same shapes as the v1 training scenarios).
CUSTOM_DEFAULTS: dict[str, dict[str, float | int]] = {
    "steady": {"rate": 100.0},
    "diurnal": {
        "baseline": 150.0,
        "amplitude": 100.0,
        "period_seconds": 3600.0,
        "phase_seconds": 0.0,
    },
    "ramp": {"start_rate": 40.0, "end_rate": 360.0},
    "spike": {
        "baseline": 80.0,
        "spike_rate": 420.0,
        "spike_start_seconds": 1500.0,
        "spike_duration_seconds": 300.0,
    },
    "bursty": {
        "baseline": 90.0,
        "burst_rate": 250.0,
        "burst_probability": 0.08,
        "burst_duration_seconds": 90.0,
        "noise_std": 10.0,
        "seed": 101,
    },
}

BuildRequest = Callable[[], ScenarioSession]


def main() -> None:
    st.set_page_config(page_title="ScaleRL Scenario Lab", page_icon="🏙️", layout="wide")
    request, build_clicked = _sidebar()
    if build_clicked or SESSION_KEY not in st.session_state:
        _build(request)

    error = st.session_state.get(ERROR_KEY)
    if error:
        st.sidebar.error(f"Scenario not built: {error}")
    session: ScenarioSession | None = st.session_state.get(SESSION_KEY)
    if session is None:
        st.title("🏙️ ScaleRL Scenario Lab")
        st.info("Fix the scenario settings in the sidebar and press **Build / Reset Scenario**.")
        return
    _render(session)


# --- sidebar: scenario form ---------------------------------------------------


def _sidebar() -> tuple[BuildRequest, bool]:
    bar = st.sidebar
    bar.header("Scenario")
    source = bar.radio(
        "Source", ["Benchmark v1", "Custom synthetic"], key="source", help=SOURCE_HELP
    )
    duration, interval = _timing(source)
    build_scenario = (
        _benchmark_form() if source == "Benchmark v1" else _custom_form(duration, interval)
    )

    bar.header("Simulator")
    defaults = SimulatorConfig().replicas
    min_replicas = bar.number_input(
        "Min replicas",
        1,
        100,
        defaults.min_replicas,
        key="min",
        help="Lowest replica count the simulated deployment is allowed to have.",
    )
    initial_replicas = bar.number_input(
        "Initial replicas",
        1,
        100,
        defaults.initial_replicas,
        key="initial",
        help="Number of active replicas available when the episode begins.",
    )
    max_replicas = bar.number_input(
        "Max replicas",
        1,
        100,
        defaults.max_replicas,
        key="max",
        help="Highest replica count the controller is allowed to request.",
    )
    startup_delay = bar.number_input(
        "Startup delay (s)",
        0.0,
        3600.0,
        defaults.startup_delay_seconds,
        30.0,
        key="delay",
        help=(
            "Simulated time between requesting a new replica and that replica becoming "
            "able to serve traffic."
        ),
    )
    capacity = bar.number_input(
        "Service capacity per replica (RPS)",
        0.01,
        10_000.0,
        defaults.service_capacity_rps,
        key="capacity",
        help=(
            "Maximum request rate one active replica can process. Demand above total "
            "active capacity builds a queue."
        ),
    )
    cost = bar.number_input(
        "Cost per replica-hour ($)",
        0.0,
        1000.0,
        defaults.cost_per_hour,
        0.05,
        key="cost",
        help="Simulated infrastructure price for keeping one replica active for one hour.",
    )
    sla = bar.number_input(
        "SLA p95 target (s)",
        0.01,
        60.0,
        SimulatorConfig().sla.latency_target_seconds,
        0.05,
        key="sla",
        help=(
            "Maximum acceptable p95 latency. A completed tick violates the SLA when p95 "
            "latency exceeds this target."
        ),
    )

    bar.header("Manager")
    kind: ManagerKind = bar.selectbox(
        "City manager",
        list(MANAGERS),
        format_func=MANAGERS.__getitem__,
        key="manager",
        help=MANAGER_HELP,
    )
    manager = _manager_form(
        kind,
        min_replicas=int(min_replicas),
        initial_replicas=int(initial_replicas),
        max_replicas=int(max_replicas),
    )

    def request() -> ScenarioSession:
        # All validation happens here, inside the guarded build.
        config = SimulatorConfig(
            timing=TimingConfig(
                control_interval_seconds=interval, episode_duration_seconds=duration
            ),
            replicas=ReplicaConfig(
                min_replicas=int(min_replicas),
                initial_replicas=int(initial_replicas),
                max_replicas=int(max_replicas),
                startup_delay_seconds=float(startup_delay),
                service_capacity_rps=float(capacity),
                cost_per_hour=float(cost),
            ),
            sla=SlaConfig(latency_target_seconds=float(sla)),
        )
        return ScenarioSession(build_scenario(), config, manager)

    clicked = bar.button("Build / Reset Scenario", type="primary", key="build")
    bar.caption("Form changes apply only when you build; the running episode is not changed.")
    return request, clicked


def _benchmark_form() -> Callable[[], Scenario]:
    bar = st.sidebar
    manifest = load_benchmark_manifest()
    show_test = bar.checkbox(
        "Show held-out final-evaluation scenarios",
        key="show_test",
        help=(
            "Reveals the frozen test split. These scenarios are reserved for final "
            "evaluation and must not be used to tune controllers, rewards, models, or "
            "simulator settings."
        ),
    )
    if show_test:
        bar.warning(HELD_OUT_WARNING)
    entries = {entry.id: entry for entry in benchmark_choices(manifest, include_test=show_test)}
    workload_id: str = bar.selectbox(
        "Workload",
        list(entries),
        format_func=lambda id_: f"{id_} · {SPLIT_LABELS[entries[id_].split]}",
        key="workload",
        help=WORKLOAD_HELP,
    )
    entry = entries[workload_id]
    bar.caption(
        f"Source: {entry.source} · split: {SPLIT_LABELS[entry.split]} · "
        f"{entry.duration_seconds:g} s at {entry.control_interval_seconds:g} s ticks"
    )
    if isinstance(entry, AzureWorkload):
        bar.caption("Real Azure Functions trace window with frozen start/duration parameters.")
    else:
        shape = getattr(entry, "generator", "")
        bar.caption(
            f"{WORKLOAD_SHAPE_HELP.get(shape, 'Frozen synthetic traffic pattern.')} "
            "Its parameters are already fixed by Benchmark v1."
        )
    azure_path: str | None = None
    if isinstance(entry, AzureWorkload):
        azure_path = bar.text_input(
            "Local Azure trace path",
            DEFAULT_AZURE_PATH,
            key="azure_path",
            help="Path to the extracted Azure Functions 2021 trace; see data/README.md.",
        )
    return lambda: benchmark_scenario(workload_id, azure_csv_path=azure_path, manifest=manifest)


def _custom_form(duration: float, interval: float) -> Callable[[], Scenario]:
    bar = st.sidebar
    generator: str = bar.selectbox(
        "Generator",
        list(CUSTOM_GENERATORS),
        key="generator",
        help="Choose the shape of the synthetic request-rate pattern you want to create.",
    )
    bar.caption(WORKLOAD_SHAPE_HELP[generator])
    parameters: dict[str, Any] = {}
    for name, default in CUSTOM_DEFAULTS[generator].items():
        label = name.replace("_", " ")
        key = f"custom_{generator}_{name}"
        if isinstance(default, int):
            parameters[name] = int(bar.number_input(label, value=default, step=1, key=key))
        else:
            parameters[name] = float(bar.number_input(label, value=default, key=key))
    return lambda: custom_scenario(
        generator,
        duration_seconds=duration,
        control_interval_seconds=interval,
        parameters=parameters,
    )


def _timing(source: str) -> tuple[float, float]:
    """Return ``(episode_duration_seconds, control_interval_seconds)`` from the form."""
    if source == "Benchmark v1":
        st.sidebar.caption("Benchmark v1 timing is fixed: 3600 s episodes, 30 s ticks.")
        return 3600.0, 30.0
    duration = st.sidebar.number_input(
        "Episode duration (s)", 30.0, 86_400.0, 3600.0, 30.0, key="custom_duration"
    )
    interval = st.sidebar.number_input(
        "Control interval (s)", 1.0, 3600.0, 30.0, 5.0, key="custom_interval"
    )
    return float(duration), float(interval)


def _manager_form(
    kind: ManagerKind, *, min_replicas: int, initial_replicas: int, max_replicas: int
) -> ManagerSpec:
    bar = st.sidebar
    if kind == "random":
        seed = int(
            bar.number_input(
                "Seed",
                0,
                2**31 - 1,
                0,
                key="random_seed",
                help="Makes the Random controller reproduce the same action sequence when reset.",
            )
        )
        return ManagerSpec(kind, seed=seed)
    if kind == "static":
        target_default = min(max(initial_replicas, min_replicas), max_replicas)
        target = int(
            bar.number_input(
                "Target replicas",
                min_replicas,
                max_replicas,
                target_default,
                key="target",
                help=(
                    "Fixed total replica target. Static does not react to traffic; it "
                    "moves toward this fleet size and then holds."
                ),
            )
        )
        return ManagerSpec(kind, target_replicas=target)
    if kind == "threshold":
        low = float(
            bar.number_input(
                "Low utilization threshold",
                0.0,
                0.99,
                0.3,
                0.05,
                key="low",
                help=(
                    "When utilization falls below this value, Threshold tends to scale "
                    "down if replica bounds allow it."
                ),
            )
        )
        high = float(
            bar.number_input(
                "High utilization threshold",
                0.01,
                0.99,
                0.8,
                0.05,
                key="high",
                help=(
                    "When utilization rises above this value, Threshold tends to scale "
                    "up if replica bounds allow it."
                ),
            )
        )
        bar.caption("Replica bounds follow the simulator min/max.")
        return ManagerSpec(kind, low_threshold=low, high_threshold=high)
    return ManagerSpec(kind)


def _build(request: BuildRequest) -> None:
    """Replace the session only if the new one builds; otherwise keep the old one."""
    try:
        session = request()
    except (ValueError, TypeError, FileNotFoundError) as error:
        st.session_state[ERROR_KEY] = str(error)
        return
    st.session_state[SESSION_KEY] = session
    st.session_state.pop(ERROR_KEY, None)


# --- main page ----------------------------------------------------------------


def _render(session: ScenarioSession) -> None:
    info = session.scenario
    st.title("🏙️ ScaleRL Scenario Lab")
    st.caption(
        f"**{info.label}** · {SPLIT_LABELS[info.split]} · {info.source} · "
        f"{info.duration_seconds:g} s at {info.control_interval_seconds:g} s ticks"
    )
    if info.split == "test":
        st.warning(HELD_OUT_WARNING)
    note = azure_capacity_note(session)
    if note:
        st.info(note)

    tick, time, manager = st.columns(3)
    tick.metric("Tick", f"{session.tick} / {session.episode_ticks}")
    minutes, seconds = divmod(int(session.simulated_seconds), 60)
    time.metric("Simulated time", f"{minutes:02d}:{seconds:02d}")
    manager.metric("Manager", MANAGERS[session.manager.kind])

    city, panel = st.columns([3, 2])
    with city:
        _render_city(session)
    with panel:
        _render_manager(session)
    _render_history(session)
    _render_guidance()


def _render_city(session: ScenarioSession) -> None:
    last = session.history[-1] if session.history else None
    fleet = session.fleet
    with st.container(border=True):
        st.subheader("City View")
        st.caption(
            "Shops show the fleet **now** (what the next decision applies to). Traffic, "
            "queue, latency, and cost are from the **last completed tick**."
        )
        shops = "☕" * min(fleet["active_replicas"], 15) + "🏗️" * min(fleet["pending_replicas"], 15)
        st.markdown(f"### {shops or '—'}")
        active, pending, terminating = st.columns(3)
        active.metric("☕ Open shops (active)", fleet["active_replicas"])
        pending.metric("🏗️ Being built (pending)", fleet["pending_replicas"])
        if last and last["terminating_replicas"]:
            terminating.metric("🚧 Closing (last tick)", last["terminating_replicas"])

        if last is None:
            st.info("No customers yet: step the city to start the first tick.")
            return
        st.markdown(f"🚗 {icon_row('🚗', last['request_rate'], cap=10)}")
        st.markdown(f"👥 {icon_row('👥', last['queued_requests'], cap=10)}")
        traffic, queue, latency, sla = st.columns(4)
        traffic.metric("🚗 Traffic", f"{last['request_rate']:.1f} RPS")
        queue.metric("👥 Queue", f"{last['queued_requests']:.0f} req")
        latency.metric("⏱ p95 latency", f"{last['p95_latency_seconds']:.3f} s")
        target = session.config.sla.latency_target_seconds
        sla.metric(
            "SLA", "🚨 violated" if last["sla_violated"] else "✅ met", f"target {target:g} s"
        )
        tick_cost, total_cost, reward = st.columns(3)
        tick_cost.metric("💵 Tick cost", f"${last['infrastructure_cost']:.4f}")
        total_cost.metric("💵 Cumulative cost", f"${session.cumulative_cost:.4f}")
        reward.metric("Reward (tick)", f"{last['reward']:.3f}")


def _render_manager(session: ScenarioSession) -> None:
    last = session.history[-1] if session.history else None
    with st.container(border=True):
        st.subheader(f"👔 {MANAGERS[session.manager.kind]}")
        if last:
            requested, applied = st.columns(2)
            requested.metric("Decision", ACTION_LABELS[last["requested_action"]])
            applied.metric("Applied change", f"{last['applied_replica_change']:+d}")
            if last["requested_action"] != HOLD and last["applied_replica_change"] == 0:
                st.caption("The request hit a replica bound, so nothing changed.")
        decision = session.threshold_decision
        if decision is not None:
            utilization = "n/a" if decision.utilization is None else f"{decision.utilization:.2f}"
            st.markdown(
                f"utilization **{utilization}** · desired **{decision.desired_replicas}** · "
                f"decision **{ACTION_LABELS[decision.action]}** · reason `{decision.reason}`"
            )

        if session.done:
            st.success("Episode complete. Build or reset to run again.")
        if session.controller is None:
            down, hold, up = st.columns(3)
            for column, action, label, key in (
                (down, SCALE_DOWN, "↓ Scale down", "step_down"),
                (hold, HOLD, "— Hold", "step_hold"),
                (up, SCALE_UP, "↑ Scale up", "step_up"),
            ):
                column.button(
                    label, key=key, disabled=session.done, on_click=_manual_step, args=(action,)
                )
        else:
            step, run = st.columns(2)
            step.button("▶ Step", key="step", disabled=session.done, on_click=_controller_step)
            run.button(
                "⏭ Run to end", key="run_to_end", disabled=session.done, on_click=_run_to_end
            )
        st.button("↺ Reset episode", key="reset_episode", on_click=_reset_episode)


def _render_history(session: ScenarioSession) -> None:
    if not session.history:
        return
    frame = pd.DataFrame(session.history).set_index("tick")
    frame["requested_change"] = frame["requested_action"].map(ACTION_DELTAS)
    st.subheader("History")
    left, right = st.columns(2)
    with left:
        st.caption("Traffic (RPS)")
        st.line_chart(frame[["request_rate"]])
        st.caption("Queue depth (requests)")
        st.line_chart(frame[["queued_requests"]])
        st.caption("Scaling: requested vs applied replica change")
        st.bar_chart(frame[["requested_change", "applied_replica_change"]], stack=False)
    with right:
        st.caption("Replicas serving each tick")
        st.line_chart(frame[["active_replicas", "pending_replicas"]])
        st.caption("p95 latency vs SLA target (s)")
        st.line_chart(frame[["p95_latency_seconds", "sla_target_seconds"]])
        st.caption("Cumulative infrastructure cost ($)")
        st.line_chart(frame[["cumulative_cost"]])


def _render_guidance() -> None:
    with st.expander("❓ How to read this lab"):
        st.markdown(
            """
**Core terms**

- **Traffic / RPS:** incoming requests per second.
- **Active replicas:** service capacity available right now.
- **Pending replicas:** replicas that were requested but are still inside startup delay.
- **Queue:** requests waiting because current active capacity was not enough.
- **p95 latency:** a tail-latency proxy; roughly, 95% of requests are at or below this value.
- **SLA:** violated when p95 latency is above the configured target.
- **Cumulative cost:** simulated infrastructure cost accumulated across completed ticks.
- **Requested action vs applied change:** what the manager asked for versus what actually
  changed after replica bounds were enforced.
- **Reward:** a combined RL signal and secondary diagnostic; do not interpret it instead
  of latency, SLA, queue, and cost.

**Typical cause/effect**

`traffic rises → utilization/queue rise → manager may scale up → startup delay →`
`pending replica becomes active → queue/latency may recover → cost increases`
"""
        )

    with st.expander("🧪 MLflow and this lab"):
        st.markdown(
            """
The **Scenario Lab is an interactive sandbox**. Clicking Step, Scale up, or Run to end
 does **not** automatically create an MLflow run.

**MLflow (#17)** records reproducible `train`, `tune`, and `evaluate` experiment runs
produced by the experiment pipeline: their configuration, workload, seed, metrics, Git
version, and artifacts.

A later **Results Explorer (#39)** will let this dashboard browse those saved MLflow runs.

So it is normal for MLflow to stay empty while you only experiment here. Final
benchmark/report numbers should come from MLflow-tracked runs, not from an ad-hoc City
View session.
"""
        )


# --- button callbacks (run before the page re-renders) ---------------------------


def _session() -> ScenarioSession:
    session: ScenarioSession = st.session_state[SESSION_KEY]
    return session


def _manual_step(action: int) -> None:
    if not _session().done:
        _session().step_manual(action)


def _controller_step() -> None:
    if not _session().done:
        _session().step_controller()


def _run_to_end() -> None:
    _session().run_to_end()


def _reset_episode() -> None:
    _session().reset()


main()
