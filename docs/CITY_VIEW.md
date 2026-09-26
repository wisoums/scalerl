# Scenario Lab: City View

The City View is an interactive Streamlit page for watching one ScaleRL episode tick by tick, either deliberately (Inspect) or as automatic playback (Live City): traffic arrives, a manager scales the fleet, replicas start up, queues build or drain, and latency, SLA, and cost follow.

It is a **view onto the real simulator**. Every value comes from `AutoscalingEnv` and the existing controllers; the dashboard never re-implements queueing, replica lifecycle, latency, cost, reward, workload, or controller logic.

## Install and launch

```bash
pip install -e ".[dashboard]"
python -m scalerl.dashboard
```

or, from the repository root:

```bash
streamlit run src/scalerl/dashboard/app.py
```

Streamlit is optional: `scalerl`, its environment, controllers, workloads, and benchmarks never import it.

## The city

| City | Simulator |
|---|---|
| 🚗 cars / customers arriving | incoming requests (RPS) |
| ☕ open shop | active replica |
| 🏗️ shop being built | pending replica (still in startup delay) |
| 🚧 shop closing | terminating replica |
| 👥 customer line | request queue |
| ⏱ waiting time | p95 latency proxy |
| 🚨 late customers | SLA violation |
| 💵 operating shops | infrastructure cost |
| 👔 city manager | autoscaling controller |
| ↓ — ↑ | scale down / hold / scale up |

The city reads top to bottom as a flow: 🚗 incoming traffic → ☕🏗️ service → 👥 waiting queue → ⏱ latency / 🚨 SLA → 💵 cost.

**Traffic vs queue.** **🚗 Incoming traffic** is the NEW requests arriving during the last completed tick, in requests per second. **👥 Waiting queue** is requests that ALREADY arrived but could not yet be processed. They are different quantities with different units, and each has its own exact metric.

Icons are decorative and capped (for example `🚗🚗🚗🚗🚗🚗🚗🚗🚗🚗 +170` above **180 RPS**); one icon is never one request, and the metric next to it is the authoritative value. Pending replicas are shown as a truthful count of starting shops, without per-replica countdowns (the simulator exposes no public startup-progress API).

**Two moments are shown on purpose.** The shops show the fleet **now**, i.e. what the next decision applies to (`env.replica_counts`). Traffic, queue, latency, SLA, and cost are from the **last completed tick** (that step's `info`). A replica requested earlier can finish starting at the end of a tick, so these can legitimately differ.

## Building a scenario

Choose settings in the sidebar, then press **Apply settings & restart**. This applies the sidebar configuration and rebuilds the scenario from tick 0. Changing the form alone never alters the running episode; if the new settings are invalid, the error is shown and the current episode keeps running unchanged.

**Reset episode** is intentionally different: it restarts the already-built scenario from tick 0 without applying unapplied sidebar edits.

### Benchmark v1

Workloads come from the frozen benchmark manifest (`scalerl.benchmarks`), showing each workload's ID, source, split, duration, and interval. Benchmark timing is fixed at 3600 s episodes with 30 s ticks.

The picker lists **train** and **validation** workloads. Held-out **test** workloads appear only after ticking **Show held-out final-evaluation scenarios**, which displays this warning:

> Held-out test workloads are for final evaluation only. Do not use this view to tune controller, model, reward, or simulator settings.

This is an experiment-integrity reminder, not a security boundary.

### Azure workloads

Azure benchmark workloads read the locally extracted trace. Enter its path (default `data/raw/AzureFunctionsInvocationTraceForTwoWeeksJan2021.txt`; see [data/README.md](../data/README.md) for setup). The file is never uploaded through the browser, and it is read once when the scenario is built, not on every interaction.

The v1 Azure windows are low-demand (train/validation mean 0.45–2.29 RPS, peak 4.9 RPS). With the default 50 RPS per replica, one replica serves everything, so the page shows a note that scaling may look uneventful. Demand is never rescaled and capacity is never changed automatically. Lowering service capacity in the sidebar is exploratory only; formal calibration follows the train/validation-only rules in [EXPERIMENTS.md](EXPERIMENTS.md) and is recorded in MLflow.

### Custom synthetic playground

Build exploratory traffic with the existing generators (steady, diurnal, ramp, spike, bursty), their parameters, and a custom episode duration and control interval. Invalid values are rejected by the generators' own validation.

### Simulator settings

Min, initial, and max replicas; startup delay; service capacity per replica; cost per replica-hour; and SLA p95 target.

## Managers

| Manager | Behavior |
|---|---|
| 🕹️ Manual | **↓ Scale down**, **— Hold**, **↑ Scale up**: each click is exactly one simulation tick |
| 🎲 Random | `RandomController` with a chosen seed |
| 📌 Static | `StaticController` holding a target replica count |
| 🔮 Predictive | `PredictiveController`: a linear-trend forecast of the last **history window** completed request rates (default 4), sized with a **target utilization** (default 0.8); bounds, capacity, startup delay, and tick length come from the simulator. Capacity is sized for **effective sizing demand = forecast arrivals + backlog recovery** (waiting queue ÷ tick length), and it never scales down while requests are waiting (#63). Shows observed demand, forecast arrivals, waiting queue, backlog recovery, effective sizing demand, horizon (ticks and seconds), desired replicas, decision, and reason (`queue_recovery`: scaled up only because of the queue; `backlog_hold`: scale-down blocked by the queue). It forecasts from completed traffic only, so it cannot anticipate a spike with no prior trend; queue recovery helps only after a queue has built |
| 🌡️ Threshold | `ThresholdController` with low/high utilization thresholds and **cooldown ticks** (after a real scaling change, hold this many decisions; 0 disables it); shows its own decision diagnostics (utilization, desired replicas, action, reason, and remaining cooldown) |

Controller managers offer **▶ Step** (one tick) and **⏭ Run to end**. They decide exactly as the evaluation runner does: `controller.act(observation, decision_info(env, info))`. **↺ Reset episode** restarts the same scenario from tick 0. When the episode reaches its last tick, stepping stops and the final city and history stay visible.

Manager panels use only diagnostics the controller really provides. Random and Static show the action (Static also its target replicas), with no invented reason. Threshold shows decision, utilization, desired replicas, reason, and cooldown remaining. Predictive keeps **forecast arrivals** and **effective sizing demand** separate. Diagnostics explain the decision that produced the last completed tick, so they use what was known before that tick ran (for example the queue of the tick before).

## Inspect vs Live City

A compact control toolbar offers two ways to watch the **same** `ScenarioSession`: the same controller, environment, and history. Switching modes never rebuilds or resets the city. The page opens in Inspect mode, paused, at 1x, and never advances on its own.

Conceptually:

```text
Mode [ Inspect | Live City ]   playback controls   Speed [0.5x 1x 2x 5x]
```

The toolbar is intentionally compact so City View appears near the top of the page rather than below a large controls card.

| Mode | Controls |
|---|---|
| 🔍 Inspect | Manual **↓ / — / ↑**, or controller **▶ Step** and **⏭ Run to end**; **↺ Reset episode** |
| 🏙️ Live City | **▶ Play**, **⏸ Pause**, **⏯ Step once** (while paused), speed **0.5x / 1x / 2x / 5x**, **↺ Reset episode** |

- **Discrete ticks only.** Every Live City frame is one real `ScenarioSession.step_controller()` call, exactly as Step does. Nothing is interpolated between ticks and no fake queue or latency states are drawn; charts plot real history, one row per tick. Animation does not make ScaleRL a continuous-time simulator.
- **Speed is presentation cadence only.** The UI summarizes this as `1 frame = <tick length> simulated · speed changes display cadence only`. At 1x, playback shows about one simulator tick per real second; 0.5x ≈ one every 2 s, 2x ≈ one every 0.5 s, and 5x ≈ one every 0.2 s. Speed never changes the control interval, workload, controller, startup delay, simulated clock, or reward. In Benchmark v1 each frame is still 30 simulated seconds. Changing speed keeps the session and history and only reschedules the next frame.
- **Stopping.** Pause takes effect on the next rerun. **Reset episode** pauses and returns the currently built scenario to tick 0 without applying sidebar edits. **Apply settings & restart** pauses and rebuilds from the sidebar configuration; it also pauses when the new build fails validation, leaving the old city visible but paused. Reaching the last tick pauses automatically, keeps the final city visible, and disables Play. Leaving Live City for Inspect also pauses. **⏭ Run to end** exists only in Inspect mode.
- **Manual has no autoplay.** Live City needs a controller-driven manager; with Manual, Play is disabled with an explanation and no HOLD actions are invented. Choose manual actions in Inspect mode.
- **How it cannot double-step.** The live region is a Streamlit fragment re-run by `st.fragment(run_every=period)` only while playing. A pure `PlaybackState` (`scalerl.dashboard.playback`, no Streamlit import) decides whether a tick is due, using monotonic time. One due timer run advances at most one tick, with no catch-up bursts. Ordinary reruns from clicks and widget changes never step. The fragment always reads the current session from Streamlit state, so a replaced city is never advanced by an old timer.
- **Accessibility.** Exact numbers, status text (for example `LIVE CITY • PLAYING • 2x` or `LIVE CITY • PAUSED`), and the SLA label never rely on motion; nothing flashes, and Inspect mode stays the non-animated alternative.

**Robustness dynamics (#65).** The Scenario Lab runs the nominal simulator; capacity jitter and telemetry delay are not exposed as Lab controls yet. Everything the City View shows comes from physical step `info`, i.e. what actually happened. If delayed telemetry is shown in a later issue, it must be labeled as the controller's view, distinct from the physical city.

## History

After each tick the page charts traffic, replicas serving each tick (active and pending), queue depth, p95 latency against the SLA target, requested versus applied replica change, and cumulative infrastructure cost. Each history row is the step's raw environment `info` plus a few presentation fields (SLA target, cumulative cost, next-decision replica counts).

## Relationship to MLflow

| Tool | Purpose |
|---|---|
| MLflow ([MLOPS.md](MLOPS.md)) | experiment lineage, params, metrics, artifacts |
| City View | interactive understanding of one simulation |
| Results Explorer (#39, planned) | browsing stored experiment runs |

The City View, including Live City playback, does not log to MLflow and is not an experiment database; it is interactive Scenario Lab exploration. Numbers used in results must come from tracked MLflow runs.
