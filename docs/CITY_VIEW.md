# Scenario Lab: City View

The City View is an interactive Streamlit page for watching one ScaleRL episode tick by tick: traffic arrives, a manager scales the fleet, replicas start up, queues build or drain, and latency, SLA, and cost follow.

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

Icons are decorative and capped (for example `🚗🚗🚗🚗🚗🚗🚗🚗🚗🚗 +170`); the metrics next to them show exact values.

**Two moments are shown on purpose.** The shops show the fleet **now**, i.e. what the next decision applies to (`env.replica_counts`). Traffic, queue, latency, SLA, and cost are from the **last completed tick** (that step's `info`). A replica requested earlier can finish starting at the end of a tick, so these can legitimately differ.

## Building a scenario

Choose settings in the sidebar, then press **Build / Reset Scenario**. Changing the form never alters the running episode until you build; if the new settings are invalid, the error is shown and the current episode keeps running unchanged.

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
| 🌡️ Threshold | `ThresholdController` with low/high utilization thresholds and **cooldown ticks** (after a real scaling change, hold this many decisions; 0 disables it); shows its own decision diagnostics (utilization, desired replicas, action, reason, and remaining cooldown) |

Controller managers offer **▶ Step** (one tick) and **⏭ Run to end**. They decide exactly as the evaluation runner does: `controller.act(observation, decision_info(env, info))`. **↺ Reset episode** restarts the same scenario from tick 0. When the episode reaches its last tick, stepping stops and the final city and history stay visible.

## History

After each tick the page charts traffic, replicas serving each tick (active and pending), queue depth, p95 latency against the SLA target, requested versus applied replica change, and cumulative infrastructure cost. Each history row is the step's raw environment `info` plus a few presentation fields (SLA target, cumulative cost, next-decision replica counts).

## Relationship to MLflow

| Tool | Purpose |
|---|---|
| MLflow ([MLOPS.md](MLOPS.md)) | experiment lineage, params, metrics, artifacts |
| City View | interactive understanding of one simulation |
| Results Explorer (#39, planned) | browsing stored experiment runs |

The City View does not log to MLflow and is not an experiment database. Numbers used in results must come from tracked MLflow runs.
