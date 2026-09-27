# Architecture

ScaleRL is an experimental autoscaling platform for comparing traditional controllers with deep reinforcement-learning policies under identical workload/configuration contracts.

## System layers

1. **Workload sources** — synthetic generators and recorded-trace loaders both produce `WorkloadTrace`.
2. **Cloud simulator** — models replicas, startup delay, queueing, service capacity, latency, and infrastructure cost.
3. **Gymnasium environment** — exposes observations, actions, transitions, truncation, raw metrics, and reward.
4. **Controllers** — random, static, threshold/target-tracking, predictive, optional tabular Q-learning, DQN, and PPO.
5. **Evaluation harness** — runs controllers against identical workload/config/seed inputs and records raw system metrics.
6. **MLOps layer** — MLflow tracks run metadata/metrics/artifacts; Docker standardizes training/runtime; CI validates package/container/training plumbing.
7. **Serving/demo layer** — later phases expose policies through FastAPI and domain-specific dashboards.

## Workload boundary

The environment consumes `WorkloadTrace`, not a specific generator/source. This keeps synthetic workloads and real Azure trace slices interchangeable for controller/evaluation code.

## Initial MDP

### Observation

The v1 observation (`AutoscalingEnv`) is a `Box` in `[0, 1]` describing the state after the last completed tick. With `h = observation.traffic_history_ticks` (v1: 4) and `k` pending-readiness buckets:

| Index | Feature |
|---|---|
| 0 … h−1 | recent demand pressure, newest first: `0` = latest completed tick, `h−1` = oldest (zero until enough ticks have run); each `rate / (rate + max_replicas * service_capacity_rps)` |
| h | utilization of active capacity |
| h+1 | queue pressure: `queued / (queued + max-fleet capacity per tick)` |
| h+2 | latency pressure: `p95 / (p95 + latency_target)` |
| h+3 | active replicas / `max_replicas` |
| h+4 | tick cost / cost of `max_replicas` for one tick |
| h+5 | episode progress |
| h+6 … h+6+k−1 | pending replicas that activate after 1 … k more ticks, each / `max_replicas` |

`AutoscalingEnv.observation_features` names every position. The observation size is config-dependent:

```text
observation_size = h + 6 + k
k = ceil(startup_delay_seconds / control_interval_seconds)   # 0 when startup delay is 0
```

The v1 default (`h = 4`, 60 s startup delay, 30 s ticks, so `k = 2`) has 12 features. `h = 1` reproduces the earlier `7 + k` layout. The traffic history only ever contains demand consumed by completed ticks; the next workload value is never observed.

Bucketing pending replicas by readiness keeps the observation Markov when startup delay spans several ticks. For a fixed `SimulatorConfig` the shape never changes across workloads, seeds, resets, actions, or episodes; changing the traffic history length, startup delay, or control interval may change it.

**Model compatibility:** a trained DQN/PPO policy is only directly usable with an environment whose observation/action spaces match its training environment. MLflow model/run metadata records this contract; training/serving code must reject obvious incompatibilities rather than silently loading them.

Features the simulator does not model (such as CPU utilization) are intentionally not fabricated.

### Action space (versioned action contracts, #79)

The environment always takes an integer **action code** from a Gymnasium `Discrete` space. What a code means is the versioned contract `SimulatorConfig.action.semantics`, encoded in one place (`scalerl.environment.actions.ActionContract`):

| Contract | Space | Codes |
|---|---|---|
| `delta-v1` (default; every run and model before #79) | `Discrete(3)` | `0` scale down, `1` hold, `2` scale up; their replica-count **effects** are `-1/0/+1` (effects are not codes: `step(-1)` is invalid) |
| `desired-replicas-v1` | `Discrete(max_replicas - min_replicas + 1)` | code `c` requests `min_replicas + c` committed replicas (default: codes `0..9` -> targets `1..10`) |

Either way the code becomes a **requested target** for committed capacity (`active + pending`). The environment starts or cancels `|target - committed|` replicas in that one step through the normal lifecycle: new replicas start pending and wait out the startup delay, and reductions cancel the newest pending replicas before terminating active ones. Targets are clipped to the configured minimum and maximum. `info` reports `requested_action` (the code), `requested_replica_target`, and `applied_replica_change` (the signed change, which can exceed one replica under `desired-replicas-v1`); all three are current control-plane facts, never delayed telemetry.

Replica counts stay discrete under both contracts. There are no continuous actions: DQN acts on either `Discrete` space directly and PPO uses its categorical policy. `SimulatorConfig()` keeps `delta-v1` as its default, and it serializes exactly as before #79 unless another contract is requested. A learned model's compatibility contract records `action_semantics_version`, so a model never runs under another contract, even one with the same action count.

### Transition model

A simulation tick applies the controller action, reads workload demand, processes queued/new requests using current active capacity, computes latency/SLA/cost/reward, then advances replica lifecycle and simulation time.

Controller-specific state such as threshold cooldown/stabilization belongs to that controller, not the core environment.

## Predictive baselines

`predictive-v1` (`controllers/predictive.py`, `linear-trend` + `forecast-plus-backlog-v1`) and `predictive-seasonal-v1` (`controllers/proactive_predictive.py`, `historical-profile-plus-linear-v1` + `proactive-scaleout-conservative-scalein-v1`, #80) are separate controllers. They share pure helpers: the startup horizon (`startup_ticks`), `linear_trend_forecast`, `size_replicas` and `backlog_recovery_rate`. Both encode their targets through the environment's `ActionContract`, derived from the simulator config.

`HistoricalDemandProfile` is an immutable per-tick median of aligned TRAIN traces; validation and test workloads are refused as history. Forecast records store only what was known at decision time, and actual demand is joined for scoring after the episode.

## MLOps flow

```text
workload/config
      |
      v
training/evaluation -----> MLflow run
      |                     | params/tags
      |                     | metrics
      |                     | model/config/results artifacts
      v                     v
controller/model ------> reproducible evaluation
                              |
                              v
                       MLflow UI / ScaleRL dashboard
```

See `docs/MLOPS.md` for the run contract, Docker roles, and CI/CD boundaries.

## Design principle

ScaleRL must never assume that RL is superior. Every RL policy is evaluated against fair simpler baselines under the same held-out traces, seeds, constraints, and metrics.
