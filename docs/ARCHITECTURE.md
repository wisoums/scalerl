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

The v1 observation (`AutoscalingEnv`) is a `Box` in `[0, 1]` describing the state after the last completed tick:

| Index | Feature |
|---|---|
| 0 | demand pressure: `rate / (rate + max_replicas * service_capacity_rps)` |
| 1 | utilization of active capacity |
| 2 | queue pressure: `queued / (queued + max-fleet capacity per tick)` |
| 3 | latency pressure: `p95 / (p95 + latency_target)` |
| 4 | active replicas / `max_replicas` |
| 5 | tick cost / cost of `max_replicas` for one tick |
| 6 | episode progress |
| 7 … 7+k−1 | pending replicas that activate after 1 … k more ticks, each / `max_replicas` |

The observation size is config-dependent:

```text
observation_size = 7 + ceil(startup_delay_seconds / control_interval_seconds)   # delay > 0
observation_size = 7                                                            # delay == 0
```

Bucketing pending replicas by readiness keeps the observation Markov when startup delay spans several ticks. For a fixed `SimulatorConfig` the shape never changes across workloads, seeds, resets, actions, or episodes; changing startup delay or control interval may change it.

**Model compatibility:** a trained DQN/PPO policy is only directly usable with an environment whose observation/action spaces match its training environment. MLflow model/run metadata records this contract; training/serving code must reject obvious incompatibilities rather than silently loading them.

Features the simulator does not model (such as CPU utilization) are intentionally not fabricated.

### Action space

- `0`: scale down by one replica
- `1`: hold
- `2`: scale up by one replica

Actions are bounded by configurable minimum and maximum replica counts.

### Transition model

A simulation tick applies the controller action, reads workload demand, processes queued/new requests using current active capacity, computes latency/SLA/cost/reward, then advances replica lifecycle and simulation time.

Controller-specific state such as threshold cooldown/stabilization belongs to that controller, not the core environment.

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
