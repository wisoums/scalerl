# Architecture

ScaleRL is an experimental autoscaling platform for comparing traditional controllers with deep reinforcement-learning policies under identical simulated workloads.

## System layers

1. **Workload generator** — emits reproducible request-rate traces: steady, diurnal, bursty, ramp, and shock workloads.
2. **Cloud simulator** — models replicas, startup delay, queueing, service capacity, latency, failures, and infrastructure cost.
3. **Gymnasium environment** — exposes observations, actions, transitions, termination rules, and reward signals.
4. **Controllers** — random, static, threshold/target-tracking, predictive, DQN, and PPO.
5. **Evaluation harness** — runs controllers against held-out workloads with fixed seeds and records latency, SLA violations, cost, scaling churn, and reward.
6. **Serving/demo layer** — later phases expose trained policies through FastAPI and a dashboard in shadow/recommendation mode.

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

The observation size is therefore **config-dependent**:

```
observation_size = 7 + ceil(startup_delay_seconds / control_interval_seconds)   # delay > 0
observation_size = 7                                                            # delay == 0
```

Bucketing pending replicas by readiness keeps the observation Markov when startup delay spans several ticks. For a fixed `SimulatorConfig` the shape never changes across workloads, seeds, resets, actions, or episodes; changing startup delay or control interval may change it.

**Model compatibility:** a trained DQN/PPO policy is only directly usable with an environment whose observation and action spaces match its training environment. Enforcing model/config compatibility belongs to the later training and serving work.

Features the simulator does not model yet (CPU utilization, request-rate trend or forecasts, time since last scaling action) are intentionally not observed.

### Action space

The MVP uses a discrete action space:

- `0`: scale down by one replica
- `1`: hold
- `2`: scale up by one replica

Actions are bounded by configurable minimum and maximum replica counts.

### Transition model

A simulation tick advances workload demand, replica startup/shutdown state, request service, queue length, latency, cost, and controller cooldown state.

## Design principle

ScaleRL must never assume that RL is superior. Every RL policy is evaluated against simple and predictive baselines under the same traces, seeds, constraints, and metrics.
