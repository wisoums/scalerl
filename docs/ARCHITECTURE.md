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

The initial observation vector is expected to include normalized values for:

- CPU utilization
- request rate
- request-rate trend
- queue length
- p95 latency
- active replicas
- pending replicas
- estimated infrastructure cost
- time since last scaling action

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
