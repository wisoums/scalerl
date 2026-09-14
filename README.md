# ScaleRL

**Deep reinforcement learning for adaptive, cost-aware cloud autoscaling.**

ScaleRL is an experimental ML systems project that studies whether reinforcement-learning policies can make better sequential autoscaling decisions than conventional reactive and predictive controllers when jointly optimizing **latency, SLA compliance, infrastructure cost, and scaling stability**.

> The goal is not to assume RL is better. ScaleRL is designed to measure **when it helps, when it does not, and why**.

## Research question

**Can a reinforcement-learning controller outperform traditional reactive and predictive autoscaling policies under unpredictable workloads while maintaining application service-level objectives?**

## Why this problem matters

Common autoscalers react to metrics such as CPU utilization or use forecasts to provision capacity ahead of predictable demand. These approaches are effective, but scaling is also a sequential decision problem: a decision made now affects future latency, queueing, cost, and capacity because replicas take time to start and workloads can change before the next control step.

ScaleRL models those delayed consequences and evaluates learned policies against strong non-RL baselines.

## Planned controllers

| Controller | Role |
| --- | --- |
| Random policy | Sanity check |
| Static capacity | Reference baseline |
| Threshold / target tracking | Reactive industry-style baseline |
| Predictive autoscaler | Forecasting baseline |
| DQN | Discrete-action deep RL |
| PPO | Policy-gradient comparison |

## Initial RL formulation

**Observation:** CPU utilization, request rate/trend, queue depth, p95 latency, active/pending replicas, cost, and time since the previous scaling action.

**Actions:** scale down, hold, or scale up.

**Objective:** minimize latency, SLA violations, cloud cost, queueing, and unnecessary scaling churn while obeying hard replica limits.

## Evaluation

Policies will be tested on reproducible workload families including steady traffic, daily seasonality, gradual ramps, abrupt spikes, and stochastic bursts. Final results will use held-out workloads and multiple random seeds.

Primary metrics:

- p95 latency
- SLA violation rate
- infrastructure cost / replica-hours
- completed or dropped requests
- scaling-action count
- cumulative reward

## Architecture

```text
Workload traces
      |
      v
Cloud simulator ---> Metrics / state
      |                   |
      |                   v
      |              Controller
      |        (baseline / DQN / PPO)
      |                   |
      +<------ action ----+
      |
      v
Evaluation harness ---> latency | SLA | cost | churn
```

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md), [`docs/REWARD_DESIGN.md`](docs/REWARD_DESIGN.md), and [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) for the current design.

## Roadmap

Development proceeds from simulation and deterministic baselines to deep RL, rigorous benchmarking, and finally a production-inspired shadow-mode demo. See [`ROADMAP.md`](ROADMAP.md).

## Development

The project uses Python 3.11+, Gymnasium, PyTorch, Stable-Baselines3, pytest, Ruff, and mypy.

```bash
git clone https://github.com/wisoums/scalerl.git
cd scalerl
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

## Project status

**Early development.** The current focus is building a deterministic, testable cloud workload simulator before introducing RL agents.

## License

MIT
