# ScaleRL

**Deep reinforcement learning for adaptive, cost-aware cloud autoscaling.**

ScaleRL is an experimental ML systems project that studies whether reinforcement-learning policies can make better sequential autoscaling decisions than conventional reactive and predictive controllers when jointly optimizing **latency, SLA compliance, infrastructure cost, and scaling stability**.

> The goal is not to assume RL is better. ScaleRL is designed to measure **when it helps, when it does not, and why**.

## Research question

**Can a reinforcement-learning controller outperform traditional reactive and predictive autoscaling policies under unpredictable workloads while maintaining application service-level objectives?**

## Why this problem matters

Common autoscalers react to metrics such as CPU utilization or use forecasts to provision capacity ahead of predictable demand. These approaches are effective, but scaling is also a sequential decision problem: a decision made now affects future latency, queueing, cost, and capacity because replicas take time to start and workloads can change before the next control step.

ScaleRL models those delayed consequences and evaluates learned policies against strong non-RL baselines.

## Why reinforcement learning?

RL is used here as a **hypothesis to evaluate**, not because autoscaling must be solved with RL.

Autoscaling has several properties that make RL a reasonable direction to test: decisions are repeated over time, actions affect future system states, replica startup introduces delayed consequences, service quality and infrastructure cost conflict, and there is no dataset containing a known optimal scaling action for every possible system state.

Other ML paradigms still have useful roles:

- **Supervised learning / forecasting** can predict future demand, but prediction alone does not decide how to trade future latency, cost, SLA risk, and scaling churn. A predictive controller is therefore included as a baseline.
- **Unsupervised learning** can discover workload regimes or anomalies, but does not directly learn a sequential control policy for the project objective.
- **Contextual bandits** optimize actions without fully modeling how those actions change later states; autoscaling actions can change capacity and queueing several control intervals into the future.
- **Classical control and model predictive control** are strong alternatives when system dynamics are known and modelable. They are not treated as obsolete; an MPC baseline is a possible extension.

The project will reject the RL hypothesis if well-tuned simpler controllers provide an equal or better cost/service trade-off on held-out workloads.

See [`docs/WHY_RL.md`](docs/WHY_RL.md) for the full paradigm-selection argument and falsifiable hypothesis.

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

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md), [`docs/REWARD_DESIGN.md`](docs/REWARD_DESIGN.md), [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md), and [`docs/WHY_RL.md`](docs/WHY_RL.md) for the current design.

## Roadmap

The target for the portfolio-ready v1.0 release is **October 31, 2026**. The critical path is the simulator, strong baselines, DQN/PPO, held-out multi-seed benchmarking, and a concise demo/results package. Production-inspired serving and advanced failure scenarios are secondary to completing rigorous ML evaluation.

See [`ROADMAP.md`](ROADMAP.md) for the dated execution plan and scope boundaries.

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
