# ScaleRL

**Deep reinforcement learning for adaptive, cost-aware cloud autoscaling.**

ScaleRL is an experimental ML systems project that studies whether reinforcement-learning policies can make better sequential autoscaling decisions than conventional reactive and predictive controllers when jointly optimizing **latency, SLA compliance, infrastructure cost, and scaling stability**.

> The goal is not to assume RL is better. ScaleRL is designed to measure **when it helps, when it does not, and why**.

## Research question

**Can a reinforcement-learning controller outperform traditional reactive and predictive autoscaling policies under unpredictable workloads while maintaining application service-level objectives?**

## Why this problem matters

Common autoscalers react to utilization or use forecasts to provision capacity ahead of predictable demand. Scaling is also a sequential decision problem: a decision made now affects future latency, queueing, cost, and capacity because replicas take time to start and workloads can change before the next control step.

ScaleRL models those delayed consequences and evaluates learned policies against strong non-RL baselines.

## Why reinforcement learning?

RL is used here as a **hypothesis to evaluate**, not because autoscaling must be solved with RL.

Autoscaling has several properties that make RL reasonable to test: decisions repeat over time, actions affect future states, replica startup introduces delayed consequences, service quality and infrastructure cost conflict, and there is no dataset containing a known optimal scaling action for every state.

Other paradigms remain first-class comparisons:

- **Forecasting/supervised learning** predicts demand but does not by itself optimize the sequential cost/SLA trade-off.
- **Classical reactive autoscaling** is a strong industry-style baseline and must be tuned fairly.
- **Classical control/MPC** is a valid extension when dynamics are known.

The project rejects the RL hypothesis if well-tuned simpler controllers provide an equal or better cost/service trade-off on held-out workloads.

See [`docs/WHY_RL.md`](docs/WHY_RL.md).

## Controllers

| Controller | Role |
| --- | --- |
| Random policy | Sanity check |
| Static capacity | Fixed-cost reference |
| Threshold / target tracking | Tuned reactive baseline |
| Predictive autoscaler | Forecasting baseline |
| Tabular Q-learning | Optional educational learned baseline |
| DQN | Primary discrete-action deep RL |
| PPO | Policy-gradient comparison |

## Workloads

ScaleRL uses one `WorkloadTrace` contract for both generated and recorded traffic.

Planned/active workload sources include:

- synthetic steady, seasonal, ramp, spike, and bursty traffic;
- selected slices of the **Microsoft Azure Functions Invocation Trace 2021** for real-production-trace evaluation.

The raw Azure dataset is not committed. Source/license/citation and the local data layout are documented in [`data/README.md`](data/README.md).

Training, validation/tuning, and final held-out test workloads are explicitly separated before deep-RL tuning.

## Evaluation

Primary metrics:

- p95 latency proxy / latency summaries
- SLA violation rate
- infrastructure cost / replica-hours
- queued/completed/dropped requests
- scaling-action count/churn

Reward is reported as a secondary metric rather than the only success criterion.

Final comparisons use identical held-out traces/configs/seeds and multiple seeds.

## MLOps

ScaleRL uses:

- **MLflow 3.x** for experiment/run tracking, metrics, configuration/model artifacts, and run IDs;
- **Docker** for reproducible training and later inference/demo environments;
- **GitHub Actions** currently for lint/format/type/test/package checks; container and training/MLflow smoke stages are planned in Issue #45;
- **Stable-Baselines3 + PyTorch** for DQN/PPO training.

MLflow is intentionally optional for the simulator core.

See [`docs/MLOPS.md`](docs/MLOPS.md) and [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md).

## Architecture

```text
Synthetic / Azure workload traces
              |
              v
       Cloud simulator
              |
              v
       Gymnasium environment
              |
              v
   baseline / DQN / PPO controller
              |
              v
       Evaluation metrics
              |
       +------+------+
       |             |
       v             v
    MLflow      ScaleRL dashboard
 runs/artifacts  domain plots
```

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md), [`docs/REWARD_DESIGN.md`](docs/REWARD_DESIGN.md), [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md), [`docs/MLOPS.md`](docs/MLOPS.md), and [`docs/WHY_RL.md`](docs/WHY_RL.md).

## Roadmap

Portfolio-ready v1.0 target: **October 31, 2026**.

The critical path is fair baselines, frozen held-out data, MLflow/Docker/CI reproducibility, DQN/PPO, multi-seed synthetic + real-trace evaluation, and an honest results package.

See [`ROADMAP.md`](ROADMAP.md).

## Development

Core development:

```bash
git clone https://github.com/wisoums/scalerl.git
cd scalerl
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,mlops,dashboard]"
pytest
```

The `mlops` extra installs MLflow and the `dashboard` extra installs Streamlit; their tests need them, but the core simulator, controllers, and workloads work without either. To browse tracked runs locally:

```bash
export MLFLOW_TRACKING_URI=sqlite:///mlflow.db
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

The full Docker Compose MLOps stack is tracked in Issue #44.

### Scenario Lab City View

Watch the simulator interactively: traffic → scaling → queue → latency/cost, driven manually or by a baseline controller.

```bash
pip install -e ".[dashboard]"
python -m scalerl.dashboard
```

It opens at <http://localhost:8501>; stop it with `Ctrl+C`.

A quick tour:

1. **Manual:** press **↑ Scale up** a few times and watch 🏗️ pending shops become ☕ open shops after the startup delay, while the queue and latency respond.
2. **Threshold controller:** in the sidebar choose **City manager → 🌡️ Threshold**, press **Build / Reset Scenario**, then **▶ Step** or **⏭ Run to end**. The manager panel shows each decision and its reason.
3. **More dramatic traffic:** pick the workload `syn-train-spike` and build.
4. **Real Azure data:** with the trace extracted locally (see [data/README.md](data/README.md)), pick `azure-train-129600`; the default path points at `data/raw/`. One replica handles its low demand at the default capacity, so lower **Service capacity per replica** to about `1` and rebuild to see scaling. That change is for exploration only.

See [docs/CITY_VIEW.md](docs/CITY_VIEW.md) for the full guide.

## Project status

The deterministic simulator/Gymnasium environment and random/static baselines are implemented. Current work is focused on fair reactive/predictive baselines plus real-data and MLOps infrastructure before deep-RL benchmark training.

## License

ScaleRL source code is MIT licensed. External datasets retain their own licenses; see `data/README.md`.
