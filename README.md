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
| Predictive autoscaler | Current startup-aware linear-trend + backlog baseline (#14/#63) |
| Cloud-style proactive predictive | `predictive-seasonal-v1` (#80): TRAIN-only historical profile + linear trend, direct scale-out, one-at-a-time scale-in, backlog recovery |
| Tabular Q-learning | Optional educational learned baseline |
| DQN | Primary discrete-action deep RL |
| PPO | Policy-gradient comparison |

## Workloads

ScaleRL uses one `WorkloadTrace` contract for both generated and recorded traffic.

Planned/active workload sources include:

- synthetic steady, seasonal, ramp, spike, and bursty traffic;
- selected slices of the **Microsoft Azure Functions Invocation Trace 2021** for real-production-trace evaluation.

The raw Azure dataset is not committed. Source/license/citation and the local data layout are documented in [`data/README.md`](data/README.md).

**Azure dataset attribution.** Real-trace results use the Microsoft Azure Functions Invocation Trace 2021 (CC-BY), from [Azure/AzurePublicDataset](https://github.com/Azure/AzurePublicDataset). Cite: Yanqi Zhang, Íñigo Goiri, Gohar Irfan Chaudhry, Rodrigo Fonseca, Sameh Elnikety, Christina Delimitrou, Ricardo Bianchini. "Faster and Cheaper Serverless Computing on Harvested Resources." *Proceedings of the ACM Symposium on Operating Systems Principles (SOSP)*, October 2021.

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

### Methodology gates before held-out evaluation

Validation-only #19 exposed an important failure mode: the original SLA-first DQN/PPO selector can prefer a trivial near-full-fleet policy because cost matters only after SLA is minimized. That result is preserved rather than overwritten.

Before opening the held-out test suite in #46, ScaleRL now freezes four additional methodology decisions:

- **#78 — constrained model selection (frozen):** `selection-v2-cost-under-sla` requires a candidate's SLA violation rate to be no worse than the tuned Threshold's on **every** validation workload, then minimizes mean normalized cost among feasible candidates (queue, churn, SLA only as tie-breakers; reward never used; within one algorithm family; nothing selected if nothing is feasible). The spec is committed at [`benchmarks/v1/selection-v2-cost-under-sla.json`](benchmarks/v1/selection-v2-cost-under-sla.json); the v1 selectors and #19 results are unchanged. Applied diagnostically to #19's seed artifacts, it still selects the near-full-fleet DQN/PPO policies (DQN 1/5 seeds feasible, PPO 5/5 identical); see [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md#78--cost-aware-selection-under-an-sla-constraint).
- **#79 — action semantics (frozen: `desired-replicas-v1`):** the historical `delta-v1` contract (`Discrete(3)` codes `0/1/2` = scale-down/hold/scale-up, effects `-1/0/+1`) is kept bit-for-bit. A discrete **desired replica count** contract was added: code `c` requests `min_replicas + c` replicas in one decision, which still wait out startup delay. A validation-only experiment used 20 matched DQN + 20 matched PPO configurations per contract and the frozen #78 selection. Under its predeclared principle, [`desired-replicas-v1`](benchmarks/v1/action-contract-v2.json) is the final contract for #20/#72/#46. Its consequences are mixed and reported as-is: PPO improves, no DQN configuration met the SLA constraint under the new contract, and Predictive oscillates on bursty traffic. The reward is unchanged; replica-change magnitude is reported separately. Horizontal replica counts are integers, so the issue is action granularity, not a need for continuous actions; see [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md#79--action-granularity-not-continuous-vs-discrete).
- **#80 — cloud-style proactive predictive baseline (frozen: [`predictive-seasonal-v1`](benchmarks/v1/predictive-baseline-v1.json)):** `predictive-v1` (`linear-trend` + `forecast-plus-backlog-v1`) is unchanged. The new baseline combines a TRAIN-only per-tick median historical profile with the linear trend (`max` of the two). It scales out directly to its startup-aware target and scales in by at most one replica, never below observed or forecast need and never while a backlog remains. It is inspired by cloud predictive scaling (recurring history plus capacity launched ahead of time), not an emulation of it. Validation findings:
  - On `syn-val-bursty` it cuts replica movement from 249 to 115 and SLA from 0.600 to 0.267.
  - Benchmark v1's synthetic validation has no recurring history.
  - On Azure validation the historical profile did not improve forecast accuracy, and every controller stays at one replica.

  See [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md#80--cloud-style-proactive-predictive-baseline).
- **#81 — startup-delay robustness (frozen: [`startup-robustness-v1`](benchmarks/v1/startup-robustness-freeze-v1.json)):** `robustness-v1` is unchanged. A separate extension adds a seeded per-replica startup-delay stress model, `tri-point-multiplicative-v1`: 0.5× / 1.0× / 1.5× nominal startup with probabilities 0.25 / 0.50 / 0.25. It is not a provider calibration. It has its own RNG stream, and the sampled realizations are hidden from controllers, which see nominal readiness only. There are two scenarios (`startup-delay-jitter`, `combined-startup-robustness`, seeds 0–4). In validation, controller trade-offs stayed stable.
- **#20 — reward ablation (frozen: [`full-cost-low-v1`](benchmarks/v1/reward-contract-v1.json)):** eight predeclared reward variants were compared under `desired-replicas-v1`, with the #79 DQN candidates and the fixed PPO `ppo-c08` trained per reward on seeds 0–4. Only the halved cost weight (latency 1, cost 0.5, SLA 1, queue 1, churn 0.1) produced a feasible DQN and a feasible PPO on every validation workload. The DQN SLA margin on bursty traffic is exactly zero, and PPO runs a near-full fleet under this reward. The package-default `RewardWeights()` is unchanged; see [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md#reward-ablation-20).

Kubernetes HPA itself calculates an integer **desired replica count**, which is why #79 tests direct target-capacity semantics instead of treating horizontal autoscaling as a continuous CPU/RAM action problem: <https://kubernetes.io/docs/concepts/workloads/autoscaling/horizontal-pod-autoscale/>.

All four decisions use train/validation evidence only and must be frozen before #46.

**#72 — sim-to-real protocol (frozen: [`sim-to-real-protocol-v1`](benchmarks/v1/sim-to-real-protocol-v1.json)).** The contract for the later local systems-in-the-loop / Knative sim-to-real validation is fixed before any live or held-out result. It covers:
- the controllers to deploy: Threshold, both predictive baselines, the canonical DQN and PPO artifacts chosen by a validation-only seed rule, and Knative-native;
- three fixed-offset 10-minute Azure replay windows, each with frozen arrival schedules for load seeds 0–2;
- 30 s / 1–10 replica `desired-replicas-v1` control;
- a shared observation builder used by simulation and live control;
- descriptive transfer reporting with no pass/fail and no "RL must win".

Local Knative is not claimed to reproduce any cloud provider. See [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md#sim-to-real-protocol-72).

## Held-out Azure results (#46)

The final held-out simulator evaluation ran the #72-frozen controllers on the two frozen Azure TEST hours, `azure-test-993600` and `azure-test-1166400`. The spec ([`heldout-evaluation-v1`](benchmarks/v1/heldout-evaluation-v1.json), `b32b2f3d3dd6`) was committed at `ebccda6` **before** any held-out episode ran.

Result: [`heldout-results-v1`](benchmarks/v1/heldout-results-v1.json), `d9d3fb985f2d`. It covers 264 `robustness-v1` + 220 startup-robustness cases and 15 #72 replay references, with one MLflow run each in `scalerl-heldout-v1`. No run was superseded.

**Primary held-out Azure nominal result.** The table shows normalized cost; no controller violated the SLA.

| Controller | azure-test-993600 | azure-test-1166400 |
|---|---|---|
| static-v1 (5 replicas) | 0.500 | 0.500 |
| random-v1 (sanity, 5 seeds) | 0.572 | 0.572 |
| threshold-v1 | **0.100** | **0.100** |
| predictive-v1 | **0.100** | **0.100** |
| predictive-seasonal-v1 | **0.100** | **0.100** |
| DQN `dqn-c14` seed 0 | 1.000 | 1.000 |
| PPO `ppo-c08` seed 4 | 0.935 | 0.934 |

- **A conventional baseline wins this held-out comparison.** Both TEST hours carry only 1.3–1.5 requests/s against 50 requests/s per replica. Threshold and both Predictive baselines hold the minimum fleet for the whole hour.
- **Both learned policies overprovision.** DQN requests all 10 replicas on its first decision and holds them; PPO stays at 9–10. That is 9–10× the baselines' cost with no SLA or queueing benefit. The learned policies reduce the simulator's mean p95 by only about 0.5 ms (0.0201 s vs 0.0206 s), operationally negligible relative to the 0.5 s SLA target.
- **Robustness evidence is weak.** Capacity jitter, delayed telemetry and stochastic startup did not materially change scaling behavior, SLA, queueing or the cost conclusion on these low-load traces; seeded capacity jitter produces only small latency variation (at most 0.3 ms in mean p95). The load never stresses the system, so this is **not** evidence of robustness under stress.
- **Scope of the result.** It is decisive about overprovisioning at low load and silent about SLA trade-offs under load. The Azure TEST hours are weak autoscaling stress cases. This was not changed after the results; a calibrated-amplitude study would need its own predeclared protocol.
- **Replay references for #76.** Each frozen #72 window has one nominal simulator reference per controller, mapped to its three live load schedules (not three replicates).

Details, run IDs and limitations: [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md#held-out-azure-evaluation-46).

## MLOps

ScaleRL uses:

- **MLflow 3.x** for experiment/run tracking, metrics, configuration/model artifacts, and run IDs;
- **Docker Compose** for a reproducible local stack: Scenario Lab, trainer, MLflow server (PostgreSQL + S3-compatible Garage artifacts), and Optuna Dashboard; a later inference/demo image is separate (#24);
- **GitHub Actions** as the reproducibility gate. Every PR runs lint/format/strict types, tests on Python 3.11 and 3.12, a package build with a clean wheel install, and the full Docker Compose stack smoke: a tracked ScaleRL run with MLflow → PostgreSQL metadata and Garage artifacts, Optuna → PostgreSQL seen by the Dashboard, and a tiny CPU Stable-Baselines3 learning smoke. Version tags publish the runtime image to GHCR (amd64 + arm64); benchmark-scale training never runs in CI;
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

The critical path is fair baselines, frozen held-out data, MLflow/Docker/CI reproducibility, DQN/PPO, multi-seed validation evidence, the pre-held-out methodology gates (#78–#81 and #20), then held-out synthetic/Azure evaluation and an honest results package.

See [`ROADMAP.md`](ROADMAP.md).

## Development

Core development:

```bash
git clone https://github.com/wisoums/scalerl.git
cd scalerl
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,mlops,dashboard,tuning]"
pytest
```

The `mlops`, `dashboard`, and `tuning` extras install MLflow, Streamlit, and Optuna; their tests need them, but the core simulator, controllers, and workloads work without any of them. To browse tracked runs locally:

```bash
export MLFLOW_TRACKING_URI=sqlite:///mlflow.db
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

### Full local stack (Docker Compose)

```bash
scripts/setup-local-stack.sh   # once: .env with your UID/GID, outputs/, data/raw/
docker compose up --build
```

This starts the Scenario Lab at <http://localhost:8501>, MLflow at <http://localhost:5000>, and the Optuna Dashboard at <http://localhost:8080>. Behind them run PostgreSQL (separate `mlflow` and `optuna` databases) and an S3-compatible artifact store (Garage). Run evaluations and tuning with `docker compose run --rm trainer <command>`. It is a local reproducibility stack, not a production deployment; see [docs/DOCKER.md](docs/DOCKER.md) for commands, persistence, secrets, and the smoke test.

### DQN training

Train the first deep-RL controller (Stable-Baselines3 DQN) on one train workload and validate it on validation workloads, tracked in MLflow:

```bash
python -m scalerl.training.dqn --workload syn-train-bursty \
    --validation-workload syn-val-bursty --timesteps 200000 --seed 0 \
    --tracking-uri sqlite:///outputs/mlflow.db --output outputs/dqn-v1.json
```

The same command runs in the full stack with `docker compose run --rm trainer python -m scalerl.training.dqn …`. Tune with `python -m scalerl.tuning.dqn`. Held-out test workloads are refused. See [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md#dqn-15) and [docs/MLOPS.md](docs/MLOPS.md).

### PPO training

The second deep-RL controller (Stable-Baselines3 PPO: actor/critic) uses the same observation, reward, splits, MLflow schema, model bundle, and evaluation as DQN. Its default budget is 204,800 timesteps, i.e. 100 rollouts of 2,048:

```bash
python -m scalerl.training.ppo --workload syn-train-bursty \
    --validation-workload syn-val-bursty --timesteps 204800 --seed 0 \
    --tracking-uri sqlite:///outputs/mlflow.db --output outputs/ppo-v1.json
```

Docker: `docker compose run --rm trainer python -m scalerl.training.ppo …`. Tune with `python -m scalerl.tuning.ppo`. See [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md#ppo-16).

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

The deterministic simulator/Gymnasium environment; random, static, tuned threshold, and queue-aware predictive baselines; the frozen synthetic + Azure benchmark; MLflow tracking, Optuna studies, and the Docker Compose stack; and the Scenario Lab with Live City are implemented. The GitHub Actions reproducibility gate (#45) and the DQN and PPO training pipelines (#15/#16: SB3 DQN and PPO, compatibility-checked model bundles, MLflow lineage, Optuna tuning on train/validation) are in place. Robustness scenarios (#65, `robustness-v1`: nominal, ±10% seeded capacity jitter, one-tick delayed telemetry, and both combined) are defined for evaluating fixed controllers. Multi-seed evaluation (#19: canonical train/validation tuning, five-seed DQN/PPO model families, matched-dynamics robustness evaluation, and descriptive statistics) is in place; see [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md#multi-seed-evaluation-19). Its validation-only run exposed that the v1 SLA-first selector can choose near-full-fleet DQN/PPO policies, so the project now resolves #78 (cost-under-SLA selection), #79 (desired-replica action semantics), #80 (stronger proactive predictive baseline), #81 (startup-delay stochasticity), and #20 (reward ablation), and freezes the #72 sim-to-real protocol, **before** opening #46. The held-out Azure evaluation (#46) is complete. On the two low-traffic Azure TEST hours, the Threshold and Predictive baselines matched every controller's SLA at 9–10× lower cost than the canonical DQN and PPO, which overprovision. Because the Azure hours never stress the system, this is not a verdict on RL under load; see [Held-out Azure results](#held-out-azure-results-46).

## License

ScaleRL source code is MIT licensed. External datasets retain their own licenses; see `data/README.md`.
