# ScaleRL

**A personal student project for learning reinforcement learning, ML engineering, and cloud autoscaling.**

ScaleRL is my ongoing learning playground for asking a deliberately narrow question:

> **Inside a simplified autoscaling simulator, can DQN or PPO learn useful scaling behavior across different workload patterns without wasting capacity compared with simpler reactive and predictive controllers?**

This repository is **not a claim that I solved cloud autoscaling**, and it is not meant to be a complete model of a real production platform. Real autoscaling can depend on many more factors than this project currently models: CPU and memory pressure, concurrency, request classes, execution-time distributions, network/storage bottlenecks, resource requests and limits, multi-service dependencies, heterogeneous instances, provider-specific behavior, failures, cold starts, costs, SLOs, control-plane delays, and much more.

I use research-style habits such as frozen train/validation/test splits, reproducible experiment specs, multiple seeds, baselines, MLflow lineage, and honest negative results because they are useful ways to **learn how to evaluate ML systems carefully**. They should not be read as a claim of academic or production completeness.

> **Development note:** this is also an AI-assisted learning project. I use coding assistants heavily for implementation, review, debugging, and iteration. The goal is to understand the system, make and challenge the design decisions, inspect failures, and learn from the experiments rather than to pretend every line was written manually from first principles.

The code is public and MIT-licensed, but ScaleRL is maintained primarily as a **personal learning project**, not as a production autoscaling product or a community-governed research effort. I expect its scope, datasets, simulator assumptions, and experiments to keep evolving over time.

## What I am trying to learn

ScaleRL combines several topics I wanted to understand together:

- reinforcement learning for sequential control;
- DQN and PPO with Stable-Baselines3;
- strong non-RL baselines rather than comparing RL only with weak references;
- train/validation/test isolation;
- hyperparameter tuning with Optuna;
- experiment/model lineage with MLflow;
- reproducible Docker/CI workflows;
- failure analysis and out-of-distribution behavior;
- eventually, simulator-to-real validation with a local Knative testbed.

RL is treated as a **hypothesis**, not as the expected winner.

A simple Threshold or Predictive controller is a perfectly valid result if it provides an equal or better service/cost trade-off.

See [docs/WHY_RL.md](docs/WHY_RL.md).

## Current scope: deliberately simplified

The current v1 learned-policy world is intentionally small enough to understand and reproduce.

Typical frozen values include:

| Assumption | Current v1 value |
| --- | --- |
| Control interval | 30 s |
| Replica range | 1–10 |
| Initial replicas | 1 |
| Service capacity | 50 requests/s per replica |
| Startup delay | 60 s |
| SLA target | 0.5 s |
| Learned action contract | desired-replicas-v1 |
| Learned observation | scalerl-observation-v1 |

That means a result in ScaleRL means:

> "This controller behaved this way under this specific simulator and experiment contract."

It does **not** automatically mean:

> "This controller would behave the same way for arbitrary Kubernetes/serverless/cloud workloads."

That distinction is one of the main reasons the project is still ongoing.

## Controllers

| Controller | Purpose |
| --- | --- |
| Random | Sanity check |
| Static | Fixed-capacity reference |
| Threshold | Strong reactive baseline |
| Predictive | Short-history/startup-aware baseline |
| predictive-seasonal-v1 | Stronger proactive predictive baseline |
| DQN | Discrete deep-RL policy |
| PPO | Policy-gradient comparison |
| Tabular Q-learning | Optional educational follow-up |

## Current workload support

Implemented today:

- reproducible synthetic workload generators;
- selected windows from the **Microsoft Azure Functions Invocation Trace 2021**.

The raw Azure dataset is not committed. Setup, attribution, and data semantics are in [data/README.md](data/README.md).

The next benchmark phase plans to investigate broader workload diversity, including additional real datasets and more deliberately varied synthetic regimes. Those sources are **planned until their loaders and frozen benchmark manifests actually land**.

## What happened in the first held-out experiment?

The first frozen held-out Azure evaluation is preserved exactly as it happened.

Spec:

- heldout-evaluation-v1
- ID b32b2f3d3dd6

Result:

- heldout-results-v1
- ID d9d3fb985f2d

It contains 264 robustness-v1 cases, 220 startup-robustness cases, and 15 short-window replay references, all tracked in MLflow.

### Nominal normalized cost

No controller violated the SLA on either Azure TEST hour.

| Controller | azure-test-993600 | azure-test-1166400 |
| --- | ---: | ---: |
| Static, 5 replicas | 0.500 | 0.500 |
| Random, 5-seed mean | 0.572 | 0.572 |
| Threshold | **0.100** | **0.100** |
| Predictive | **0.100** | **0.100** |
| Seasonal Predictive | **0.100** | **0.100** |
| DQN dqn-c14, seed 0 | 1.000 | 1.000 |
| PPO ppo-c08, seed 4 | 0.935 | 0.934 |

The held-out Azure hours averaged only about **1.3–1.5 requests/s**, while one simulated replica could serve **50 requests/s**.

So one replica was enough.

- Threshold and both predictive baselines stayed at one replica.
- DQN requested all 10 replicas.
- PPO stayed around 9–10 replicas.
- There were no SLA violations, queues, or dropped requests.
- The learned policies therefore used roughly **9–10× the normalized capacity cost** without a meaningful service benefit on these particular low-load traces.

That is a useful failure case, but it is a **narrow result**.

It does not establish that RL is generally bad at autoscaling. It shows that the v1 policies, trained under a much narrower workload distribution, generalized poorly to this near-idle regime.

Full provenance and limitations: [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md).

## What comes next: benchmark-v2

Instead of changing the old TEST set until RL looks better, the next phase starts a **new experiment with new untouched test data**.

The main question becomes:

> **If DQN/PPO are trained across a much broader, predeclared workload distribution, do they learn a more general cost/SLA scaling policy?**

The critical path is:

~~~text
#115  freeze benchmark-v2 methodology
  ↓
#116  heterogeneous dataset/provenance pipeline
  ↓
#117  workload taxonomy + balanced episode sampler
  ↓
#118  train/select generalist DQN/PPO
  ↓
#119  fresh untouched benchmark-v2 TEST evaluation
  ↓
#26   final synthesis of what improved and what did not
~~~

### Stage A — workload generalization

The first follow-up intentionally keeps the simulator world fixed and changes mainly the **training workload distribution**.

That lets the project test one clean idea:

> Was the v1 failure mainly caused by narrow training coverage?

Planned workload diversity includes different:

- traffic intensities relative to available capacity;
- steady/ramp/spike/burst/seasonal/noisy patterns;
- quiet-to-flash-crowd transitions;
- source applications and trace domains.

### Stage B — environment generalization

A separate later issue, #120, studies whether the policy contract itself must change before one model could sensibly transfer across different:

- fleet sizes;
- per-replica capacities;
- startup delays;
- control intervals;
- SLA targets;
- initial capacity;
- service characteristics.

This stage may require a future observation-v2 and/or action-v2.

Those contracts are **not implemented or selected yet**.

## Why not just add every possible production parameter now?

Because then it becomes very difficult to understand why a model improved or failed.

ScaleRL currently prefers small, versioned experiments:

~~~text
change one important assumption
          ↓
freeze the protocol
          ↓
train / validate
          ↓
open fresh test data
          ↓
keep the result even if it is bad
~~~

There are many production factors the simulator still ignores. Some may eventually be worth adding; others may be outside the scope of a student project.

The goal is not to recreate an entire cloud provider.

The goal is to keep learning from increasingly realistic, testable versions of the problem.

## Tooling

ScaleRL currently uses:

- **Stable-Baselines3 + PyTorch** for DQN/PPO;
- **Optuna** for hyperparameter studies;
- **MLflow** for run/model/artifact lineage;
- **Gymnasium** for the RL environment contract;
- **Docker Compose** for the local experiment stack;
- **GitHub Actions** for lint/type/test/package/container smoke checks;
- **Streamlit** for Scenario Lab.

MLflow is optional for the simulator core.

## Architecture

~~~text
Synthetic / recorded workloads
             │
             ▼
     autoscaling simulator
             │
             ▼
      Gymnasium environment
             │
     ┌───────┴────────┐
     ▼                ▼
 baselines         DQN / PPO
     │                │
     └───────┬────────┘
             ▼
   system metrics + reward
             │
      ┌──────┴──────┐
      ▼             ▼
   MLflow      Scenario Lab
~~~

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Run locally

Core development:

~~~bash
git clone https://github.com/wisoums/scalerl.git
cd scalerl
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,mlops,dashboard,tuning]"
pytest
~~~

Local stack:

~~~bash
scripts/setup-local-stack.sh
docker compose up --build
~~~

Typical services:

- Scenario Lab: http://localhost:8501
- MLflow: http://localhost:5000
- Optuna Dashboard: http://localhost:8080

See [docs/MLOPS.md](docs/MLOPS.md) and [docs/DOCKER.md](docs/DOCKER.md).

## Project status

The project already has:

- deterministic simulator/environment plumbing;
- synthetic + Azure 2021 workloads;
- Threshold and predictive baselines;
- DQN/PPO training;
- Optuna tuning;
- MLflow lineage;
- multi-seed evaluation;
- versioned action/reward/robustness experiments;
- frozen held-out v1 evidence;
- Docker/CI;
- Scenario Lab.

The current focus is **not to declare a final research result**. It is to broaden the workload/data coverage and see whether a better-trained RL policy actually becomes more general.

Later work may include:

- non-stationary/failure scenarios (#21);
- generalist workload training/evaluation (#115–#119);
- environment-generalization study (#120);
- local Knative sim-to-real-v2 (#121, #73–#76);
- compute-efficiency measurements (#111);
- learning/dashboard/portfolio polish.

See [ROADMAP.md](ROADMAP.md).

## Limitations

ScaleRL is intentionally incomplete.

Current limitations include:

- simplified queue/service/latency/cost models;
- fixed v1 fleet/capacity/startup/SLA assumptions;
- limited workload domains so far;
- no claim that the simulator reproduces Azure/AWS/GCP internals;
- no CPU/memory/I/O/network/multi-service model comparable to a production cluster;
- learned policies may exploit simulator-specific structure;
- real-system validation is still future work;
- benchmark-v2 and environment-generalization work are not complete.

If this project keeps growing, some of those assumptions will be versioned and revisited. Others may remain intentionally simplified.

## License

ScaleRL source code is MIT licensed.

External datasets retain their own licenses and attribution requirements; see [data/README.md](data/README.md).
