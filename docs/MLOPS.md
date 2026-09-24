# MLOps Architecture

ScaleRL uses MLOps tooling to make controller training and benchmark results reproducible, inspectable, and traceable to the code/configuration that produced them.

## Principles

1. **No orphan results.** Any number promoted to the README or final report must be traceable to an MLflow run ID, or to a documented aggregate of run IDs.
2. **Held-out data stays held out.** Training/tuning runs must never use the frozen test suite from Issue #18.
3. **Core simulation remains headless.** The simulator/environment must not require MLflow, Docker, or a tracking server to run unit tests.
4. **One controller interface.** Baselines and learned policies use the same environment/evaluation path.
5. **Model compatibility is explicit.** DQN/PPO artifacts record the observation/action-space contract and the relevant environment configuration.

## MLflow

MLflow is the canonical experiment tracker for training, tuning, and final evaluation.

Install the optional tooling with:

```bash
pip install -e ".[dev,mlops]"
```

### Local development mode

For lightweight local use:

```bash
mlflow server --host 127.0.0.1 --port 5000
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
```

MLflow's local server uses SQLite by default. This is sufficient for single-developer local experimentation.

Issue #44 adds the full Docker Compose stack with PostgreSQL and MinIO.

### Run contract

Training/tuning/evaluation runs should record:

#### Parameters / tags

- algorithm/controller;
- Git commit SHA;
- ScaleRL/Python dependency metadata;
- workload ID and split;
- training/evaluation seeds;
- simulator configuration;
- reward weights;
- observation shape;
- action-space size;
- startup delay;
- control interval;
- algorithm hyperparameters;
- training timesteps/episodes.

#### Metrics

At minimum where applicable:

- episodic/total reward;
- infrastructure cost;
- SLA violations/rate;
- p95 latency summaries;
- queue metrics;
- scaling actions/churn;
- training loss/learning statistics.

#### Artifacts

Where applicable:

- resolved experiment config;
- trained model/checkpoint;
- evaluation JSON/CSV;
- learning curves/plots;
- benchmark summaries.

### Learned-model compatibility

Because the v1 observation size depends on startup delay and control interval, learned model artifacts must record enough metadata to validate compatibility before inference.

At minimum record:

- observation shape;
- action-space size;
- startup delay;
- control interval;
- max replicas/service capacity as relevant;
- Git/config identifier.

A model should not be silently loaded into an incompatible environment.

## Docker

Two container concerns are intentionally separate:

- **Issue #44:** training + MLflow/PostgreSQL/MinIO MLOps stack.
- **Issue #24:** later production-inspired inference/demo image.

The training image must not bake the raw Azure dataset into an image. Data is mounted/provided at runtime.

## CI/CD

### Current CI

The checked-in GitHub Actions workflow currently verifies:

```text
PR / push
├── Ruff lint
├── Ruff format check
├── mypy
├── pytest + coverage
├── sdist/wheel build
└── clean wheel install/import smoke
```

### Planned in Issue #45

Issue #45 will expand CI into the full ML/MLOps pipeline:

```text
PR
├── current checks above
├── Docker build
├── container smoke
├── short SB3 training smoke
└── isolated MLflow logging smoke

release tag
└── publish versioned image to GHCR
```

These Docker, training-smoke, MLflow-smoke, and GHCR stages are **planned** and are not part of the current checked-in workflow yet.

Benchmark-scale RL training will never run in CI.

## Results UI responsibilities

- **MLflow UI:** experiment/run tracking, parameters, metrics, artifacts, model lineage.
- **ScaleRL dashboard:** domain-specific autoscaling interpretation: traffic, replicas, queue, latency/SLA, cost, and actions.

The custom dashboard must not become a second experiment database.
