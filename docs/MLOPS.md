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

The core simulator, workloads, controllers, and benchmarks never import MLflow. `scalerl.mlops` imports it only when a run is started; without the extra, `start_tracked_run` raises an `ImportError` with this install hint.

### Tracking a run

```python
from scalerl.environment import SimulatorConfig
from scalerl.mlops import RunSpec, start_tracked_run

spec = RunSpec(
    run_kind="tune",  # train | tune | evaluate
    controller="threshold",
    workload_id="azure-val-734400",  # from the benchmark manifest (#18)
    workload_split="validation",  # must match the manifest
    simulator_config=SimulatorConfig(...),
    simulator_config_source="calibrated_train_validation",
    calibration_workload_ids=("azure-train-129600", "azure-val-734400"),
    calibration_note="why these values were chosen",
    seed=7,
    evaluation_seeds=(100, 101),
    hyperparameters={"low_threshold": 0.3, "high_threshold": 0.8},
)

with start_tracked_run(spec, experiment_name="scalerl") as run:
    ...  # run the controller
    run.log_metrics({"episode_reward": total_reward, "sla_violation_rate": rate})
    run.log_metric("train_loss", loss, step=step)
    run.log_artifact_dict("evaluation.json", summary)
    run.log_artifact("model.zip", artifact_path="model")
print(run.run_id)
```

`start_tracked_run` creates or selects the experiment, starts the run, and logs all the lineage below automatically. The run ends `FINISHED` on success, `FAILED` on an exception (which is re-raised, never swallowed), and `KILLED` on `KeyboardInterrupt`.

Metrics are only what the caller computed and passes in. Values must be finite real numbers, and anything not computed is simply not logged (no placeholder zeros). Use `step` for time series such as learning curves.

### What every run records

**Tags:** `scalerl.run_kind`, `scalerl.controller`, `scalerl.benchmark_version`, `scalerl.workload_id`, `scalerl.workload_split`, `scalerl.simulator_config_source`, `scalerl.git_sha`, `scalerl.git_dirty`, `scalerl.version`, and MLflow's `mlflow.source.git.commit` when the SHA is known.

**Params:** `run_kind`, `controller`, `benchmark_version`, `workload_id`, `workload_split`, `simulator_config_source`, and, when provided, `seed`, `evaluation_seeds`, `training_steps`, `training_episodes`, `calibration_workload_ids`, `calibration_note`. Nested values are flattened for filtering: `sim.*` (every `SimulatorConfig` field), `reward.*` (every `RewardWeights` field), `hp.*` (hyperparameters), and `compat.*` (the compatibility contract).

**Artifacts** (under `scalerl/`) are the exact source of truth; the flattened params are only for searching:

| File | Content |
|---|---|
| `resolved_run.json` | run ID, full `RunSpec`, software, compatibility |
| `simulator_config.json` | complete nested `SimulatorConfig` |
| `reward_weights.json` | complete `RewardWeights` |
| `software.json` | Git SHA and dirty flag; ScaleRL and Python versions; MLflow, Gymnasium, NumPy, pandas, Pydantic, Stable-Baselines3, and PyTorch versions (`not_installed` when absent) |
| `compatibility.json` | observation shape, action count, startup delay, control interval, max replicas, service capacity, benchmark version |

The Git SHA comes from an explicit `git_sha=` argument, then `SCALERL_GIT_SHA` or `GITHUB_SHA`, then `git rev-parse` next to the installed package, and otherwise `unknown`, so runs from an installed wheel without `.git/` still work.

### Held-out and calibration guardrails

`RunSpec` validates against the benchmark manifest from #18:

- `workload_split` must match the manifest, so a test workload cannot be relabeled;
- `train` and `tune` runs cannot target `test` workloads; `evaluate` runs may, once all choices are frozen;
- `simulator_config_source` records **why** the configuration has its values:
  - `default`: exactly `SimulatorConfig()`;
  - `predeclared`: fixed before looking at any workload statistics;
  - `calibrated_train_validation`: derived from the workloads listed in `calibration_workload_ids`, which must all be `train`/`validation` workloads; a `test` workload is rejected.

**Azure capacity.** The validated Azure v1 train/validation windows average 0.45–2.29 RPS and peak at 4.9 RPS, so with the default `service_capacity_rps = 50` a single replica serves them and autoscaling barely matters. Later experiments may choose a different simulator configuration to make Azure autoscaling meaningful, but only from train/validation workloads, recorded with `simulator_config_source="calibrated_train_validation"` and its calibration workload IDs. Azure demand is never normalized or rescaled, and test statistics never drive capacity, replica bounds, rewards, thresholds, predictive settings, or DQN/PPO hyperparameters.

### Learned-model compatibility

The v1 observation size depends on startup delay and control interval, so a trained policy only fits environments with the same observation/action contract. `EnvironmentCompatibility.from_env(env)` or `.from_config(config)` captures it from a real `AutoscalingEnv`, and `trained.require_compatible(current)` raises with every mismatching field. Every run logs this contract as `compatibility.json`; DQN/PPO model loading (#15/#16) must check it before inference.

### Tracking location

No URL is hard-coded. `tracking_uri=None` honors `MLFLOW_TRACKING_URI` and MLflow's defaults, and an explicit `tracking_uri=` overrides both. The same code works with:

- local SQLite:

  ```bash
  export MLFLOW_TRACKING_URI=sqlite:///mlflow.db
  mlflow ui --backend-store-uri sqlite:///mlflow.db
  ```

  Artifacts are written to `./mlruns` under the working directory (both are gitignored);
- a local server: `mlflow server --host 127.0.0.1 --port 5000` with `MLFLOW_TRACKING_URI=http://127.0.0.1:5000`;
- the later #44 containerized server, by pointing `MLFLOW_TRACKING_URI` at it.

Unit tests use a temporary SQLite store and never contact a server.

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
├── pytest + coverage (installed with dev,mlops, so MLflow tracking tests run
│   against a temporary SQLite store)
├── sdist/wheel build
└── clean wheel install/import smoke (no extras: core and scalerl.mlops
    import without MLflow; packaged benchmark manifest loads)
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

- **MLflow:** experiment lineage, params, metrics, and artifacts.
- **City View (#37):** visual understanding of what the simulator and controller are doing during a run.
