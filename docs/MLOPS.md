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

`start_tracked_run` creates or selects the experiment (parallel workers creating the same new experiment all join it), starts the run, and logs all the lineage below automatically. The run ends `FINISHED` on success, `FAILED` on an exception (which is re-raised, never swallowed), and `KILLED` on `KeyboardInterrupt`.

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
| `compatibility.json` | observation shape, action count, and every config field that defines observation features: traffic history ticks, startup delay, control interval, episode duration, max replicas, service capacity, cost per hour, SLA latency target; plus benchmark version |

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

The v1 observation size depends on the traffic history length (`traffic_history_ticks`), startup delay, and control interval, and feature values are normalized by episode duration, max replicas, service capacity, hourly price, and the SLA latency target. A trained policy therefore only fits environments where every one of these matches, not just the observation shape: a model trained with 4 history ticks is rejected by an environment with any other history length, even if the shapes happened to coincide. `min_replicas` and `initial_replicas` are excluded because they change dynamics and the starting state, not what any feature measures; a test requires every new `SimulatorConfig` field to be explicitly classified. `EnvironmentCompatibility.from_env(env)` or `.from_config(config)` captures it from a real `AutoscalingEnv`, and `trained.require_compatible(current)` raises with every mismatching field. Every run logs this contract as `compatibility.json`; DQN/PPO model loading (#15/#16) must check it before inference.

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

## Hyperparameter search (Optuna)

Responsibilities stay separate:

| Component | Answers |
|---|---|
| **Optuna** (`scalerl.tuning`) | What configuration should we try next? |
| **MLflow** (`scalerl.mlops`) | What exactly happened when we tried it? (canonical lineage) |
| **Benchmark manifest** (#18) | Which workloads may be used for tuning? (train/validation only) |
| **Scenario Lab** ([CITY_VIEW.md](CITY_VIEW.md)) | What does one simulation look like, interactively? |

Install with `pip install -e ".[tuning]"`, which brings Optuna 5.x and the `mlops` extra (MLflow), since trials log through `context.track`. `StudySpec` and `require_tuning_workloads` work without Optuna; `run_study` raises an `ImportError` with this install hint otherwise. ScaleRL does not use Optuna's deprecated `MLflowCallback`; trials log through `start_tracked_run` like every other run.

### Running a study

```python
from scalerl.tuning import StudySpec, TrialContext, run_study

spec = StudySpec(
    name="threshold-v1",
    objective_name="threshold-cost-sla",
    objective_version="v1",
    search_space_version="v1",
    direction="minimize",
    tuning_workload_ids=("syn-train-spike", "syn-val-bursty"),  # train/validation only
    sampler="grid",  # grid | tpe | random
    sampler_seed=42,
    grid={
        "high_threshold": [0.6, 0.7, 0.8],
        "low_threshold": [0.2, 0.3],
        "cooldown_ticks": [3, 5, 10],
    },
    storage="sqlite:///optuna.db",  # None = in-memory
)


def objective(context: TrialContext) -> float:
    high = context.trial.suggest_categorical("high_threshold", [0.6, 0.7, 0.8])
    ...
    scores = []
    for workload_id in spec.tuning_workload_ids:
        with context.track(run_spec_for(workload_id), experiment_name="threshold-v1") as run:
            ...  # evaluate one workload, log its metrics
            scores.append(score)
    return aggregate(scores)


study = run_study(spec, objective)
```

The objective and its selection metric belong to the consumer (#13, #15, #16). The shared layer never assumes that the highest RL reward is the best autoscaler; it records the objective name, objective version, search-space version, and value.

### One trial, many MLflow runs

MLflow keeps its one-workload-per-run lineage (#17). A trial that evaluates several workloads opens one run per workload through `context.track(run_spec)`, then aggregates them into the trial value:

```text
Optuna trial 7 (high=0.8, low=0.2, cooldown=5)
├── syn-train-spike   → MLflow run A
├── syn-val-bursty    → MLflow run B
└── azure-val-734400  → MLflow run C
aggregate objective   → trial 7 value
```

Links go both ways:

- the trial's `mlflow_run_ids` user attribute lists every run ID in order (recorded as soon as each run starts, so a failed trial still points to its runs);
- when the trial ends, **every** one of its runs is tagged `scalerl.optuna.trial_state` with the final Optuna state (`COMPLETE`, `PRUNED`, or `FAIL`), including runs that closed before the trial was pruned or failed;
- each run carries `scalerl.optuna.study` and `scalerl.optuna.trial` tags and `hp.optuna.*` params: study, trial, sampler, sampler seed, pruner, objective name/version, search-space version, storage (credentials stripped), and the trial's suggested parameters (`hp.optuna.params.*`).

`context.track` only accepts `train`/`tune` runs on the study's declared tuning workloads.

### Storage, resume, and determinism

- `storage=None` is in-memory; `sqlite:///optuna.db` persists locally. No Optuna server is needed.
- `n_trials` is the study's **total** budget. Re-running the same spec loads the existing study and runs only the remaining trials, so an interrupted study continues where it stopped without repeating finished trials. A grid study with `n_trials=None` runs until every combination is done.
- The study stores its definition (objective, versions, workloads, sampler, seed, grid, pruner). Resuming with a different definition is refused; use a new study name.
- Samplers are always seeded, and **a resumed study proposes exactly the same trials as an uninterrupted one**. Optuna does not persist a sampler's random state, so random and TPE sampling are seeded per trial from `(sampler_seed, trial number)`; the grid sampler reads visited combinations from storage. Without this, a resumed study would restart its random sequence and repeat earlier configurations.
- Trials run sequentially by default (`n_jobs=1`). `n_jobs > 1` is available explicitly, but parallel completion order can change the history that adaptive samplers such as TPE see.
- Multi-objective studies are deferred until a consumer needs them.

### Trial states

| Outcome | Optuna trial | MLflow run |
|---|---|---|
| objective returns | `COMPLETE` | `FINISHED` |
| objective raises | `FAIL` (error re-raised unless its type is in `run_study(..., catch=...)`) | `FAILED` if raised inside `track`, otherwise `FINISHED` |
| `context.prune()` | `PRUNED` | `FINISHED` |

Every run of the trial is also tagged `scalerl.optuna.trial_state` with that Optuna state, so the trial outcome is visible on each run regardless of its own status.

Pruning plumbing (`pruner="median"`, `context.report`, `context.should_prune`) is available but only meaningful once training exposes legitimate train/validation intermediate metrics (#15/#16). Never prune on held-out test results.

### Threshold study (#13)

The first consumer is the threshold baseline:

```bash
python -m scalerl.tuning.threshold \
    --storage sqlite:///outputs/threshold-optuna.db \
    --tracking-uri sqlite:///outputs/mlflow.db \
    --output outputs/threshold-v1.json
```

It runs the 18-point grid on the synthetic train/validation workloads (override with `--workload`, and pass `--azure-csv` for Azure train/validation entries), logs one MLflow run per workload per trial, selects with the `threshold-sla-first` v1 rule, and writes a `ThresholdTuningResult` JSON. `--simulator-config` with `--config-source` and `--calibration-workload` records a non-default simulator configuration's provenance. Re-running the command resumes the study without re-evaluating finished grid points. See [EXPERIMENTS.md](EXPERIMENTS.md#threshold-baseline-13).

### Tuning guardrails

- `StudySpec` rejects any `tuning_workload_ids` outside the manifest's train/validation splits; `azure-test-993600` or any other held-out workload fails immediately.
- Each tracked run is additionally validated by `RunSpec` (#17), including simulator-config provenance: calibrated configs cite train/validation workloads only.
- Final held-out evaluation is never part of an Optuna study.

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
├── pytest + coverage (installed with dev,mlops,dashboard,tuning, so MLflow,
│   Streamlit City View, and tiny Optuna study tests run on temporary SQLite stores)
├── sdist/wheel build
└── clean wheel install/import smoke (no extras: core, scalerl.mlops,
    scalerl.dashboard, and scalerl.tuning import without MLflow, Streamlit, or
    Optuna; packaged benchmark manifest loads)
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
- **Scenario Lab City View** ([CITY_VIEW.md](CITY_VIEW.md)): interactive, domain-specific understanding of one simulation: traffic, replicas, queue, latency/SLA, cost, and manager actions. It does not log to MLflow.
- **Results Explorer (#39, planned):** browsing stored MLflow runs.

The custom dashboard must not become a second experiment database.
