# MLOps Architecture

ScaleRL uses MLOps tooling to make controller training and benchmark results reproducible, inspectable, and traceable to the code/configuration that produced them.

## Principles

1. **No orphan results.** Any number promoted to the README or final report must be traceable to an MLflow run ID, or to a documented aggregate of run IDs.
2. **Held-out data stays held out.** Training/tuning runs must never use the frozen test suite from Issue #18.
3. **Core simulation remains headless.** The simulator/environment must not require MLflow, Docker, or a tracking server to run unit tests.
4. **One controller interface.** Baselines and learned policies use the same environment/evaluation path.
5. **Model compatibility is explicit.** DQN/PPO artifacts record the observation/action-space contract and the relevant environment configuration, including telemetry delay and the capacity-jitter model (#65). Jitter fraction and dynamics seed are evaluation conditions, not contract fields.

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

### DQN training runs and model bundles (#15)

```bash
# lightweight local
python -m scalerl.training.dqn --workload syn-train-bursty \
    --validation-workload syn-val-bursty --timesteps 200000 --seed 0 \
    --tracking-uri sqlite:///outputs/mlflow.db --output outputs/dqn-v1.json

# full stack: MLflow server, artifacts in Garage
docker compose run --rm trainer python -m scalerl.training.dqn \
    --workload syn-train-bursty --validation-workload syn-val-bursty --output outputs/dqn-v1.json
```

Other options include `--hyperparameters <json>` (a `DQNHyperparameters` file or a DQN tuning result), `--hp name=value` overrides, `--log-interval`, `--simulator-config`/`--config-source`/`--calibration-workload`, `--reward-weights`, and `--azure-csv` for Azure train/validation workloads.

- **Training run** (`run_kind=train`, `controller=dqn`):
  - `RunSpec` metadata as for every run, plus every DQN setting as `hp.*` (with `hp.dqn_config_version` and `hp.hyperparameter_source`) and `training_steps`;
  - a learning curve every `--log-interval` timesteps (default 1,000; about 200 points per 200k run), logged from SB3's public callback hooks:
    - `train/episode_reward_mean` (window) and `train/episode_reward_mean_100`;
    - `train/episodes`, `train/exploration_rate`, `train/loss`, `train/n_updates`;
    - a value SB3 has not produced yet is omitted, never logged as zero;
  - `training_timesteps`, `training_episodes`, `training_seconds`;
  - aggregate `validation.*` metrics and `scalerl/validation_summary.json`;
  - the model bundle under `model/`.
- **Validation runs:** one `evaluate` run per validation workload, with the shared `EpisodeMetrics` and tag `scalerl.model_source_run_id=<training run>`. They hold the raw validation metrics; the training run only aggregates them.
- **Model bundle** (`model/`), loadable anywhere with `scalerl.rl.load_sb3_controller(bundle_dir, env)`:
  - `model.zip`: SB3's native save, the canonical policy artifact;
  - `compatibility.json`: the `EnvironmentCompatibility` of the training environment;
  - `metadata.json`: algorithm, config version, benchmark version, training workload, seed, timesteps, hyperparameters, ScaleRL version, and training run ID.
- **Loading:** the loader derives the contract from the *current* environment and calls `require_compatible`. A mismatch in any recorded field (not just the observation shape) raises before the model is loaded. No MLflow Model Registry is used.
- **Result file:** the training result (`DQNTrainingResult` JSON in `outputs/`) records the run IDs, workloads, seed, timesteps, hyperparameters, compatibility, validation aggregates, and the model URI `runs:/<id>/model`. It contains no model bytes.
- **Tuning:** `python -m scalerl.tuning.dqn --train-workload … --validation-workload … --n-trials 20 --timesteps 200000` uses `--storage` (default `$OPTUNA_STORAGE_URI`, else `sqlite:///outputs/dqn-optuna.db`) and writes a `DQNTuningResult`. It includes the selected trial's hyperparameters, validation metrics, training run ID, and all its MLflow run IDs; see [EXPERIMENTS.md](EXPERIMENTS.md#dqn-15).

### PPO training runs (#16)

```bash
# lightweight local
python -m scalerl.training.ppo --workload syn-train-bursty \
    --validation-workload syn-val-bursty --timesteps 204800 --seed 0 \
    --tracking-uri sqlite:///outputs/mlflow.db --output outputs/ppo-v1.json

# full stack: MLflow server, artifacts in Garage
docker compose run --rm trainer python -m scalerl.training.ppo \
    --workload syn-train-bursty --validation-workload syn-val-bursty --output outputs/ppo-v1.json
```

PPO uses the same pipeline (`scalerl.training.common`), CLI options, run schema, and bundle loader as DQN, with the default experiment `scalerl-ppo` (runs `train-ppo-<workload>` and `evaluate-ppo-<workload>`). The differences:

- **Params:** `hp.algorithm=sb3-ppo`, `hp.ppo_config_version`, and `hp.observation_normalization=env-v1` / `hp.reward_normalization=none` alongside every PPO setting.
- **Learning curve:** it shares episodes, `episode_reward_mean(_100)`, `n_updates`, `loss`, and `learning_rate` with DQN, plus PPO's own values:
  - `policy_gradient_loss`, `value_loss`, `entropy_loss`;
  - `approx_kl`, `clip_fraction`, `clip_range`, `explained_variance`.
- **Checkpoints:** `checkpoints/step-NNNNNNN.zip` every `--checkpoint-interval` timesteps (default 51,200; must be a multiple of `n_steps`; `0` disables them), plus `checkpoints/manifest.json`.
- **Result file:** a `PPOTrainingResult` JSON, which also records rollouts, the normalization policy, and the checkpoints.

The final model is the same `model/` bundle with `metadata.json` `algorithm=ppo`. `scalerl.rl.load_sb3_controller` loads DQN and PPO bundles alike, picking the SB3 class from the metadata, and existing DQN bundles are unchanged. `python -m scalerl.tuning.ppo` (storage default `$OPTUNA_STORAGE_URI`, else `sqlite:///outputs/ppo-optuna.db`) writes a `PPOTuningResult`; see [EXPERIMENTS.md](EXPERIMENTS.md#ppo-16).

Both training runs also log the final update's values (loss, update count, …) at the final step, so the learning curve's last point reflects the finished model.

### Robustness evaluations (#65)

A tracked robustness evaluation (`evaluate_robustness_tracked`) is an `evaluate` run in the experiment `scalerl-robustness`, named `robustness-<controller>-<workload>-<scenario>-seed<n>`. It records:

- **Params:** `robustness_scenario` and `robustness_version`, and through the simulator config `sim.dynamics.capacity_jitter_fraction`, `sim.dynamics.telemetry_delay_ticks`, and `sim.dynamics.dynamics_seed`.
- **Contract:** `compat.telemetry_delay_ticks` and `compat.capacity_jitter_model`.
- **Tags:** `scalerl.robustness_scenario`, `scalerl.robustness_version`, `scalerl.dynamics_seed`, `scalerl.capacity_jitter_model`, and, where given, `scalerl.model_source_run_id` and `scalerl.robustness.perturbed_compatibility`.
- **Metrics:** the usual `EpisodeMetrics`, plus `dynamics.mean/min/max_capacity_multiplier`.
- **Artifact (optional):** raw physical step infos under `robustness/`.

Non-nominal scenarios are recorded with `simulator_config_source="predeclared"`: they are predeclared, never the default simulator. A `calibrated_train_validation` base config keeps its lineage (`calibration_workload_ids`/`calibration_note` are forwarded). `RunSpec` requires the scenario name and version together and accepts them **only on `evaluate` runs**, so a train or tune run can never be labeled as a robustness result. Existing non-robustness runs are unchanged. The perturbation tag is derived from the controller itself: an `SB3Controller` loaded with `robustness_evaluation=True` carries its perturbed fields, so a caller cannot forget them.

### Tracking location

No URL is hard-coded. `tracking_uri=None` honors `MLFLOW_TRACKING_URI` and MLflow's defaults, and an explicit `tracking_uri=` overrides both. The same code works with:

- local SQLite:

  ```bash
  export MLFLOW_TRACKING_URI=sqlite:///mlflow.db
  mlflow ui --backend-store-uri sqlite:///mlflow.db
  ```

  Artifacts are written to `./mlruns` under the working directory (both are gitignored);
- a local server: `mlflow server --host 127.0.0.1 --port 5000` with `MLFLOW_TRACKING_URI=http://127.0.0.1:5000`;
- the Docker Compose stack (#44): containers use `MLFLOW_TRACKING_URI=http://mlflow:5000` (PostgreSQL metadata, artifacts proxied to the S3-compatible Garage store), and the browser uses <http://localhost:5000>. See [DOCKER.md](DOCKER.md).

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
- In the Docker Compose stack the trainer gets `OPTUNA_STORAGE_URI=postgresql+psycopg://…@postgres:5432/optuna` (the `postgres` extra provides the driver), and the native Optuna Dashboard at <http://localhost:8080> reads the same database. `python -m scalerl.tuning.threshold` defaults `--storage` to `$OPTUNA_STORAGE_URI` when set, otherwise to local SQLite.
- `n_trials` is the study's **total** budget. Re-running the same spec loads the existing study and runs only the remaining trials, so an interrupted study continues where it stopped without repeating finished trials. A grid study with `n_trials=None` runs until every combination is done.
- The study stores its definition (objective, versions, workloads, sampler, seed, grid, pruner, and `identity_context`). Resuming with a different definition is refused, naming what changed; use a new study name. Consumers put every other experiment-defining input in `identity_context`, such as the simulator config, its provenance, and a fingerprint of the workload data.
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

It runs the 18-point grid on the synthetic train/validation workloads (override with `--workload`, and pass `--azure-csv` for Azure train/validation entries), logs one MLflow run per workload per trial, selects with the `threshold-sla-first` v1 rule, and writes a `ThresholdTuningResult` JSON. `--simulator-config` with `--config-source` and `--calibration-workload` records a non-default simulator configuration's provenance. Re-running the command resumes the study without re-evaluating finished grid points. The study's identity includes the full simulator config, config source, calibration workload IDs, reward weights, and a SHA-256 fingerprint of the built workload traces, so a re-run with a different config or different workload data (for example another Azure file) is refused instead of mixing or reusing stale trials. See [EXPERIMENTS.md](EXPERIMENTS.md#threshold-baseline-13).

### Tuning guardrails

- `StudySpec` rejects any `tuning_workload_ids` outside the manifest's train/validation splits; `azure-test-993600` or any other held-out workload fails immediately.
- Each tracked run is additionally validated by `RunSpec` (#17), including simulator-config provenance: calibrated configs cite train/validation workloads only.
- Final held-out evaluation is never part of an Optuna study.

## Docker

Two container concerns are intentionally separate:

- **Issue #44 (done):** the local full stack in [DOCKER.md](DOCKER.md): the ScaleRL runtime image (Scenario Lab + trainer), the MLflow server with PostgreSQL metadata and an S3-compatible artifact store (Garage), and the Optuna Dashboard on PostgreSQL.
- **Issue #24:** later production-inspired inference/demo image.

Lightweight SQLite mode stays first-class; Compose is an additional mode. No image contains the raw Azure dataset: it is bind-mounted read-only at runtime. `scripts/compose-smoke.sh` verifies a real tracked run, artifact round-trip through the store, and Optuna Dashboard connectivity.

## CI/CD

GitHub Actions is ScaleRL's reproducibility gate. Every pull request and every push to `main` runs [`ci.yml`](../.github/workflows/ci.yml):

```text
quality (Python 3.12) ──────┐  Ruff lint, Ruff format check, mypy --strict
tests (Python 3.11, 3.12) ──┼─→ docker-mlops (ubuntu, native linux/amd64, 30 min cap)
package (Python 3.12) ──────┘    ├── setup-local-stack.sh (.env from .env.example, runner UID/GID)
                                 ├── docker compose config
                                 ├── Bake build of compose.yaml images (GitHub Actions cache)
                                 ├── scripts/compose-smoke.sh (same script as locally)
                                 ├── on failure: compose ps + logs uploaded as `compose-logs`
                                 └── always: docker compose down --volumes
```

| Job | Guarantees |
|---|---|
| `quality` | `ruff check .`, `ruff format --check .`, `mypy src` (strict), with all extras installed so every code path is type-checked |
| `tests` | pytest with coverage (terminal report + `coverage.xml` artifact) on Python 3.11 and 3.12. MLflow, Streamlit, and Optuna tests run on temporary SQLite stores; the Compose config is checked statically. No coverage threshold is enforced |
| `package` | `python -m build` builds sdist and wheel. The wheel alone (no extras) is installed in a clean venv and used from outside the checkout. It must import from site-packages, report the `pyproject.toml` version through `importlib.metadata`, load the packaged benchmark manifest, and import core, `scalerl.mlops`, `scalerl.dashboard`, and `scalerl.tuning` without importing MLflow, Streamlit, or Optuna |
| `docker-mlops` | Runs only after the three jobs above pass. It builds the ScaleRL runtime, MLflow, and Optuna Dashboard images from `compose.yaml` (PostgreSQL and Garage are pulled), starts the real stack from empty volumes, and runs [`scripts/compose-smoke.sh`](../scripts/compose-smoke.sh), listed below |

The Compose smoke covers:

- a real ScaleRL tracked run (predictive baseline on a synthetic TRAIN workload), with params, metrics, and `scalerl/` artifacts;
- those artifacts downloaded back through MLflow's proxy, i.e. MLflow → PostgreSQL metadata and MLflow → Garage artifacts;
- a temporary Optuna study in the PostgreSQL `optuna` database, seen by the native Optuna Dashboard, then deleted;
- Scenario Lab health and trainer write access to `outputs/`;
- the [SB3 smoke](../scripts/sb3_smoke.py) in the trainer image, for **DQN and PPO**: SB3's env checker on the real `AutoscalingEnv`, then a CPU `DQN` (`MlpPolicy` 32×32, `learning_starts=0`, `train_freq=1`, `batch_size=16`, `buffer_size=256`, seed 0). It learns for 64 timesteps, so 64 gradient updates must change the Q-network weights. It then checks that the prediction drives the env and that a save/load round-trip predicts the same action. This is infrastructure only, with no performance assertion. PPO then runs a tiny CPU `PPO` (32-step rollouts, batch 16, 1 epoch, actor/critic 32×32, 64 timesteps) and must change both actor and critic weights, drive the env, and survive save/load. Both run in about a second, with no performance assertion.

CI uses no repository secrets, no Azure data (`data/raw` stays empty), no GPU, and no paid services. Docker layers are cached with BuildKit's GitHub Actions cache, but a cache miss just rebuilds from the hash-pinned requirements, so correctness never depends on the cache. No PostgreSQL, Garage, or MLflow state is cached: every run starts from empty volumes, so first-time initialization is exercised too. A newer commit on the same PR cancels the older run; `main` and release runs are never cancelled.

**Releases.** Pushing a tag `vX.Y.Z` runs [`release.yml`](../.github/workflows/release.yml):

- It first re-runs the whole `ci.yml` gate on the tagged commit (as a reusable workflow).
- It then checks that the tag matches the `pyproject.toml` version.
- Finally it publishes the ScaleRL runtime/trainer image, built from the same `Dockerfile`, to GHCR for `linux/amd64` and `linux/arm64` (arm64 through QEMU):
  - tags `ghcr.io/wisoums/scalerl:X.Y.Z`, `:X.Y`, and `:sha-<short>`, with no automatic `latest`;
  - OCI labels for source, revision, and version;
  - SLSA provenance (`mode=max`) and an SBOM.

Only that publish job has `packages: write`, through `GITHUB_TOKEN` (no PAT); every other job is `contents: read`. Pull requests (including forks) and pushes to `main` never publish. PostgreSQL and Garage are upstream images and are not republished; the MLflow and Optuna Dashboard images are infrastructure and are built locally.

Benchmark-scale RL training, Optuna tuning grids, Azure workloads, and multi-seed runs never run in CI.

## Results UI responsibilities

- **MLflow UI:** experiment/run tracking, parameters, metrics, artifacts, model lineage.
- **Scenario Lab City View** ([CITY_VIEW.md](CITY_VIEW.md)): interactive, domain-specific understanding of one simulation: traffic, replicas, queue, latency/SLA, cost, and manager actions. It does not log to MLflow.
- **Results Explorer (#39, planned):** browsing stored MLflow runs.

The custom dashboard must not become a second experiment database.
