# Experimental Methodology

ScaleRL evaluates every controller under the same simulator configuration and workload traces.

## Controllers

- Random policy — sanity check
- Static capacity — fixed-cost reference
- Threshold/target-tracking autoscaler — tuned reactive baseline
- Predictive autoscaler — forecasting baseline
- Tabular Q-learning — optional educational learned baseline
- DQN — discrete deep-RL policy
- PPO — policy-gradient comparison

Random/static are sanity/reference points. The tuned threshold and predictive controllers are the meaningful non-RL bars for deep RL.

### Controller information sets

| Controller | Decides from |
|---|---|
| Random / Static | nothing / current replica counts |
| Threshold | current utilization and replica counts (reactive, no history) |
| Predictive (#14) | its own explicit forecast from past demand |
| DQN / PPO | the environment observation, including the recent traffic window below |

### Recent traffic context

For v1, learned policies receive the latest four completed demand observations, newest first: `t`, `t−1`, `t−2`, `t−3` (`observation.traffic_history_ticks = 4`). At the 30-second benchmark interval this is two minutes of short-term traffic context.

No future workload values are exposed: the window only contains demand from ticks that have already run. It lets learned policies recognize short-term trends (rising, falling, stable demand) but does not make unpredictable future spikes knowable. The predictive baseline (#14) remains the explicit forecasting controller.

The history length is a predeclared v1 design choice, not tuned, and never chosen on held-out test workloads. It is part of model compatibility, so the main comparison holds it fixed across learned controllers.

## Workloads

### Synthetic

- steady
- diurnal/seasonal
- gradual ramp
- abrupt spike
- bursty stochastic traffic
- recovery after overload

### Real production traces

Issue #43 adds selected slices from the Microsoft Azure Functions Invocation Trace 2021. The loader returns the same `WorkloadTrace` abstraction as synthetic generators, so controllers and evaluation code do not special-case real data.

Dataset source/license/citation are documented in `data/README.md`.

## Train / validation / test split

Before deep-RL hyperparameter tuning, Issue #18 freezes an explicit workload manifest ([`benchmarks/v1/`](../benchmarks/v1/README.md)):

- **training:** policy learning;
- **validation:** threshold/predictive/RL tuning and model selection;
- **held-out test:** final reporting only.

Held-out traces must not be used for threshold grid search, reward tuning, DQN/PPO hyperparameter selection, early stopping decisions, or model selection.

The main controller comparison keeps the simulator configuration fixed so learned policies have a compatible observation/action space.

## Experiment tracking

Every training, tuning, and evaluation run is tracked in MLflow through `scalerl.mlops.start_tracked_run` (see [MLOPS.md](MLOPS.md)). A run records its controller, run kind, benchmark workload and split, seeds, Git commit and software versions, the complete simulator configuration, reward weights, hyperparameters, and the observation/action compatibility contract.

Any number promoted to the README or final results must be traceable to an MLflow run ID, or to a documented aggregate of run IDs.

### Simulator configuration and calibration

Each run records not only the simulator configuration but **why** it has those values (`simulator_config_source`: `default`, `predeclared`, or `calibrated_train_validation` with the calibration workload IDs).

Azure capacity or other simulator calibration may use **train and validation workloads only**. Held-out test statistics must not drive:

- capacity (`service_capacity_rps`);
- replica bounds;
- reward weights;
- threshold settings;
- predictive settings;
- DQN/PPO hyperparameters.

Tracking enforces this: calibration workload IDs from the test split are rejected, and `train`/`tune` runs cannot target test workloads. Azure demand itself is never normalized or rescaled.

## Hyperparameter tuning

Threshold, predictive, DQN, and PPO tuning run as Optuna studies through `scalerl.tuning` (see [MLOPS.md](MLOPS.md#hyperparameter-search-optuna)):

- Optuna proposes configurations; MLflow records every evaluation; the benchmark manifest decides which workloads may be used.
- Studies may use **train and validation workloads only**. Held-out test workloads are rejected by `StudySpec` and never participate in tuning, pruning, or model selection.
- A trial that evaluates several workloads creates one MLflow run per workload and aggregates them into its objective; the trial records all of its run IDs and every run records its study and trial.
- Each consumer (#13, #15, #16) defines and versions its own objective, e.g. a documented combination of or constraint on cost, SLA violation rate, p95 latency, queue, churn, and reward. Highest reward alone is not assumed to be best.
- Studies are seeded, sequential by default, and resumable from local SQLite storage.

## Fair comparison

For every final comparison:

- same simulator configuration;
- same held-out traces;
- same evaluation seeds;
- same metric definitions;
- no best-seed cherry-picking.

## Metrics

Primary:

- p95 latency proxy / latency summaries
- SLA violation rate/count
- infrastructure cost / replica-hours
- queued/completed/dropped requests
- scaling-action count/churn

Secondary:

- episodic return
- queue depth
- learning curves/sample efficiency
- action distribution

Reward is never used as the only success measure.

## MLflow run contract

MLflow is the canonical experiment tracker. See `docs/MLOPS.md`.

Every reported run/aggregate records enough information to reproduce it, including:

- Git commit;
- controller/algorithm;
- simulator config;
- workload identifier and split;
- seeds;
- reward weights;
- observation/action-space metadata;
- hyperparameters/training steps;
- raw system metrics;
- model/evaluation artifacts when applicable.

Any number promoted to the README/final report must be traceable to an MLflow run ID or documented aggregate of run IDs.

## Reproducibility

Final comparisons use multiple seeds and retain per-seed raw results. Summary statistics report dispersion rather than only a single best run.

CI verifies code/package/container/training plumbing with short smoke jobs. Benchmark-scale RL training is intentionally outside CI.

## Success criterion

The project is successful even if RL does not dominate every baseline. The research objective is to identify when learned sequential policies are useful, when simple controllers are sufficient, and what trade-offs cause each result.
