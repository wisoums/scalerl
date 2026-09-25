# Experimental Methodology

ScaleRL evaluates every controller under the same simulator configuration and workload traces.

## Controllers

- Random policy — sanity check
- Static capacity — fixed-capacity reference
- Threshold/target-tracking autoscaler — tuned reactive baseline
- Predictive autoscaler — explicit forecasting baseline
- Tabular Q-learning — optional educational learned baseline
- DQN — discrete deep-RL policy
- PPO — policy-gradient comparison

Random/static are sanity/reference points. The tuned threshold and predictive controllers are the meaningful non-RL bars for deep RL.

### Controller information sets

| Controller | Decides from |
|---|---|
| Random / Static | nothing / current replica counts |
| Threshold | current utilization and replica counts (reactive, no history) |
| Predictive (#14) | a linear-trend forecast of its own window of past completed demand (`info["request_rate"]`), plus replica counts |
| DQN / PPO | the environment observation (recent traffic window + system state); learned sequential policies |

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

## Threshold baseline (#13)

The tuned threshold controller is the **main reactive baseline** that DQN and PPO must beat; random and static controllers are only sanity checks.

- **Reactive rule:** scale one replica up when utilization of active capacity is above the high threshold, down when below the low threshold, counting pending replicas toward desired capacity.
- **Cooldown (anti-thrashing):** after a step whose applied replica change is non-zero, the next `cooldown_ticks` controller decisions hold (reason `cooldown`). Bound-clipped requests (applied change 0) never start a cooldown, and capacity outside the controller's bounds is still corrected during cooldown. Cooldown is symmetric for scale-up and scale-down in v1, matching the single `cooldown_ticks` grid parameter. On a workload oscillating across both thresholds every tick, cooldown 3 reduced applied scaling changes from 59 to 14 of 60 ticks.
- **Tuning:** `python -m scalerl.tuning.threshold` evaluates the exact grid high {0.6, 0.7, 0.8} × low {0.2, 0.3} × cooldown {3, 5, 10} (18 configurations) with Optuna's grid sampler, on the benchmark's synthetic train and validation workloads by default. Each workload evaluation is its own MLflow run; each trial stores its validation aggregates.
- **Selection rule `threshold-sla-first` v1** (validation workloads only): lowest mean SLA violation rate, then lowest mean normalized cost (episode cost ÷ cost of `max_replicas` for the whole episode), then lowest mean queue pressure (`queued / (queued + max-fleet tick capacity)`), then lowest churn rate (ticks with an applied change ÷ episode ticks), then the smallest `(high, low, cooldown)` as an exact tie-break. The Optuna objective is the mean validation SLA violation rate.
- Reward is logged as a secondary metric only and never selects the configuration. Held-out test workloads are rejected before any evaluation. Only the thresholds and cooldown are tuned; simulator settings, reward weights, replica bounds, and `traffic_history_ticks` stay fixed.
- The selected configuration, rule, validation metrics, study/trial identity, and MLflow run IDs are written to a JSON result (`ThresholdTuningResult`) and to the study's user attributes. Study databases and results live in ignored local paths (`outputs/`), not Git.

## Predictive baseline (#14)

The predictive controller asks: if recent demand has a visible trend, can a simple, transparent forecast request capacity early enough to beat replica startup delay, where a reactive threshold would be late?

v1 design (predeclared, not tuned):

- **History:** the last 4 completed request rates, kept by the controller itself from `info["request_rate"]`. It never reads the trace, the RL observation, or generator parameters.
- **Forecast:** one sample → persistence; two or more → an ordinary least-squares line through `(tick, rate)`, extrapolated; negative forecasts are clamped to 0.
- **Horizon:** `1 + ceil(startup_delay / control_interval)` ticks after the latest sample, i.e. the first tick a replica requested now can serve (3 ticks, 90 s, for the v1 60 s / 30 s timing). It uses the simulator's own startup arithmetic and is verified against the environment.
- **Capacity:** `desired = ceil(forecast / (service_capacity_rps × 0.8))` (target utilization 0.8 leaves headroom), clamped to the replica bounds.
- **Action:** compare with committed capacity `active + pending`: scale up if below, down if above, else hold. Pending replicas count, so capacity already starting is not requested again, and a scale-down cancels the newest pending replica first.
- **No look-ahead:** forecasts use only completed ticks. A truly random spike with no prior trend cannot be forecast; the controller reacts only once the spike is part of its history.

Behavior on the synthetic suite, with the default simulator config: on a clean ramp it requests capacity well before a reactive threshold (first scale-up after tick 2 vs 15), at slightly higher cost. **Known limitation:** capacity is sized for forecast *arrivals* only, not queued backlog, and a burst's falling edge drives the trend down; on bursty workloads it under-provisions and SLA violations stay high. Whether to add backlog awareness is a separate design decision, not part of v1.

**Forecast accuracy** is scored only after an episode, by joining each forecast's target tick to that tick's actual request rate (forecasts past the episode end are ignored): count, MAE, RMSE, and mean signed error (bias), all in RPS. MAPE is not used because demand can be near zero.

`python -m scalerl.evaluation.predictive` evaluates it on the synthetic train/validation workloads (held-out workloads are rejected), one MLflow run per workload (`run_kind="evaluate"`, `controller="predictive"`) with its settings, forecast accuracy, and the same system metrics as the threshold study (shared `scalerl.evaluation` metrics).

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
