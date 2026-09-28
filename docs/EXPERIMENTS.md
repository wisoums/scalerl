# Experimental Methodology

ScaleRL evaluates every controller under the same simulator configuration and workload traces.

## Controllers

- Random policy — sanity check
- Static capacity — fixed-capacity reference
- Threshold/target-tracking autoscaler — tuned reactive baseline
- Predictive autoscaler (`predictive-v1`) — explicit short-window forecasting baseline
- Cloud-style proactive predictive (`predictive-seasonal-v1`, #80) — historical profile + linear trend, proactive scale-out, conservative scale-in
- Tabular Q-learning — optional educational learned baseline
- DQN — discrete deep-RL policy
- PPO — policy-gradient comparison

Random/static are sanity/reference points. The tuned threshold and predictive controllers are the meaningful non-RL bars for deep RL.

### Controller information sets

| Controller | Decides from |
|---|---|
| Random / Static | nothing / current replica counts |
| Threshold | current utilization and replica counts (reactive, no history) |
| Predictive (#14, #63) | a linear-trend forecast of its own window of past completed demand (`info["request_rate"]`), plus the current waiting queue (`info["queued_requests"]`, sizing only, never the forecast) and replica counts |
| Proactive predictive (#80) | the same completed demand and queue as Predictive, plus an immutable per-tick median profile of TRAIN traces (when recurring history exists); never the current trace or generator parameters |
| DQN (#15) / PPO (#16) | **only** the environment observation (recent traffic window + system state) through the `SB3Controller` adapter, which ignores `info`; learned sequential policies |

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
- **Capacity (#63, policy `forecast-plus-backlog-v1`):** `backlog_recovery_rps = queued_requests / control_interval_seconds` (clear the current queue in one tick, a fixed rule: `backlog_recovery_ticks = 1`), `effective_demand_rps = forecast_rps + backlog_recovery_rps`, and `desired = ceil(effective_demand_rps / (service_capacity_rps × 0.8))` (target utilization 0.8 leaves headroom), clamped to the replica bounds. With an empty queue this is exactly the #14 forecast-only sizing.
- **Action:** compare with committed capacity `active + pending`: scale up if below; if above, scale down only when no requests are queued, otherwise hold (`backlog_hold`); else hold. Pending replicas count, so capacity already starting is not requested again, and a scale-down cancels the newest pending replica first. The reason `queue_recovery` marks a scale-up that the forecast alone would not have made.
- **No look-ahead:** forecasts use only completed ticks. A truly random spike with no prior trend cannot be forecast; the controller reacts only once the spike is part of its history.

Behavior on the synthetic suite, with the default simulator config: on a clean ramp it requests capacity well before a reactive threshold (first scale-up after tick 2 vs 15), at slightly higher cost.

**Forecast vs backlog (#63).** The forecast answers "how many *new* requests will arrive?"; the queue is work that already arrived and was not served. The #14 version sized capacity for forecast arrivals only, so on a burst's falling edge the trend dropped, it scaled down while thousands of requests were still waiting, and bursty SLA violations stayed around 0.93–0.98. The forecast itself is unchanged by #63 (same method, history, horizon, and forecast records and scores); only capacity sizing adds the backlog term. The backlog term uses nothing beyond the latest completed tick's queue, so it cannot see the future either: a random spike is still unpredictable, and queue recovery only helps once overload has built a queue.

Development comparison (default simulator config, synthetic train/validation workloads only, one deterministic episode each; forecast-only is a test-only reference of the #14 rule, not a public option):

| Workload | Policy | SLA violation rate | Mean queue | Max queue | Normalized cost | Mean active replicas | Scaling actions |
|---|---|---:|---:|---:|---:|---:|---:|
| syn-train-bursty | forecast-only (#14) | 0.983 | 24,393 | 62,939 | 0.340 | 2.82 | 79 |
| syn-train-bursty | queue-aware (#63) | 0.450 | 4,312 | 27,784 | 0.512 | 4.34 | 89 |
| syn-val-bursty | forecast-only (#14) | 0.933 | 10,561 | 29,550 | 0.389 | 3.23 | 92 |
| syn-val-bursty | queue-aware (#63) | 0.408 | 4,322 | 28,335 | 0.507 | 4.29 | 96 |
| syn-train-spike | forecast-only (#14) | 0.192 | 4,368 | 54,900 | 0.266 | 2.51 | 17 |
| syn-train-spike | queue-aware (#63) | 0.150 | 4,163 | 54,900 | 0.297 | 2.79 | 21 |
| syn-val-ramp-down | forecast-only (#14) | 0.208 | 5,804 | 47,453 | 0.520 | 5.07 | 15 |
| syn-val-ramp-down | queue-aware (#63) | 0.150 | 4,435 | 47,453 | 0.536 | 5.21 | 17 |
| syn-train-ramp-up | both (queue stays empty) | 0.000 | 0 | 0 | 0.562 | 5.47 | 9 |

Trade-off: on bursty workloads, queue-aware sizing roughly halves SLA violations and cuts the mean queue 2.4–5.7×, for about 30–50% higher cost; on the other workloads it helps a little for a few percent more cost, and a workload that never queues behaves exactly as before. Peak queues on a sudden spike are unchanged, because the spike was not predictable and replicas still take startup time. Bursty SLA violations remain high (about 0.4), so this is a fairer baseline, not a solved workload.

**Forecast accuracy** is scored only after an episode, by joining each forecast's target tick to that tick's actual request rate (forecasts past the episode end are ignored): count, MAE, RMSE, and mean signed error (bias), all in RPS. MAPE is not used because demand can be near zero.

`python -m scalerl.evaluation.predictive` evaluates it on the synthetic train/validation workloads (held-out workloads are rejected), one MLflow run per workload (`run_kind="evaluate"`, `controller="predictive"`) with its settings (including `hp.capacity_policy = "forecast-plus-backlog-v1"` and `hp.backlog_recovery_ticks = 1`), forecast accuracy, and the same system metrics as the threshold study (shared `scalerl.evaluation` metrics).

### Predictive realism gap and #80

The current Predictive controller is already **startup-delay-aware**: it forecasts far enough ahead for newly requested capacity to become ready. It is therefore more than a reactive threshold policy.

It is **not**, however, equivalent to a production cloud predictive-scaling service. Its forecast uses only a short recent window and a linear trend. Cloud predictive scaling can use much longer historical patterns and schedule capacity ahead of forecasted demand. AWS documents both recurring-pattern forecasting and a `SchedulingBufferTime` that advances launch time so capacity can be ready before the forecasted load arrives:

- <https://docs.aws.amazon.com/autoscaling/application/userguide/aas-predictive-scaling-how-it-works.html>
- <https://docs.aws.amazon.com/autoscaling/ec2/userguide/ec2-auto-scaling-predictive-scaling.html>

Issue #80 therefore adds a stronger **cloud-style proactive predictive baseline** while preserving the existing `predictive-v1` controller/manifest identity, whose current forecast method is `linear-trend` and capacity policy is `forecast-plus-backlog-v1`. The stronger baseline must remain past-only, startup-aware, queue-aware, reproducible, and frozen on train/validation evidence before #46.

## DQN (#15)

DQN is ScaleRL's first deep-RL controller, using Stable-Baselines3's DQN. It fits the problem directly:

- **Actions:** `Discrete(3)` under the historical `delta-v1` contract: 0 scale down, 1 hold, 2 scale up. DQN learns one Q-value per action and, when evaluated, picks the highest. #79 adds the discrete `desired-replicas-v1` contract (`Discrete(10)` at the default bounds); DQN is equally valid there, since replica counts are discrete.
- **Observation:** the frozen v1 vector from #58, unchanged. Its 12 values (default config) are:
  - the 4 most recent completed request rates, newest first;
  - utilization, queue pressure, latency pressure, active replicas, cost, and episode progress;
  - pending-replica readiness buckets.
- **Reward:** exactly `AutoscalingEnv`'s reward, with the default `RewardWeights`. There is no DQN-specific reward; reward studies are #20.
- **What it cannot see:** the trace, `info`, generator parameters, benchmark metadata, or anything in the future. SB3 only receives `obs, reward, terminated/truncated` through the Gymnasium API. Out-of-bound scale requests are clamped by the environment, as for every controller.

**Training vs evaluation.** Training explores (ε-greedy) on one TRAIN workload. Evaluation loads the saved final policy and predicts with `deterministic=True`, with no further learning. It runs through the shared `run_episode` / `evaluate_controller_episode` and reports the same `EpisodeMetrics` as every baseline. The `SB3Controller` adapter implements the common `Controller` protocol and ignores `info`. Unlike Threshold and Predictive, which read documented raw values, it cannot see anything but the observation.

**Data rules in #15.** `python -m scalerl.training.dqn` trains on exactly one TRAIN workload (one MLflow `train` run) and validates on VALIDATION workloads only (one MLflow `evaluate` run each, tagged `scalerl.model_source_run_id`). Validation workloads, and held-out test workloads in either role, are refused before any trace is built or model trained. There is no multi-workload curriculum in v1. Held-out evaluation is #46.

**`dqn-v1` hyperparameters** are predeclared, not tuned, and not claimed optimal. They are SB3's DQN defaults, adapted to a 200k-step budget on 120-tick episodes:

| Setting | `dqn-v1` | SB3 default |
|---|---|---|
| learning_rate | 5e-4 | 1e-4 |
| buffer_size | 100,000 | 1,000,000 |
| learning_starts | 1,000 | 100 |
| batch_size | 64 | 32 |
| gamma | 0.99 | 0.99 |
| train_freq / gradient_steps | 4 / 1 | 4 / 1 |
| target_update_interval | 1,000 | 10,000 |
| exploration_fraction | 0.2 | 0.1 |
| exploration eps initial → final | 1.0 → 0.05 | 1.0 → 0.05 |
| net_arch | [64, 64] | [64, 64] |

The default budget is **200,000 timesteps**, about 1,667 episodes, which takes about 30 s on a laptop CPU. Budgets must be a multiple of `train_freq` (4 in `dqn-v1`): SB3's off-policy loop only checks the budget between `train_freq`-step rollouts, so 97 steps would silently train 100. ScaleRL refuses such budgets before training, so the recorded `training_steps`, the bundle metadata, and the model always agree. Tests and CI use tiny budgets (≤ 256 timesteps; the CI smoke uses 64) and make no performance assertions.

**Seeding.** One explicit seed drives SB3's network initialization, exploration, and environment resets. Runs with the same seed and config reproduce on the same platform and library versions. ScaleRL does not claim bit-identical results across PyTorch versions or hardware.

**Tuning (`dqn-search-v1`).** `python -m scalerl.tuning.dqn` runs a #54 Optuna study:

- **Sampler:** seeded TPE (`sampler_seed=42`, trial-seeded so resumes are exact), trials sequential, no pruning.
- **Each trial:** trains with the trial's settings on the declared TRAIN workload, using the same training seed for every trial, then validates the final policy on the declared VALIDATION workloads. All of a trial's runs are `run_kind="tune"`, tracked through `TrialContext.track`.
- **Search space**, predeclared from standard SB3/DQN ranges and never from held-out results:

| Parameter | Range |
|---|---|
| learning_rate | log-uniform 1e-5 – 1e-3 |
| gamma | {0.95, 0.98, 0.99, 0.995} |
| batch_size | {32, 64, 128} |
| target_update_interval | {250, 1,000, 5,000} |
| exploration_fraction | uniform 0.05 – 0.5 |
| exploration_final_eps | uniform 0.01 – 0.1 |
| net_arch | {64×64, 128×128, 256×256} |

Buffer size, learning starts, train frequency, gradient steps, initial epsilon, the algorithm, action space, observation, and reward stay fixed.

**Selection rule `dqn-sla-first` v1** (VALIDATION only; the Optuna objective is step 1):

1. mean SLA violation rate;
2. then mean normalized cost;
3. then mean queue pressure;
4. then mean churn;
5. then the lower trial number.

Episode reward is recorded but never selects. The selected configuration is **retrained** as the `train` run of record with `python -m scalerl.training.dqn --hyperparameters <DQNTuningResult JSON>` (recorded as `hyperparameter_source=optuna:<study>#trial<n>`). No trial model silently becomes the official model.

**Scope of #15.** It establishes that DQN can be trained reproducibly, tuned fairly on train/validation data, and evaluated identically to the baselines. It makes **no claim that DQN beats any baseline**; performance conclusions come from later multi-seed and held-out evaluation (#19, #46).

## PPO (#16)

PPO is ScaleRL's second deep-RL controller, using Stable-Baselines3's PPO. It shares DQN's contract through `scalerl.training.common`: the same observation, reward, splits, MLflow schema, model bundle, compatibility check, deterministic evaluation, and SLA-first tuning.

**Actor/critic, not Q-values.** DQN learns `Q(state, action)` for the three actions. PPO learns two networks:

- an **actor** π(action | state), a probability over scale down / hold / scale up;
- a **critic** V(state), the expected future return, which turns observed returns into advantages for the actor's update.

In ScaleRL they are separate MLPs of the same widths: 12 → 64 → 64 with tanh, then 3 action logits (actor) or 1 value (critic).

Training repeats one loop: collect a rollout of `n_steps` transitions with the current stochastic policy, estimate advantages (GAE), update both networks for `n_epochs` over minibatches, then collect again. There is no replay buffer, target network, or epsilon exploration: PPO explores by sampling its own policy. Evaluation uses the most likely action (`deterministic=True`) through the same `SB3Controller`, which ignores `info`.

**Same inputs as DQN.** PPO uses the frozen v1 observation and exactly `AutoscalingEnv`'s reward. It gets no PPO-specific features or reward, and reward studies stay in #20.

**Normalization, made explicit:**

| | v1 choice | Why |
|---|---|---|
| Observation | `AutoscalingEnv`'s own normalization (`hp.observation_normalization=env-v1`); **no** SB3 `VecNormalize` | features are already normalized; no extra running statistics to save or mismatch |
| Reward | the ScaleRL reward as is (`hp.reward_normalization=none`); **no** `VecNormalize` reward scaling | PPO and DQN learn from identical reward semantics |
| Advantages | PPO's per-minibatch `normalize_advantage=True` (logged as `hp.normalize_advantage`) | an optimizer detail inside PPO's loss; it changes neither observations nor rewards |

**`ppo-v1` hyperparameters** are SB3's PPO defaults: predeclared, not tuned, and not claimed optimal.

| Setting | `ppo-v1` |
|---|---|
| learning_rate | 3e-4 |
| n_steps (rollout size, 1 env) | 2,048 |
| batch_size | 64 (32 minibatches per epoch, dividing the rollout exactly) |
| n_epochs | 10 |
| gamma / gae_lambda | 0.99 / 0.95 |
| clip_range | 0.2 |
| ent_coef / vf_coef | 0.0 / 0.5 |
| max_grad_norm | 0.5 |
| normalize_advantage | true |
| net_arch | actor [64, 64], critic [64, 64] |

**Budget.** SB3 PPO trains whole rollouts, so a budget that is not a multiple of `n_steps` silently trains longer; ScaleRL refuses such budgets before training.

- The default is **204,800 timesteps**: exactly 100 rollouts of 2,048 steps and about 1,700 episodes, taking about 35 s on a laptop CPU.
- It is deliberately close to DQN's 200,000 but not equal, because 200,000 is not a whole number of 2,048-step rollouts.
- The budget is predeclared and logged, not adjusted after seeing results.

**Checkpoints.** A checkpoint is saved every 51,200 timesteps (25 rollouts; configurable, off with `--checkpoint-interval 0`).

- Checkpoints are saved right after an update, so `checkpoints/step-N.zip` is the policy after exactly N timesteps of learning.
- `checkpoints/manifest.json` lists them.
- Checkpoints are intermediate artifacts; the final `model/` bundle is the model of record. No checkpoint is selected by validation in #16.

**Tuning (`ppo-search-v1`).** `python -m scalerl.tuning.ppo` runs the same seeded-TPE study runner as DQN (sampler seed 42, sequential trials, no pruning, `run_kind="tune"`, same training seed for every trial). The search space was predeclared from standard SB3/PPO ranges, never from held-out results:

| Parameter | Range |
|---|---|
| learning_rate | log-uniform 1e-5 – 1e-3 |
| gamma | {0.95, 0.98, 0.99, 0.995} |
| gae_lambda | {0.9, 0.95, 0.98} |
| clip_range | {0.1, 0.2, 0.3} |
| ent_coef | {0.0, 0.001, 0.01} |
| n_steps | {512, 1,024, 2,048} |
| batch_size | {64, 128, 256} |
| n_epochs | {5, 10, 20} |
| net_arch (actor = critic) | {64×64, 128×128, 256×256} |

Every `batch_size` divides every `n_steps`, so no minibatch is ever truncated. The trial budget must be a multiple of 2,048 (the least common multiple of the rollout sizes; the default 204,800 is), so every trial trains exactly the same number of timesteps. `vf_coef`, `max_grad_norm`, advantage normalization, and the normalization policy stay fixed.

**Selection rule `ppo-sla-first` v1.** It is the same as DQN's, so the two selection philosophies are comparable: on VALIDATION only, lowest mean SLA violation rate, then cost, then queue pressure, then churn, then trial number. Reward is recorded but never selects. Retrain the selection as the `train` run of record with `python -m scalerl.training.ppo --hyperparameters <PPOTuningResult JSON>`.

**Scope of #16.** It establishes that PPO trains, tunes, and evaluates under the same contract as DQN and the baselines. It makes **no claim that PPO beats DQN or any baseline**; comparisons come from tuned, multi-seed, and held-out evaluation (#19, #46).

## Robustness scenarios (#65)

Do controller results hold up when the world is less ideal? #65 adds two controlled, seeded perturbations of the simulator. It then evaluates the **same fixed controllers and models** under them, with no retraining and no per-scenario tuning. They are robustness tests, **not** claims that ScaleRL models AWS, Azure, or GCP precisely.

| Condition | What changes |
|---|---|
| **Nominal** | nothing: the perfectly repeatable reference simulator |
| **Capacity jitter** | seeded variation in *actual* service throughput |
| **Telemetry delay** | the manager sees stale load/queue/latency measurements, but knows its current replicas |
| **Combined** | both at once |

**Capacity jitter** (model `uniform-multiplicative-v1`):
- Each tick's real per-replica capacity is `service_capacity_rps × m`, with `m ~ Uniform(1 − f, 1 + f)`.
- The queue serves with that capacity, and utilization, queue delay, p95 latency, and SLA are computed from the same realization, so it is physics rather than cosmetic noise.
- Price does not jitter.
- Each environment has its own RNG, seeded by `dynamics_seed` and restarted on every reset, with one draw per tick whatever the action. The same seed therefore gives every controller the same multiplier sequence, independent of evaluation order.
- With `f = 0` the multiplier is exactly 1.0 and no RNG is drawn.
- Step `info` reports `capacity_multiplier`, `effective_service_capacity_rps_per_replica`, and `effective_total_service_capacity_rps`.
- Normalizers (demand pressure, queue pressure, max-fleet capacity) stay **nominal**, so metrics are comparable across scenarios.

**Telemetry delay** (`telemetry_delay_ticks = d`):
- **Physical view.** Step `info` always says what actually happened; evaluation metrics (`EpisodeMetrics`) come only from it.
- **Controller view.** Both learned policies (the observation) and rule-based controllers (`decision_info`) see monitoring measurements from the completed tick `d` ticks before the latest one: request rate and recent-demand history, utilization, queue, p95 latency, SLA status, and tick cost. These come from a bounded buffer of completed-tick snapshots (at most `d + traffic_history_ticks`, cleared on reset).
- **What stays current.** Control-plane facts: active, pending, and terminating replicas, pending readiness, episode progress and clock, and the latest requested/applied action. Threshold's cooldown is therefore never late.
- **Before a `d`-old measurement exists:**
  - the observation's measurement features are zero, as at reset;
  - `decision_info` has **no** measurement fields, so Threshold holds (`no_sample`) instead of reading a fake 0% utilization, and Predictive records no sample.
- **Readable example.** Physical demand 100, 200, 300. After tick 2, the history is 300, 200, 100, 0 with `d = 0`, and 200, 100, 0, 0 with `d = 1`. No future values are ever visible.

**`robustness-v1` scenarios** are frozen before held-out evaluation (#46) and never tuned to favor a controller:

| Scenario | capacity_jitter_fraction | telemetry_delay_ticks |
|---|---|---|
| `nominal` | 0.00 | 0 |
| `capacity-jitter` | 0.10 (±10%: 50 RPS/replica becomes 45–55 per tick) | 0 |
| `delayed-telemetry` | 0.00 | 1 (30 s of monitoring lag at the benchmark cadence) |
| `combined-robustness` | 0.10 | 1 |

Optional predeclared jitter levels for a later sensitivity plot are 0.00, 0.05, 0.10, and 0.20; the main comparison uses 0.10.

**How it is used:**
- `scalerl.evaluation.robustness.apply_scenario(config, scenario, dynamics_seed=…)` builds a scenario config without mutating the base.
- `evaluate_robustness` / `evaluate_robustness_tracked` evaluate one fixed controller for one workload, scenario, and seed. They return the shared `EpisodeMetrics`, realized-capacity diagnostics (mean/min/max multiplier; never objectives), and the raw physical infos for per-seed storage.
- #19 will loop workloads × scenarios × dynamics seeds × controllers over this.

**Learned models and delay.** Telemetry delay and the jitter model are part of the model-compatibility contract; the jitter fraction and dynamics seed are not, because they are evaluation conditions.
- A nominally trained DQN/PPO therefore loads under capacity jitter as usual.
- Under delayed telemetry the normal loader refuses it. Evaluating a fixed nominal model with stale telemetry is exactly the robustness question, so it requires the explicit, evaluation-only `load_sb3_controller(..., robustness_evaluation=True)`.
- That path permits only a telemetry-delay difference, still rejects every other mismatch, and records the perturbed field (MLflow tag `scalerl.robustness.perturbed_compatibility`).
- Contracts saved before #65 load as nominal.

**No provider-specific first-request cold-start penalty.** ScaleRL already models delayed capacity availability through replica startup delay. A separate provider-specific first-request latency penalty could double-count startup effects without a more specific runtime model.

**Startup-delay variability is still missing from robustness-v1.** The startup duration itself is deterministic in the current simulator. Issue #81 adds a separate seeded per-replica startup-delay robustness extension without modifying the already-frozen four `robustness-v1` scenarios.

## Multi-seed evaluation (#19)

One training seed or one lucky noise realization is not evidence. #19 evaluates fixed controllers, including **every** training seed of each learned policy, across validation workloads × `robustness-v1` scenarios × seeds. It keeps one raw row per episode and reports distributions. It does not pick a winner or a "best seed". This is **single-seed development result ≠ multi-seed evidence**, and it is still **validation only**: held-out test workloads are rejected before anything runs and stay sealed until #46.

**Three seeds, three meanings:**

| Seed | What it changes | v1 values |
|---|---|---|
| training seed | which learned policy exists (network init, exploration): DQN seed 0 and seed 1 are two different models | 0, 1, 2, 3, 4 |
| evaluation seed | the episode/controller reset; matters for stochastic controllers (Random). Learned policies and rule controllers are deterministic | 0 (Random: 0–4) |
| dynamics seed | the #65 physical-world realization: the capacity-jitter sequence | jitter scenarios: 0–4; others: 0 only |

**Matched dynamics ("same weather").** For a given workload, scenario, and dynamics seed, every controller (Threshold, Predictive, each DQN/PPO seed, …) experiences the identical exogenous capacity-multiplier sequence, independent of evaluation order. Controller actions still change queues and latency.

**No pseudo-replication.** In `nominal` and `delayed-telemetry` there is no capacity jitter, so a deterministic controller produces the same episode for every dynamics seed. Those scenarios therefore run dynamics seed 0 only, instead of five identical copies counted as five samples. Zero spread there is a valid result.

**Canonical controller preparation** (`scripts/multiseed_prepare.sh`, train/validation only):
- **Threshold:** the exact #13 18-point grid (`threshold-sla-first` v1) on the synthetic train/validation workloads.
- **Predictive:** the fixed #63 policy (`forecast-plus-backlog-v1`, 4-tick history, target utilization 0.8); not tuned.
- **DQN / PPO:** the predeclared v1 Optuna studies. Each uses `dqn-search-v1` / `ppo-search-v1`, seeded TPE (seed 42), 20 trials of 200,000 / 204,800 timesteps, trained on `syn-train-bursty`, and selected with the SLA-first rule on `syn-val-steady-high`, `syn-val-ramp-down`, and `syn-val-bursty`. The selected configuration is then retrained with training seeds 0–4, and all five models are kept.
- **Static:** `static-v1` holds 5 replicas (half of the default `max_replicas`), a fixed cost reference predeclared and not tuned.
- **Random:** a sanity reference, never a target.

**Controller manifest** (`controller-manifest.json`, schema `controllers-multiseed-v1`) names every variant (`threshold-v1`, `predictive-v1`, `static-v1`, `random-v1`, `dqn-seed0` … `ppo-seed4`) with its parameters, tuning lineage (study, trial, validation metrics, MLflow run IDs) and, for learned models, the training seed, training run, `runs:/<id>/model` URI, hyperparameter source, and compatibility contract. It selects no seed. #72 will freeze a deployment artifact later, under a rule declared before any held-out or live evaluation.

**Plan and cases.** `EvaluationPlan` (`multiseed-v1`) records the benchmark version, workloads, scenarios and robustness version, dynamics seeds, the resolved variants (with their training and evaluation seeds), simulator config and provenance, reward weights, and summary method; `plan_id` is a hash of all of that. The plan is saved before anything runs. Cases are the Cartesian product in a fixed order, with stable IDs `variant|workload|scenario|dyn<d>|eval<e>`. Each case gets a fresh controller; learned bundles are loaded strictly, or via the #65 robustness-only path under delayed telemetry.

**Statistics (`descriptive-v1`).** Each row reports n, mean, median, sample SD, min, max, Q1, Q3, and IQR (linear-interpolation quartiles). With n = 1, the SD is reported as unavailable rather than 0. Summaries are per workload and scenario, never pooled across them, at two levels:
- **variant:** one configuration or model over its raw runs (dynamics seeds × evaluation seeds);
- **controller:** across a controller's variants, each represented by its variant mean. For DQN/PPO this is the spread across the five training seeds, the primary model variability. A single-configuration rule controller has n = 1 here, and its dynamics spread is in its variant row.

Every row names its raw MLflow runs. There are no p-values or significance claims; a bootstrap CI is not part of `descriptive-v1`.

**Paired deltas** (`paired-deltas.csv`): `controller − threshold-v1` for SLA violation, cost, queue pressure, p95 latency, and churn, computed only for rows sharing a workload, scenario, and dynamics seed. They are descriptive, not a ranking.

**Reproduce the canonical #19 development run:**

```bash
scripts/multiseed_prepare.sh                         # tuning + 5-seed retraining (resumable)
python -m scalerl.evaluation.multiseed manifest      # -> outputs/multiseed-v1/controller-manifest.json
python -m scalerl.evaluation.multiseed run \
    --manifest outputs/multiseed-v1/controller-manifest.json \
    --tracking-uri sqlite:///outputs/mlflow.db --output-dir outputs/multiseed-v1   # add --resume after an interruption
```

The outputs are `evaluation-plan.json`, `controller-manifest.json`, `raw-results.jsonl`/`.csv`, `summary.json`/`.csv`, and `paired-deltas.csv`. They live under `outputs/` and are not committed. Each case is also an MLflow `evaluate` run in the experiment `scalerl-multiseed`. `--resume` skips completed case IDs, recovers cases whose MLflow run finished but whose local row was lost, and never counts a case twice.

## Pre-held-out methodology revisions (#78–#81)

The validation-only #19 run did what it was supposed to do: it exposed weaknesses in the **experimental design** before the held-out test suite was opened. These findings are kept as evidence; they are not erased or replaced.

### #78 — cost-aware selection under an SLA constraint

The original learned-policy selector was lexicographic: minimize SLA violations first, then consider cost. On validation data that can reward a trivial policy that scales to almost the whole fleet and holds there (normalized cost about 0.97), because cost matters only after SLA has already been minimized.

That is a **selection-objective failure mode**, not evidence of a pipeline bug. #78 freezes a v2 rule before any further model selection. The v1 selectors (`dqn-sla-first` v1, `ppo-sla-first` v1, `threshold-sla-first` v1), their Optuna studies and selected trials, and the #19 results are **not modified or rerun**; they remain the historical record.

**`selection-v2-cost-under-sla`** (`scalerl.evaluation.model_selection`, frozen spec [`benchmarks/v1/selection-v2-cost-under-sla.json`](../benchmarks/v1/selection-v2-cost-under-sla.json), spec ID `418876d6c8e9`) asks: *among candidates that meet an acceptable service level, which uses the least infrastructure?*

1. **Reference service level:** the tuned Threshold `threshold-v1` (high 0.6, low 0.2, cooldown 3; `threshold-sla-first` v1, study `threshold-v1`, trial 14), as measured in the #19 nominal validation runs (plan `44f2ddd31094`):

   | Validation workload | Threshold SLA violation rate | #19 MLflow run |
   |---|---|---|
   | `syn-val-steady-high` | 0.275 | `7bebdb2ca0004ac88100ebf254fdd0c2` |
   | `syn-val-ramp-down` | 0.35 | `d7a340898c754d9686acea67ba6306b5` |
   | `syn-val-bursty` | 0.20833333333333334 | `1bd190df336e451e8b2638a9769bda1e` |

2. **Feasibility, per workload:** a candidate is feasible only if `candidate SLA <= Threshold SLA + 1e-12` on **every** workload. The `1e-12` is a floating-point equality epsilon, not an SLA allowance. A good mean cannot hide a failure on one workload.
3. **Objective among feasible candidates:** equal-weight mean over the three workloads of normalized cost; ties broken by mean queue pressure, then churn rate, then SLA violation rate, then a deterministic candidate key (an index such as a trial number or training seed, then the candidate ID). **Reward is never used for selection.**
4. **No feasible candidate:** nothing is selected (`selection_succeeded = false`, `selected_candidate_id = null`) and the constraint is never relaxed. A **diagnostic-only** fallback (lowest mean SLA, then cost, queue, churn, key) is reported with each workload's threshold, candidate SLA, excess, maximum positive excess and mean excess. It is not a selection.
5. **Conditions:** nominal validation only (`robustness-v1` `nominal`, dynamics seed 0, evaluation seed 0). Robustness scenarios remain evaluation evidence and never become a hidden selection objective. Evidence files containing any held-out test row are refused, and there is no option to include them.
6. **Scope:** selection is within one candidate family (DQN candidates against DQN candidates, PPO against PPO). It never produces a global winner across algorithms.

**Two selection levels.** The rule is generic. #79 and #20 apply it to *hyperparameter/configuration* candidates (e.g. Optuna trials) in new, predeclared studies. Applied to #19's evidence, it can only choose among the five **already-trained seed artifacts** of one v1 configuration. That is a diagnostic of the frozen rule on existing evidence, **not** a retroactive re-selection of the v1 hyperparameters (DQN trial 16, PPO trial 7), which were chosen by the v1 rule and stay as they are.

**Diagnostic on the #19 evidence (validation only; no retraining, no new evaluation):**

```bash
python -m scalerl.evaluation.model_selection diagnose \
    --spec benchmarks/v1/selection-v2-cost-under-sla.json \
    --manifest outputs/multiseed-v1/controller-manifest.json \
    --raw-results outputs/multiseed-v1/raw-results.jsonl --output-dir outputs/selection-v2
```

This writes `dqn-selection.json`, `ppo-selection.json` and `candidate-table.csv` under `outputs/` (not committed). Result on the canonical #19 run:

| Family | Feasible | Selected | Mean cost | Mean queue | Mean churn | Mean SLA |
|---|---|---|---|---|---|---|
| DQN | 1/5 (`dqn-seed0`) | `dqn-seed0` | 0.970 | 0.044 | 0.075 | 0.094 |
| PPO | 5/5 (all seeds identical) | `ppo-seed0` (candidate-key tie-break) | 0.970 | 0.044 | 0.075 | 0.094 |

DQN seeds 1–4 are cheaper (mean cost 0.81–0.85) but exceed the Threshold SLA on `syn-val-bursty` by 0.008–0.033, so they are infeasible under the frozen rule. **On this evidence the near-full-fleet policy is still what v2 selects in both families.** v2 cannot create a cheaper feasible policy from a v1 search that never looked for one; that is a finding about the v1 search, and the rule is not adjusted to force a different answer. Whether a cost-under-SLA *search* finds cheaper feasible configurations is for the studies run under #79/#20. None of this is a performance claim, and no held-out data was used.

### #79 — action granularity, not "continuous vs discrete"

**Old contract, `delta-v1`** (every run and model before #79, still `SimulatorConfig()`'s default): `Discrete(3)`, action codes `0/1/2` = scale down / hold / scale up. Their replica-count **effects** are `-1/0/+1`; effects are not codes, so `AutoscalingEnv.step(-1)` is invalid.

**New contract, `desired-replicas-v1`**: `Discrete(max_replicas - min_replicas + 1)`, where code `c` requests `min_replicas + c` committed replicas. At the benchmark defaults, codes `0..9` map to targets `1..10`.

**Why.** Under `delta-v1`, a controller that needs six more replicas needs six control decisions, which makes burst recovery artificially sequential. Real horizontal autoscalers set an integer **desired replica count** directly. Kubernetes HPA computes `desiredReplicas = ceil(currentReplicas × currentMetricValue / desiredMetricValue)` and updates the scale target: <https://kubernetes.io/docs/concepts/workloads/autoscaling/horizontal-pod-autoscale/>.

**How the target is actuated.** The environment compares the target with committed capacity (`active + pending`) and starts or cancels `|target - committed|` replicas in one step, through the unchanged lifecycle:
- New replicas start pending and still wait out the 60 s startup delay. For example, `2 → 8` gives `+6` pending, not 8 active.
- A reduction cancels the newest pending replicas first, then terminates active ones.
- Bounds hold.

`info` records `requested_action` (the code), `requested_replica_target`, and `applied_replica_change`, which can now be `+6` or `-4`. The code-to-target encoding lives in one place, `scalerl.environment.actions.ActionContract`; see [ARCHITECTURE.md](ARCHITECTURE.md#action-space-versioned-action-contracts-79).

**DQN vs PPO.** Horizontal replica counts are discrete, so DQN is fully valid under both contracts, and PPO uses its categorical policy. It is **not** true that "PPO is needed because cloud autoscaling actions are continuous"; continuous or hybrid actions would be a different problem, such as vertical CPU/RAM sizing.

**Reward unchanged.** The churn term still penalizes a scaling *event* (`applied_replica_change != 0`), so `+6` in one tick costs the same as `+1`. #79 keeps it that way and reports magnitude separately (`action.*` diagnostics; see [REWARD_DESIGN.md](REWARD_DESIGN.md)). Fewer scaling ticks do not mean less scaling. #20 owns any reward change.

**Historical reproducibility.** Every pre-#79 model, bundle and run is `delta-v1`:
- Compatibility contracts without `action_semantics_version` load as `delta-v1`.
- The default config serializes exactly as before, so v1 study identities and the #19 plan ID are unchanged.
- `delta-v1` episodes match the pre-#79 code. They were bit-identical when compared on one machine, and the committed fingerprint tests (computed from commit `c69c672`) compare them to 10 significant digits, so they hold across platforms.
- A model never loads under the other contract, even at the same action count. The #65 robustness loader never relaxes this.

**Controllers.**
- **Threshold** keeps its law and its canonical parameters (0.6/0.2, cooldown 3). Under `desired-replicas-v1` it encodes the same ±1/hold decision as the target `committed±1` or `committed`. There is no HPA-style redesign.
- **Predictive** (`predictive-v1`, `linear-trend`, `forecast-plus-backlog-v1`, all unchanged) encodes its already-computed `desired_replicas` directly, or `committed` when it holds (e.g. `backlog_hold`). Under `delta-v1` it steps one replica toward it, as before.
- **Static** requests its fixed target.
- **Random** is a sanity reference only; random targets and random ±1 steps are not comparable.
- **Config-driven builders** always act under the config's own contract: `PredictiveController.from_config` (which rejects a mismatching explicit contract), the Scenario Lab managers, threshold tuning, and #19's `make_controller`. A `desired-replicas-v1` config can therefore never receive `delta-v1` codes.

#### Experiment design (`action-semantics-experiment-v1`, frozen before training)

The only variable is the action contract. The spec [`benchmarks/v1/action-semantics-experiment-v1.json`](../benchmarks/v1/action-semantics-experiment-v1.json) (ID `899dfbb64217`) and the candidate set [`benchmarks/v1/action-semantics-candidates-v1.json`](../benchmarks/v1/action-semantics-candidates-v1.json) (`matched-action-candidates-v1`, ID `cbe3c1c0719b`) were committed before any of its models were trained.

- **Workloads:** train on `syn-train-bursty`; validate on `syn-val-steady-high`, `syn-val-ramp-down`, `syn-val-bursty`. **No test workload.**
- **Held fixed:** simulator config (only `action.semantics` differs), reward weights (unchanged), budgets (DQN 200,000 / PPO 204,800 timesteps).
- **Matched candidates:** 20 DQN and 20 PPO configurations from `dqn-search-v1` / `ppo-search-v1`, drawn by a seeded Optuna `RandomSampler` (seed 42) with **nothing trained or evaluated**. Both contracts train exactly these configurations, so they are paired by candidate ID. Two independent adaptive TPE studies would have confounded the contract with search luck.
- **Screening:** every candidate is trained with training seed 0 under each contract and validated on all three workloads (80 models).
- **Selection:** the frozen #78 rule `selection-v2-cost-under-sla` (spec `418876d6c8e9`), now at the *configuration* level, within each of the four families `{DQN, PPO} × {delta-v1, desired-replicas-v1}`. It needs SLA ≤ Threshold on every workload and then the lowest cost. There is no cross-algorithm winner, reward is never used, and nothing is selected when nothing is feasible.
- **Retraining:** every selected configuration is retrained with seeds 0–4, and all five are kept.
- **Evaluation:** nominal only (dynamics seed 0, evaluation seed 0). Robustness scenarios take no part.
- **Decision principle (predeclared):** adopt the more faithful `desired-replicas-v1` if the action/lifecycle invariants pass, DQN and PPO train and infer against it, and the experiment completes without a systematic failure attributable to the encoding. Otherwise leave the contract unresolved. Validation metrics characterize the consequence of the change; they do **not** choose the contract.

```bash
python -m scalerl.evaluation.action_semantics check        # prints the frozen plan
python -m scalerl.evaluation.action_semantics screen --family dqn-delta-v1   # and the other 3
python -m scalerl.evaluation.action_semantics select
python -m scalerl.evaluation.action_semantics retrain --family dqn-delta-v1  # each selected family
python -m scalerl.evaluation.action_semantics evaluate
python -m scalerl.evaluation.action_semantics decide --freeze
```

Every phase is resumable: completed results are files, and evaluation rows are recovered from finished MLflow runs (experiment `scalerl-action-semantics`) instead of being rerun. Evaluation, the decision, and the frozen artifact accept only the exact predeclared retraining matrix: for every selected family, seeds 0–4 of its selected candidate from this experiment. Missing, extra, or mismatched model files are refused. Outputs are written to `outputs/action-semantics-v1/` and are not committed.

#### Results (validation only)

**Configuration selection (#78 rule, 20 matched candidates per family):**

| Family | Feasible | Selected | Mean cost / queue / churn / SLA |
|---|---|---|---|
| DQN `delta-v1` | 4/20 | `dqn-c11` | 0.749 / 0.078 / 0.125 / 0.172 |
| DQN `desired-replicas-v1` | **0/20** | **none** (diagnostic only: `dqn-c01`) | — |
| PPO `delta-v1` | 8/20 | `ppo-c11` | 0.845 / 0.047 / 0.064 / 0.114 |
| PPO `desired-replicas-v1` | 14/20 | `ppo-c08` | 0.692 / 0.039 / 0.011 / 0.139 |

**DQN under `desired-replicas-v1` has no feasible configuration.** Its policies train, act validly (`Discrete(10)`), and do use multi-replica jumps (up to 7–9 replicas per tick). However, with this budget and search space they either thrash or under-provision. 17/20 have churn ≥ 0.18 (up to 0.49) and move up to 224 replicas per episode, where `delta-v1` candidates move at most 65. Five violate the SLA on every tick of `syn-val-steady-high` (rate 1.0). Every candidate violates the Threshold SLA on at least one workload. The closest, `dqn-c01`, fails only on `syn-val-bursty` (0.250 vs 0.208, +0.042), with a mean SLA excess of −0.114. Per #78, nothing is selected, the fallback is not a model of record, and no DQN `desired-replicas-v1` seeds exist. This is a learning outcome, not an encoding failure: the invariants hold, PPO learns well under the same encoding, and every learned evaluation reproduces its training-time validation exactly. It means a DQN configuration for the final contract still has to be found under a predeclared search (#20).

Matched PPO candidates tell the opposite story. All 8 candidates feasible under `delta-v1` are also feasible under `desired-replicas-v1`, and 6 more become feasible. The selected desired configuration is cheaper (0.692 vs 0.845) at a comparable SLA; its main difference is much lower churn.

**Nominal evaluation, five retrained seeds per selected family** (mean ± SD over training seeds; rules are single deterministic runs):

| Controller | Contract | Workload | SLA | Cost | Queue pressure | Churn | Replicas moved | Max per tick |
|---|---|---|---|---|---|---|---|---|
| Threshold | both | steady-high / ramp-down / bursty | 0.275 / 0.350 / 0.208 | 0.872 / 0.839 / 0.823 | 0.186 / 0.269 / 0.106 | 0.075 / 0.108 / 0.117 | 9 / 13 / 14 | 1 |
| Predictive | `delta-v1` | steady-high / ramp-down / bursty | 0.100 / 0.150 / 0.408 | 0.698 / 0.536 / 0.507 | 0.052 / 0.092 / 0.146 | 0.100 / 0.142 / 0.800 | 12 / 17 / 96 | 1 |
| Predictive | `desired-replicas-v1` | steady-high / ramp-down / bursty | 0.042 / 0.067 / 0.600 | 0.708 / 0.550 / 0.600 | 0.015 / 0.030 / 0.162 | 0.017 / 0.075 / 0.592 | 12 / 17 / 249 | 9 |
| DQN (`dqn-c11`) | `delta-v1` | bursty | 0.237 ± 0.045 | 0.661 ± 0.056 | 0.061 ± 0.018 | 0.383 ± 0.110 | 46 ± 13 | 1 |
| PPO (`ppo-c11`) | `delta-v1` | bursty | 0.132 ± 0.078 | 0.782 ± 0.134 | 0.028 ± 0.022 | 0.058 ± 0.012 | 7 ± 1 | 1 |
| PPO (`ppo-c08`) | `desired-replicas-v1` | bursty | 0.092 ± 0.086 | 0.792 ± 0.111 | 0.015 ± 0.017 | 0.020 ± 0.026 | 10 ± 5 | 7 ± 1 |

Per-workload, per-seed rows, descriptive summaries, desired − delta paired deltas (same controller, seed, and workload) and matched-candidate pairs are in `raw-results.*`, `summary.*`, `paired-deltas.csv` and `candidate-pairs.csv`. No p-values or significance claims are made.

- **Threshold** is identical under both contracts: its law is ±1 by design, so this is a contract-equivalence control that the encoding passes.
- **Predictive** improves on steady-high and ramp-down under `desired-replicas-v1`, where it reaches its computed capacity immediately. It is worse on `syn-val-bursty` (SLA 0.600 vs 0.408, 249 vs 96 replicas moved). There, its linear-trend forecast of a noisy burst is actuated at full size each tick instead of being rate-limited by ±1 steps, and it oscillates. The `delta-v1` actuator was implicitly smoothing a weak forecaster; that is a finding for #80, not something to tune away here.
- **PPO seeds vary.** Under `delta-v1`, seed 3 learns a cheap policy that fails ramp-down (SLA 0.742) and seed 4 runs near full fleet. Under `desired-replicas-v1`, the five seeds span cost 0.68–0.90 with SLA ≤ 0.19 on every workload.
- **DQN `delta-v1` seeds 1–4** miss the bursty threshold (0.24–0.27) even though the selected configuration's seed 0 passed. Seed variability is real, and the retrained seeds are reported, not re-selected.

#### Decision: `desired-replicas-v1` is the final action contract

All three predeclared criteria passed:
- the action/lifecycle invariants hold;
- DQN and PPO train and infer against `desired-replicas-v1`, 20/20 candidates each, with finite metrics and strict compatibility;
- the experiment completed: 80 screening models, 15 retrained models, 57/57 evaluation rows, and 45/45 learned evaluations identical to their training-time validation.

The decision is frozen in [`benchmarks/v1/action-contract-v2.json`](../benchmarks/v1/action-contract-v2.json) (ID `0eb5562b01e6`), with the selection outcome, all retrained model run IDs, and a validation summary. It was not chosen for better metrics: the contract was preferred before the run, and its consequences are mixed (better PPO, worse DQN selection, and worse Predictive on bursty).

Implications:
- #20, #72 and #46 use `desired-replicas-v1` explicitly, with models trained under it. `SimulatorConfig()` stays `delta-v1` for historical reproducibility.
- There is currently **no selected DQN configuration** under the final contract; #20's predeclared study must find one or report that it cannot.
- Provenance note: all 320 screening runs and the 57 evaluation runs ran from a clean commit (`fc9cd73`). Ten of the 15 retraining models ran while three documentation files were uncommitted, so they are tagged `git_dirty=true` with the same code SHA. Two interrupted retraining attempts are marked `KILLED` with a `scalerl.superseded` tag and are not models of record.

### #80 — cloud-style proactive predictive baseline

`predictive-v1` stays exactly as it was: forecast method `linear-trend` (4-sample OLS trend), capacity policy `forecast-plus-backlog-v1`, startup-aware and queue-aware. Its fingerprint tests still pass after #80 extracted its trend and sizing formulas as shared pure helpers. #80 adds a **separate** controller, `predictive-seasonal-v1` (`scalerl.controllers.proactive_predictive`). It is a *cloud-style proactive predictive baseline*: it applies the same core idea as cloud predictive scaling (recurring history plus capacity launched ahead of startup delay). It is **not** an emulation of AWS predictive scaling or any other provider's service.

- **Historical profile (`historical-demand-profile-v1`):** `profile[tick] = median(train_1[tick], …, train_n[tick])` over **TRAIN** traces with the same control interval and aligned ticks. Validation and test workloads are refused as history. The profile is immutable and content-hashed.
- **Forecast method `historical-profile-plus-linear-v1`:**
  - Horizon: `h = 1 + startup_ticks(startup_delay, control_interval)`, the `predictive-v1` arithmetic (3 ticks / 90 s by default). The linear and profile forecasts target the same tick `t = s + h`.
  - Level adaptation: `level_factor = median(observed_i / profile[tick_i])` over the latest ≤4 completed samples. Profile values below 1e-3 rps are skipped, and the factor is 1.0 when no ratio is valid.
  - Proactive forecast: `max(linear, profile[t] × level_factor)`, or `linear` when no profile value exists. There is no blend weight.
- **Capacity policy `proactive-scaleout-conservative-scalein-v1`:**
  - Sizing: `desired = ceil((proactive + queued / control_interval) / (50 rps × 0.8))`, clamped. This keeps the #63 backlog recovery.
  - Scale-out: when `desired > active + pending`, the full target is requested at once under `desired-replicas-v1`, and the new replicas still start pending.
  - Scale-in: never while requests are queued. Otherwise the target is `max(max(observed_desired, desired), committed − 1)`, so capacity never drops below observed *or* forecast need and falls by at most one replica per decision.
- **Everything is predeclared and nothing was tuned:** 4 samples, 0.8 utilization, median, `max`, one-replica scale-in. There was no Optuna run and no grid.

**Scientific limitation.** Benchmark v1's synthetic validation workloads (steady-high, ramp-down, bursty) contain **no recurring history**. `predictive-seasonal-v1` therefore runs there **without** a profile, which isolates its actuation policy: direct scale-out plus conservative scale-in, the same forecast as `predictive-v1`. No seasonal claim is made from them. Recurring history is evaluated on Azure, whose benchmark windows are the same hour (12:00–13:00) of different days. `syn-test-seasonal-shifted` is held out and was not used. Random bursts are not inherently predictable.

**Experiment `predictive-baseline-experiment-v1`** ([spec](../benchmarks/v1/predictive-baseline-experiment-v1.json), ID `096b04ef953a`, committed before any validation run):
- Threshold, `predictive-v1` and `predictive-seasonal-v1`, all under an **explicit** `desired-replicas-v1` (`action-contract-v2`); `SimulatorConfig()` still defaults to `delta-v1`, and the run refuses it.
- Nominal dynamics, dynamics seed 0, evaluation seed 0, unchanged reward.
- Workloads: the three synthetic validation workloads, plus `azure-val-734400` with the TRAIN-only profile `azure-historical-profile-v1` (sources `azure-train-129600/302400/475200`, profile ID `d050d3b8ca0f`).
- Command: `python -m scalerl.evaluation.predictive_baseline check | run | freeze`. Results go to the MLflow experiment `scalerl-predictive-baseline` and to `outputs/predictive-baseline-v1/`.
- Every case is keyed by an `input_id`: a hash of the exact trace plus, for the profiled case, the profile ID. A resume never reuses rows or MLflow runs computed from different inputs, such as another Azure CSV. Each run also logs its forecast records as an artifact, so interrupted local record files are restored instead of being published incomplete.

**Results (validation only; one deterministic run per cell):**

| Workload | Controller | SLA | Norm. cost | Queue pressure | Churn | Replicas moved | Max per tick |
|---|---|---|---|---|---|---|---|
| steady-high | Threshold | 0.275 | 0.872 | 0.186 | 0.075 | 9 | 1 |
| steady-high | predictive-v1 | 0.042 | 0.708 | 0.015 | 0.017 | 12 | 9 |
| steady-high | predictive-seasonal-v1 | 0.042 | 0.710 | 0.015 | 0.033 | 12 | 9 |
| ramp-down | Threshold | 0.350 | 0.839 | 0.269 | 0.108 | 13 | 1 |
| ramp-down | predictive-v1 | 0.067 | 0.550 | 0.030 | 0.075 | 17 | 9 |
| ramp-down | predictive-seasonal-v1 | 0.067 | 0.568 | 0.030 | 0.075 | 17 | 9 |
| bursty | Threshold | 0.208 | 0.823 | 0.106 | 0.117 | 14 | 1 |
| bursty | predictive-v1 | 0.600 | 0.600 | 0.162 | 0.592 | 249 | 9 |
| bursty | predictive-seasonal-v1 | **0.267** | 0.609 | **0.097** | 0.608 | **115** | 7 |
| azure-val | all three | 0.000 | 0.100 | 0.000 | 0.000 | 0 | 0 |

- **`syn-val-bursty`: conservative scale-in fixes most of the oscillation #79 exposed, without any history.** The forecasts are identical, but asymmetric actuation cuts replica movement from 249 to 115 and SLA from 0.600 to 0.267, and queue pressure falls from 0.162 to 0.097.
  - Cost is almost the same (0.609 vs 0.600).
  - The number of scaling ticks is about the same (churn 0.608 vs 0.592): the new policy makes about as many, smaller moves.
  - Threshold still has the lower bursty SLA (0.208), at 0.823 cost. Neither dominates, and no winner is declared.
- **steady-high and ramp-down:** SLA and queue are identical. One-at-a-time scale-in costs slightly more (0.710 vs 0.708; 0.568 vs 0.550) and adds two scaling events on steady-high.
- **Azure (recurring history; the profile was used in 117 of 119 issued forecasts, 98%):** every controller stays at `min_replicas` (1). As predeclared, one replica serves 50 rps and this window peaks at about 2.5 rps, so the system metrics cannot discriminate. The forecast diagnostics show **no accuracy gain from the historical profile here** (next table). The level factor, estimated from four tiny, quantized per-tick counts, ranged from 0.09 to 4.54, and the `max` rule biases forecasts upward by design.

| Workload | Forecaster | n | MAE (rps) | RMSE | Bias (forecast − actual) |
|---|---|---|---|---|---|
| steady-high | both (identical linear) | 117 | 0.000 | 0.000 | 0.000 |
| ramp-down | both (identical linear) | 117 | 0.062 | 0.676 | +0.062 |
| bursty | both (identical linear) | 117 | 152.2 | 236.7 | +19.8 |
| azure-val | predictive-v1 (linear) | 117 | 0.790 | 1.081 | +0.167 |
| azure-val | predictive-seasonal-v1 (max rule) | 117 | 1.240 | 1.965 | +0.916 |
| azure-val | its profile component alone | 117 | 1.137 | — | — |

Forecast accuracy is reported separately from control quality: on bursty, identical forecasts produce very different control outcomes. The frozen baseline artifact [`benchmarks/v1/predictive-baseline-v1.json`](../benchmarks/v1/predictive-baseline-v1.json) (ID `0d315559c680`) records the identity, parameters, profile definition, per-workload metrics and MLflow run IDs that #72 and #46 must include. It declares no overall winner.

**Not included:**
- #79's PPO models (the primary question here is predictive-baseline quality);
- any DQN: no desired-replicas-v1 DQN was selected in #79, and the diagnostic fallback `dqn-c01` is not a model of record;
- an oracle (optional; not built);
- any RL retraining.

**Follow-ups:**
- Azure capacity calibration on train/validation, so that recurring-history evidence can also move system metrics;
- whether level adaptation needs more than 4 samples for low-rate traces. That would be a new predeclared version, not a post-hoc change to v1.

### #81 — stochastic startup delay

`robustness-v1` stays **frozen and unchanged**: nominal, capacity jitter, delayed telemetry, and combined robustness, all with fixed startup. #81 adds a **separate** versioned extension, `startup-robustness-v1`. Actual launch time does vary in practice, with factors such as instance size and startup scripts: <https://docs.aws.amazon.com/autoscaling/ec2/APIReference/API_PredictiveScalingConfiguration.html>.

**Startup model `tri-point-multiplicative-v1`.** A bounded, seeded **robustness stress model**. It is **not** a provider-calibrated startup distribution.
- Each newly requested replica independently draws a multiplier of 0.5 (p = 0.25), 1.0 (p = 0.50) or 1.5 (p = 0.25); the expected multiplier is exactly 1.0.
- Its realized delay is `startup_delay_seconds × multiplier`. The seconds are not rounded; a replica still activates only through the per-tick lifecycle.
- At 60 s nominal and 30 s ticks, that means 30 / 60 / 90 s, ready after 1 / 2 / 3 ticks. A continuous ±25% model would mostly collapse into 2 vs 3 ticks at this cadence; three explicit points make early, nominal and late readiness visible.
- The default `fixed-v1` keeps every delay exactly nominal and draws nothing, so every earlier run and model is unchanged.

**Configuration and RNGs.**
- `DynamicsConfig.startup_delay_model` (`fixed-v1` | `tri-point-multiplicative-v1`) and `startup_delay_seed` are serialized only when not at their defaults. Pre-#81 config hashes, study identities and plan IDs therefore do not move.
- The startup RNG is a dedicated per-environment stream, **domain-separated** from the capacity-jitter stream. Capacity uses `default_rng(dynamics_seed)`, unchanged since #65. Startup uses `default_rng(SeedSequence(startup_delay_seed, spawn_key=(STARTUP_RNG_DOMAIN,)))`, where the constant is `0x73747570`. Even matched replicates with `dynamics_seed == startup_delay_seed` therefore never share a random stream. Both streams restart on reset. Each new replica takes one draw, in request order; no global random state is used.
- A `desired-replicas-v1` decision that adds five replicas draws five independent delays, and they can activate on different ticks.
- Cancellation still removes the newest pending replicas first, and a cancelled replica never activates.

**Information contract.**
- Controllers know the nominal startup delay and see pending replicas by their **nominal** readiness: their age against the configured delay.
- The observation shape is unchanged. A replica past its nominal readiness that is still physically pending counts as "due now" (the first bucket).
- The sampled realization is physical evaluation truth only. It appears in step `info` (`startup_realized_delays_seconds`, `startup_multipliers`, `physical_pending_by_ticks`, listed in `PHYSICAL_ONLY_KEYS`) and never in telemetry, `decision_info`, or the observation. Tests pin this.

**Compatibility.**
- `EnvironmentCompatibility.startup_delay_model` is compatibility-defining. Old contracts load as `fixed-v1`, and the seed is evaluation metadata only.
- Ordinary loading rejects a startup-model mismatch. The robustness-only path may perturb it, alongside telemetry delay; action semantics are never perturbable.

**Fairness.**
- Same workload, scenario, startup seed and request sequence give an identical startup realization, whatever the evaluation order.
- Controllers that request different replicas at different times consume the stream differently, so their realized delay lists legitimately differ. The guarantee is not "identical delay lists across controllers".

**Experiment `startup-robustness-experiment-v1`** ([spec](../benchmarks/v1/startup-robustness-v1.json), ID `60f1f14972a7`, committed before the run of record):
- **Scenarios:**
  - `startup-delay-jitter`: no capacity jitter, no telemetry delay, tri-point startup; startup seeds 0–4.
  - `combined-startup-robustness`: ±10% capacity jitter, 1-tick telemetry delay, tri-point startup; matched replicate indices `dynamics_seed = startup_delay_seed = s` for s in 0–4, with the two raw streams domain-separated. A startup seed gives the same startup realization in both scenarios for the same request sequence.
  - Reference: the frozen `robustness-v1` nominal scenario.
- **Setup:** validation workloads only, all under an explicit `desired-replicas-v1` (`action-contract-v2`); evaluation seed 0; unchanged reward.
- **Controllers:** Threshold, `predictive-v1`, `predictive-seasonal-v1`, and the frozen #79 PPO `ppo-c08`, training seeds 0–4. PPO was loaded from its MLflow run with **no retraining**: strictly for nominal, and through the robustness path under the startup scenarios.
- **No DQN:** no canonical desired-replicas-v1 DQN exists, and the #79 diagnostic fallback is not used.
- **Runs:** 264 runs of record in the MLflow experiment `scalerl-startup-robustness`, all at clean commit `547a6e2`. The 24 nominal rows reproduce the #79/#80 results exactly.
- **Superseded run:** a first run (spec `bec43981bfc8`, commit `eeb3ea3`) seeded both streams with the same integer, so capacity and startup jitter started from the same random sequence. Its 264 runs are tagged `scalerl.superseded`, and everything below is from the corrected run.
- **Command:** `python -m scalerl.evaluation.startup_robustness check | run | freeze`.

**Results (validation only; mean over 5 startup seeds, with min–max; nominal is one run; PPO is the mean over its 5 training seeds, each averaged over startup seeds):**

| Workload | Controller | SLA: nominal / startup-jitter / combined | Norm. cost: nominal / startup / combined | Replicas moved: nominal / startup / combined |
|---|---|---|---|---|
| steady-high | Threshold | 0.275 / 0.278 / 0.292 | 0.872 / 0.872 / 0.865 | 9 / 9 / 9 |
| steady-high | predictive-v1 | 0.042 / 0.038 / 0.053 | 0.708 / 0.707 / 0.707 | 12 / 12 / 12 |
| steady-high | predictive-seasonal-v1 | 0.042 / 0.038 / 0.053 | 0.710 / 0.709 / 0.709 | 12 / 12 / 12 |
| steady-high | PPO (5 seeds) | 0.052 / 0.054 / 0.056 | 0.795 / 0.795 / 0.795 | 7.8 / 8.1 / 8.2 |
| ramp-down | Threshold | 0.350 / 0.350 / 0.358 | 0.839 / 0.839 / 0.837 | 13 / 13 / 13 |
| ramp-down | predictive-v1 | 0.067 / 0.063 / 0.087 | 0.550 / 0.550 / 0.552 | 17 / 17 / 17 |
| ramp-down | predictive-seasonal-v1 | 0.067 / 0.063 / 0.087 | 0.568 / 0.567 / 0.569 | 17 / 17 / 17 |
| ramp-down | PPO (5 seeds) | 0.115 / 0.113 / 0.118 | 0.807 / 0.806 / 0.802 | 9.6 / 10.2 / 10.5 |
| bursty | Threshold | 0.208 / 0.213 (0.21–0.22) / 0.243 (0.22–0.26) | 0.823 / 0.823 / 0.811 | 14 / 14 / 14.6 |
| bursty | predictive-v1 | 0.600 / 0.573 (0.56–0.58) / 0.517 (0.49–0.56) | 0.600 / 0.583 / 0.570 | 249 / 234 / 179 |
| bursty | predictive-seasonal-v1 | 0.267 / 0.277 (0.28–0.28) / 0.317 (0.31–0.33) | 0.609 / 0.610 / 0.626 | 115 / 116 / 112 |
| bursty | PPO (5 seeds) | 0.092 / 0.090 / 0.098 | 0.792 / 0.792 / 0.793 | 9.6 / 9.6 / 10.3 |

- **The trade-offs are stable.** Startup-delay jitter alone moves SLA by at most 0.011 per controller and workload, except `predictive-v1` on bursty (−0.027, an improvement). No controller's relative position changes on any workload. The combined scenario (with jitter and stale telemetry) costs more, mostly through the #65 components.
- **predictive-seasonal-v1** degrades modestly on bursty (0.267 → 0.277 → 0.317) and keeps its large advantage over `predictive-v1`.
- **predictive-v1** is slightly *better* on bursty under the perturbations (0.600 → 0.573 → 0.517). Late replicas and stale telemetry damp its oscillation, and it moves fewer replicas (249 → 179). That is not a reason to prefer noise; its oscillation is the defect.
- **PPO seeds** are almost unaffected: their fleets are mostly provisioned ahead of demand, so they request few replicas (about 7–9 per episode on average).
- **Startup diagnostics:** realized delays take only the values 30, 60 and 90 s. Per-episode mean delays range from 50 to 68 s where few replicas are requested (9–11 per episode) and from 56 to 65 s on bursty for the predictive controllers (about 60–120 requests); PPO's range is 57.8–61.0 s. The maximum readiness spread within one multi-replica request is 60 s (a 30 s and a 90 s replica in the same request).

The freeze artifact [`benchmarks/v1/startup-robustness-freeze-v1.json`](../benchmarks/v1/startup-robustness-freeze-v1.json) (ID `25d65bbba36f`) records the model, scenarios, seeds, controllers, compact evidence and all 264 run IDs for #72/#46. It declares no winner, uses no held-out data, and leaves the reward unchanged.

### Freeze order

No held-out test workload may be inspected while resolving these issues.

The pre-held-out sequence is:

```text
#19 multi-seed validation evidence
  ↓
#78 constrained selection rule
  ↓
#79 final action semantics (desired-replicas-v1, frozen)
  ├─→ #80 stronger predictive baseline
  └─→ #81 startup-delay robustness extension
          ↓
#20 reward ablation on the final action contract
          ↓
#72 freeze controller artifacts + sim-to-real protocol
          ↓
#46 held-out evaluation
```

## Reward ablation (#20)

**Question.** How sensitive are learned-controller behavior and system trade-offs to the reward specification under the final #79 contract `desired-replicas-v1`? Raw system metrics are primary; episode return is secondary and never selects anything.

**Frozen design** (`reward-ablation-v1`, [spec](../benchmarks/v1/reward-ablation-v1.json), ID `f5aff8f1e8c2`, committed before any run of record):

- **Eight predeclared variants** of the *unchanged* penalty terms. Normalization is not touched; only component inclusion and weights vary.

  | Variant | latency | cost | SLA | queue | churn |
  |---|---|---|---|---|---|
  | `latency-cost-v1` | 1.0 | 1.0 | 0 | 0 | 0 |
  | `latency-cost-sla-v1` | 1.0 | 1.0 | 1.0 | 0 | 0 |
  | `latency-cost-sla-queue-v1` | 1.0 | 1.0 | 1.0 | 1.0 | 0 |
  | `full-default-v1` (current reward) | 1.0 | 1.0 | 1.0 | 1.0 | 0.1 |
  | `full-cost-low-v1` | 1.0 | 0.5 | 1.0 | 1.0 | 0.1 |
  | `full-cost-high-v1` | 1.0 | 2.0 | 1.0 | 1.0 | 0.1 |
  | `full-sla-low-v1` | 1.0 | 1.0 | 0.5 | 1.0 | 0.1 |
  | `full-sla-high-v1` | 1.0 | 1.0 | 2.0 | 1.0 | 0.1 |

- **Setup:** train on `syn-train-bursty`; validate (nominal, evaluation seed 0) on the three synthetic validation workloads; **no test workload**; every run explicitly requests `desired-replicas-v1`.
- **DQN (per reward):** the exact 20 #79 candidates (`matched-action-candidates-v1`, `cbe3c1c0719b`), trained with seed 0 for 200,000 timesteps.
  - The frozen #78 rule (`418876d6c8e9`) selects within the reward condition.
  - A selected candidate is retrained on seeds 0–4.
  - If nothing is selected, nothing is retrained: the diagnostic fallback is recorded only.
- **PPO (per reward):** the fixed #79 configuration `ppo-c08` (same hyperparameters for every reward, no search), trained fresh on seeds 0–4 for 204,800 timesteps.
- **Reward decision (predeclared):**
  - Compute equal-seed means per algorithm × reward × workload.
  - A reward is *joint-feasible* when PPO and the retrained #78-selected DQN both meet the Threshold SLA limit on every workload.
  - Among joint-feasible rewards, minimize the 50/50 equal-algorithm mean of equal-workload normalized cost; then queue pressure, churn, SLA, reward ID.
  - If none is joint-feasible, `full-default-v1` is frozen with the negative DQN result.
- **Command:** `python -m scalerl.evaluation.reward_ablation check | run --reward <id> | decide [--freeze]`.
- **Runs:** results in the MLflow experiment `scalerl-reward-ablation`; outputs in `outputs/reward-ablation-v1/`.

**Runs of record.**
- 860 runs, all `FINISHED`, all at clean commit `5d6651b` with `git_dirty=false`: 215 trainings (160 DQN screening, 15 DQN retraining, 40 PPO) plus 645 validation runs.
- A first launch at `ff50311` was aborted by a temp-file race between parallel processes (a code fix). Its 100 runs are tagged `scalerl.superseded`; every reward was rerun from `5d6651b`.
- Consistency check: `full-default-v1` reproduces #79's DQN screening and PPO models exactly (75/75 per-workload values identical).

**DQN feasibility per reward (#78, 20 candidates each):**

| Reward | Feasible | Selected | Retrained seeds 0–4: SLA (steady-high / ramp-down / bursty) | Reward-level pass |
|---|---|---|---|---|
| `latency-cost-v1` | 0/20 | none (diagnostic `dqn-c11`) | — | no |
| `latency-cost-sla-v1` | 0/20 | none (diagnostic `dqn-c08`) | — | no |
| `latency-cost-sla-queue-v1` | 1/20 | `dqn-c18` | 0.690 / 0.523 / 0.428 | **no** |
| `full-default-v1` | 0/20 | none (diagnostic `dqn-c01`) | — | no |
| `full-cost-low-v1` | 7/20 | `dqn-c14` | 0.072 / 0.167 / **0.2083** (= limit) | **yes** |
| `full-cost-high-v1` | 0/20 | none (diagnostic `dqn-c11`) | — | no |
| `full-sla-low-v1` | 0/20 | none (diagnostic `dqn-c19`) | — | no |
| `full-sla-high-v1` | 2/20 | `dqn-c16` | 0.418 / 0.457 / 0.255 | **no** |

**PPO `ppo-c08` per reward** (equal-seed means, averaged over the three workloads; pass = SLA within the limit on every workload):

| Reward | Norm. cost | SLA | Queue pressure | Churn | Pass |
|---|---|---|---|---|---|
| `latency-cost-v1` | 0.553 | 0.368 | 0.205 | 0.076 | **no** |
| `latency-cost-sla-v1` | 0.796 | 0.062 | 0.016 | 0.022 | yes |
| `latency-cost-sla-queue-v1` | 0.776 | 0.175 | 0.131 | 0.014 | yes |
| `full-default-v1` | 0.798 | 0.086 | 0.026 | 0.018 | yes |
| `full-cost-low-v1` | 0.903 | 0.033 | 0.011 | 0.014 | yes |
| `full-cost-high-v1` | 0.611 | 0.288 | 0.204 | 0.038 | **no** |
| `full-sla-low-v1` | 0.789 | 0.067 | 0.018 | 0.024 | yes |
| `full-sla-high-v1` | 0.844 | 0.050 | 0.014 | 0.017 | yes |

**Decision: `full-cost-low-v1`** (latency 1.0, cost 0.5, SLA 1.0, queue 1.0, churn 0.1), frozen in [`benchmarks/v1/reward-contract-v1.json`](../benchmarks/v1/reward-contract-v1.json) (ID `37158c261364`).
- It is the **only joint-feasible reward**. Its equal-algorithm values are: normalized cost (0.663 DQN + 0.903 PPO) / 2 = **0.783**, queue pressure 0.025, churn 0.113, SLA 0.091.
- The final reward therefore **differs from the previous default** (cost weight 1.0 → 0.5).
- `RewardWeights()` keeps its historical defaults, so every earlier run stays reproducible. Downstream work (#72/#46) must request the contract's weights explicitly.
- The decision is recomputable from the equal-seed means recorded in the artifact (tested).

**Findings.**
- **The DQN pass is knife-edge.** The retrained `dqn-c14` has an equal-seed mean SLA on `syn-val-bursty` of exactly the Threshold limit (0.2083 = 125/600). Seed 3 reaches 0.317, and SLA on ramp-down ranges 0.05–0.40 across seeds. The predeclared `≤` rule passes it; it is not a robust margin.
- **Screening on seed 0 does not transfer reliably.** For `full-sla-high-v1` and `latency-cost-sla-queue-v1`, the selected DQN candidates collapse on some retraining seeds to SLA 1.0 on at least one workload, whose seed means are far outside the limit (e.g. `dqn-c16`: SLA median 0.04 but maximum 1.0 on steady-high).
- **Cost weight is the dominant lever.**
  - Halving it (`full-cost-low-v1`) makes DQN feasible (7/20 candidates) but drives PPO toward a near-full fleet: normalized cost about 0.90 on every workload, flagged `full_fleet`.
  - Doubling it (`full-cost-high-v1`) makes PPO underprovision: SLA 0.23–0.32, queue pressure about 0.2, and it fails.
- **The SLA and queue terms are necessary.**
  - Without them (`latency-cost-v1`), PPO underprovisions on every workload (flagged `underprovisioning`; SLA 0.33–0.42 at cost about 0.55).
  - Adding only the SLA term restores feasibility.
  - With queue but no churn term (`latency-cost-sla-queue-v1`), PPO still passes but keeps a persistent backlog on ramp-down (queue pressure 0.20), and its selected DQN thrashes and underprovisions.
- **The churn term (0.1) has little PPO effect:** PPO churn is 0.01–0.04 under every reward.
- **Episode return disagrees with the #78 objective.**
  - For 7 of 8 rewards, the DQN screening candidate with the highest mean episode return is not SLA-feasible; the exception is `full-cost-low-v1`, whose top-return candidate `dqn-c01` is feasible.
  - Where the reward contains the SLA term, return and mean SLA are strongly rank-anti-correlated (Spearman −0.92 to −0.99).
  - Under `latency-cost-v1`, which has no SLA or queue term, the correlation is +0.21: higher return goes with *more* SLA violation, a textbook reward mismatch.
- Selecting on return would have picked infeasible policies, which is why reward is never a selection criterion.
- **Negative results kept:** `full-default-v1` still has no feasible DQN (0/20, reproducing #79), and 5 of 8 rewards have none.

No robustness evaluation was run in #20, including post-freeze checks: reward selection is nominal-only, and #65/#81 remain untouched. No held-out or test data was used, and the reward normalization is unchanged. The ablation compares rewards under fixed learners; it declares no algorithm winner.

## Sim-to-real protocol (#72)

#72 freezes the contract for the later **local systems-in-the-loop / Knative sim-to-real validation** before any live result exists and before #46 held-out results can influence it. It is protocol work only: no controller was run on the replay windows, no live run happened, and no model was trained.

**Research question.** Do controller behavior and trade-offs observed in simulation transfer to a real serving system? It does not require RL to win: a negative transfer result, or a conventional baseline winning, is valid evidence. Local Knative provides real HTTP traffic, containers, scheduling, networking, cold starts and measurement noise. It is **not** a claim that it reproduces AWS/Azure/GCP internals; real-cloud validation is #32.

**Artifacts** (each with a content-derived ID; regenerate or verify with `python -m scalerl.evaluation.sim_to_real freeze|check`):

| Artifact | ID |
|---|---|
| [`sim-to-real-protocol-v1.json`](../benchmarks/v1/sim-to-real-protocol-v1.json) | `b7bf45c109f1` |
| [`live-replay-manifest-v1.json`](../benchmarks/v1/live-replay-manifest-v1.json) + [`live-replay-schedules-v1/`](../benchmarks/v1/live-replay-schedules-v1/) | `7803b71fdfe8` |
| [`controller-deployment-manifest-v1.json`](../benchmarks/v1/controller-deployment-manifest-v1.json) | `553ddf3e512b` |

The protocol records `held_out_controller_outcomes_used: false` and `live_results_used: false`, and pins every upstream artifact ID (#65, #78, #79, #80, #81, #20); `check` refuses a tampered upstream.

**Controllers.**
- Tuned `threshold-v1` (0.6 / 0.2 / cooldown 3), `predictive-v1` and `predictive-seasonal-v1` as frozen by #80.
- DQN `dqn-c14` and PPO `ppo-c08` under `desired-replicas-v1` and `full-cost-low-v1` (#20), one canonical artifact each.
- `knative-native-v1`: the Knative Pod Autoscaler (concurrency metric, min 1, max 10, scale-to-zero disabled, other settings at the defaults of the version #73 pins). It is a real-system baseline with no simulated counterpart.
- Excluded: tabular Q-learning (#47 is open), static/random, and the DQN fallback families.

**Canonical seed rule (`canonical-seed-validation-v1`, validation only).** From the five #20 training seeds of the selected candidate, a seed is feasible if its SLA violation rate is ≤ the Threshold limit on every validation workload. Among feasible seeds, the lowest equal-workload mean normalized cost wins, then queue pressure, churn, SLA, and the lowest seed. If none is feasible, the smallest maximum SLA excess wins (then the same tie-breaks), with `canonical_seed_validation_feasible: false`. The full five-seed lineage is persisted, the evidence is checked against the #20 equal-seed means, and the tests recompute the choice.

| Family | Feasible seeds | Canonical | MLflow run |
|---|---|---|---|
| DQN `dqn-c14` | 0, 1, 2, 4 (seed 3 exceeds the limit on bursty and ramp-down) | seed 0 | `920b0ddb58df4d2b9f550431d2f8ceeb` |
| PPO `ppo-c08` | 0–4 | seed 4 | `9bac13a9447b4c64a44d9732568c949e` |

**Replay windows (`fixed-fraction-windows-v1`).** The source is the lexicographically smallest Azure test workload, `azure-test-1166400` (3600 s, 30 s bins). Three 600 s windows start at 20%, 50% and 80% of `duration − 600`, aligned down to the control interval: offsets **600, 1500, 2400 s** (bins 20–40, 50–70, 80–100), non-overlapping. The trace was read only to fingerprint it and to extract these predeclared windows; no window was chosen by inspection.

**Arrival schedules (`live-arrival-schedule-v1`, generator `binned-uniform-v1`).** Load seeds **0, 1, 2** per slice. Each bin's exact trace count is placed uniformly inside the bin, using `SeedSequence(load_seed, spawn_key=(0x61727276, slice_index))`, a stream domain-separated from every simulator RNG. Timestamps are frozen to the microsecond with a checksum. Every controller receives the identical schedule for a given slice and load seed.

**Control contract.** 30 s interval, min 1, max 10, `desired-replicas-v1`: action code `c` sets the desired fleet to `1 + c` (not ±1). Deterministic inference only, with no exploration and no online training. Scale-to-zero is disabled during comparisons.

**Observation contract (`scalerl-observation-v1`).** Simulation and live control must call the same builder, `scalerl.environment.observation.build_observation`, with the same constants. Simulation now calls it, bit-identical to the previous code. The protocol lists every feature's order, units, normalization (all in [0, 1], no clipping) and live equivalent. `utilization`, `queue_pressure` and `latency_pressure` are marked as **approximations** live. The cost feature is the **latest tick's** cost fraction, never accumulated cost. `episode_progress` is offset by the slice start bin so a 10-minute slice sees the progress it would inside the full trace.

**Cost contract.** Replica-seconds, plus the normalized abstract ScaleRL cost. No provider dollar pricing (#32).

**Transfer reporting (predeclared, descriptive only).**
- Absolute: simulation and real values for every controller × slice × load seed (SLA violation, p95 latency, cost proxy/replica-seconds, queue/failures, churn).
- Relative: controller − Threshold, per metric, in each domain.
- Diagnostics: sign agreement and side-by-side magnitudes.
- Secondary: Spearman rank correlation across domains.
- Knative-native: a real-only row.
- There is no binary pass/fail, no overall score and no "RL must win".

**Caveats.**
- The canonical DQN comes from a family whose bursty equal-seed mean sits exactly at the SLA limit (#20).
- Azure traffic is a few requests per second, against 50 rps per replica, so the replay may not exercise scaling. Any load amplitude scaling or capacity calibration must be a versioned protocol revision committed before live runs (#73/#74), never tuned on results.
- Live infrastructure (#73–#76) and the held-out suite (#46) are out of scope.

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
