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
| Predictive (#14, #63) | a linear-trend forecast of its own window of past completed demand (`info["request_rate"]`), plus the current waiting queue (`info["queued_requests"]`, sizing only, never the forecast) and replica counts |
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

- **Actions:** `Discrete(3)`: 0 scale down, 1 hold, 2 scale up. DQN learns one Q-value per action and, when evaluated, picks the highest.
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

That is a **selection-objective failure mode**, not evidence of a pipeline bug. #78 freezes a v2 rule before further model selection:

1. satisfy a declared service/SLA constraint relative to the tuned Threshold baseline;
2. among feasible candidates, minimize normalized cost;
3. use queue/churn/SLA only as tie-breakers.

The original #19 results remain reproducible and documented.

### #79 — action granularity, not "continuous vs discrete"

The current Gymnasium action space is discrete:

```text
scale down by 1 | hold | scale up by 1
```

That can make burst recovery artificially sequential: a controller needing six additional replicas may require six control decisions.

Real horizontal autoscalers commonly calculate an integer **desired replica count** directly. Kubernetes HPA documents:

`desiredReplicas = ceil(currentReplicas × currentMetricValue / desiredMetricValue)`

and then updates the scale target: <https://kubernetes.io/docs/concepts/workloads/autoscaling/horizontal-pod-autoscale/>.

So #79 compares:

- `delta-v1`: the existing `Discrete(3)` action codes `0/1/2` = scale-down/hold/scale-up, with replica-count effects `-1/0/+1`;
- `desired-replicas-v1`: discrete action codes mapped directly to an integer target fleet size.

The values `-1/0/+1` are **effects on replica count**, not the values accepted by `AutoscalingEnv.step()`; under the current contract `-1` is an invalid action code.

This does **not** mean DQN is invalid because it lacks continuous actions. Horizontal replica counts are discrete. Continuous/hybrid action spaces would be a different problem, such as vertical CPU/RAM allocation.

### #80 — stronger proactive predictive baseline

The current predictor is startup-aware but intentionally simple. Its existing lineage remains `predictive-v1` with forecast method `linear-trend` and capacity policy `forecast-plus-backlog-v1`; #80 must not invent or rename it to a different identifier. #80 adds a separate stronger recurring-pattern / proactive pre-provisioning baseline so the final RL comparison is not against an artificially weak forecaster.

The stronger baseline must still use past information only. An optional oracle may be used only as a clearly labeled future-peeking upper bound, never as a fair deployable competitor.

### #81 — stochastic startup delay

`robustness-v1` varies service capacity and telemetry freshness, but startup/readiness time is fixed. #81 adds a **separate** seeded per-replica startup-delay stress test. It must not retroactively alter the four frozen #65 scenarios.

AWS explicitly notes that actual launch time can vary with factors such as instance size and startup scripts: <https://docs.aws.amazon.com/autoscaling/ec2/APIReference/API_PredictiveScalingConfiguration.html>.

### Freeze order

No held-out test workload may be inspected while resolving these issues.

The pre-held-out sequence is:

```text
#19 multi-seed validation evidence
  ↓
#78 constrained selection rule
  ↓
#79 final action semantics
  ├─→ #80 stronger predictive baseline
  └─→ #81 startup-delay robustness extension
          ↓
#20 reward ablation on the final action contract
          ↓
#72 freeze controller artifacts + sim-to-real protocol
          ↓
#46 held-out evaluation
```

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
