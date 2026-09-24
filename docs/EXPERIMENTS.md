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
