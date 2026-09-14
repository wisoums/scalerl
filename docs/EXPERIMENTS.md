# Experimental Methodology

ScaleRL evaluates every controller under the same simulator configuration and workload traces.

## Controllers

- Random policy — sanity check
- Static capacity — cost/performance reference
- Threshold/target-tracking autoscaler — reactive baseline
- Predictive autoscaler — forecasting baseline
- DQN — discrete deep-RL policy
- PPO — policy-gradient comparison

## Workload families

- steady load
- diurnal/seasonal load
- gradual ramp
- sudden traffic spike
- bursty stochastic traffic
- recovery after a spike
- optional replica failure scenarios

Training and evaluation workloads must be separated. At least one workload family should be held out from RL training to test generalization.

## Metrics

Primary metrics:

- mean and p95 response latency
- SLA violation rate
- infrastructure cost
- request completion/drop rate
- number of scaling actions
- replica-hours

Secondary metrics:

- episodic return
- queue depth
- convergence/sample efficiency
- action distribution

## Reproducibility

Every reported experiment should record configuration, controller version, random seed, workload seed, training steps, and software versions. Final comparisons should use multiple seeds and report dispersion rather than a single best run.

## Success criterion

The project is successful even if RL does not dominate every baseline. The research objective is to identify when learned sequential policies are useful, when simple controllers are sufficient, and what trade-offs cause each result.
