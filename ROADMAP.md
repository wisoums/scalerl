# ScaleRL Roadmap

**Portfolio-ready v1.0 target: October 31, 2026.**

The schedule prioritizes a defensible ML result over feature count. The v1 release is successful if it contains a deterministic simulator, credible non-RL baselines, DQN and PPO, rigorous held-out evaluation, and a polished explanation of the findings. Advanced production integration and additional workload realism are stretch goals if they threaten the experimental core.

## September 14–20 — M0 + M1 simulator foundations

### Must have
- package layout and developer tooling
- CI, linting, typing, tests
- architecture, reward, experiment, and RL-rationale documentation
- deterministic simulation clock
- typed simulator configuration
- workload trace abstraction
- replica lifecycle and startup delay

### Exit condition
Core simulator components have deterministic unit tests and no dependence on wall-clock time.

## September 21–27 — Finish M1 Cloud Simulation Environment

### Must have
- synthetic workload generators
- queue/service-capacity model
- latency, SLA, and cost model
- Gymnasium environment
- environment invariants and determinism validation

### Exit condition
A fixed seed/config produces reproducible trajectories, valid observations/actions, and believable overload/recovery behavior.

## September 28–October 4 — M2 Traditional Baselines

### Must have
- random and static controllers
- threshold/target-tracking controller
- cooldown and anti-thrashing behavior
- predictive autoscaling baseline
- common controller/evaluation interface

### Exit condition
Every baseline can run the same workload trace and produce the same evaluation schema.

## October 5–11 — M3 Deep RL: DQN + experiment infrastructure

### Must have
- reproducible experiment configuration
- training/evaluation separation
- metrics/checkpoint persistence
- DQN training pipeline
- learning curves and basic sanity evaluation

### Exit condition
DQN demonstrably learns behavior above a random-policy sanity check on training/validation scenarios without using final benchmark traces.

## October 12–18 — M3/M4 PPO + benchmark harness

### Must have
- PPO training pipeline
- frozen held-out benchmark workload suite
- multi-seed evaluation harness
- consistent metrics for static, threshold, predictive, DQN, and PPO controllers

### Exit condition
All five meaningful controllers can be evaluated automatically on the same held-out suite.

## October 19–25 — M4 Analysis and research results

### Must have
- multi-seed controller comparison
- statistical summaries / dispersion
- reward-function ablation
- latency, SLA, cost, dropped/completed request, and scaling-churn analysis
- failure-case analysis
- honest conclusion about where RL wins, ties, or loses

### Exit condition
There is enough reproducible evidence to answer the project research question without relying on episodic reward alone.

## October 26–31 — M7/M8 Portfolio release

### Must have
- polished README with final result table and limitations
- reproducible benchmark command/config
- architecture and results diagrams
- small visual demo or replay of controller behavior
- resume-ready project summary
- tagged v1.0 release

### Nice to have if the core is already stable
- lightweight interactive dashboard
- FastAPI shadow-mode inference endpoint
- Dockerized demo

## Stretch after or only if ahead of schedule — M5/M6

- replica failures and degraded capacity
- changing startup times
- stronger non-stationary distribution shifts
- model predictive control baseline
- production-style shadow mode
- hard fallback/safety controller around learned policy
- full Docker/deployment documentation

These are valuable engineering additions, but they should not delay a rigorous October 31 benchmark and release.

## Definition of v1.0 done

ScaleRL v1.0 is considered portfolio-ready when:

1. the simulator is deterministic and tested;
2. static, reactive, and predictive baselines are implemented fairly;
3. DQN and PPO train reproducibly;
4. final evaluation uses held-out workloads and multiple seeds;
5. results report raw systems metrics rather than only RL return;
6. the README explains why RL was a reasonable hypothesis and whether evidence ultimately supports it;
7. another developer can reproduce the principal benchmark from documented commands.
