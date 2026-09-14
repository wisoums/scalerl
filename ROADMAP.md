# ScaleRL Roadmap

## M0 — Project Foundation

- package layout and developer tooling
- CI, linting, typing, tests
- architecture, reward, and experiment documentation

## M1 — Cloud Simulation Environment

- deterministic simulation clock
- workload trace model and generators
- replica lifecycle and startup delay
- queue/service-capacity model
- latency, SLA, and cost model
- Gymnasium environment and validation tests

## M2 — Traditional Baselines

- random and static controllers
- threshold/target-tracking controller
- cooldown and anti-thrashing behavior
- predictive autoscaling baseline

## M3 — Deep RL Agents

- DQN training/evaluation pipeline
- PPO training/evaluation pipeline
- reproducible configuration and checkpoints
- experiment tracking and learning curves

## M4 — Benchmarking

- held-out workload suite
- multi-seed evaluation
- statistical summaries
- reward and state ablations
- baseline-vs-RL comparison report

## M5 — Advanced Workloads

- abrupt spikes
- bursty/non-stationary traffic
- changing startup times
- replica failures and degraded capacity

## M6 — Production-Inspired Integration

- inference API
- shadow/recommendation mode
- safety bounds and fallback controller
- Docker image and deployment documentation

## M7 — Demo

- dashboard for traffic, replicas, latency, cost, and actions
- replayable benchmark scenarios
- polished diagrams and demo media

## M8 — v1.0 Research Release

- reproducible final benchmark
- documented findings and limitations
- release artifact and resume-ready project summary
