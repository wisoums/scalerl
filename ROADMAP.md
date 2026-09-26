# ScaleRL Roadmap

**Portfolio-ready v1.0 target: October 31, 2026.**

The critical path is now: deterministic environment -> fair baselines -> frozen train/validation/test workloads -> MLflow -> City View -> Optuna -> recent-traffic RL observation -> Docker/CI -> DQN/PPO -> robustness-v1 -> multi-seed validation evidence -> constrained model selection (#78) -> action-semantics decision (#79) -> stronger predictive + startup-delay robustness (#80/#81) -> reward freeze (#20) -> sim-to-real/artifact freeze (#72) -> held-out evaluation (#46) -> portfolio demo.

## Completed — M0/M1 foundations

- package/CI/lint/type/test foundation
- deterministic simulation clock
- validated simulator configuration
- synthetic workload traces/generators
- replica lifecycle/startup delay
- request queue/service-capacity model
- latency/SLA/cost model
- Gymnasium environment
- environment determinism/invariant validation
- common controller protocol
- random/static baselines

## September 24–October 4 — M2 baselines + M3 data/MLOps foundation

### Controllers

- #12 threshold/target-tracking baseline
- #13 cooldown/anti-thrashing and fair threshold tuning
- #14 predictive baseline
- #63 queue-aware backlog recovery for the predictive baseline
- #56 Live City autoplay / animated single-run playback after #13/#14/#63
- #47 optional tabular Q-learning educational baseline

### Data / experiment isolation

- #43 Azure Functions Invocation Trace 2021 loader
- #18 freeze explicit training, validation, and held-out test suite **before deep-RL tuning**

### MLOps + early visualization

- #17 MLflow experiment tracking/run contract
- **#37 Streamlit Scenario Lab City View foundation immediately after #17**
- #54 shared Optuna hyperparameter optimization infrastructure
- #58 four-tick recent traffic history in the RL observation contract
- #44 reproducible Docker training + Optuna Dashboard + MLflow/PostgreSQL/S3-compatible artifact store (Garage)
- #45 full CI pipeline: quality, Python 3.11/3.12 tests, package, Docker Compose MLOps smoke (MLflow/Garage/Optuna), short SB3 smoke, GHCR release image

### Exit condition

A fair tuned threshold baseline exists; synthetic and selected Azure workloads use the same `WorkloadTrace`; the final test suite is frozen; MLflow establishes reproducible run identity; the Streamlit City View makes simulator/controller behavior visible; Optuna provides shared train/validation-only tuning; Docker and CI can reproduce/track a training smoke run.

## October 5–11 — M3 DQN

### Must have

- #15 SB3 DQN training pipeline
- #58 frozen four-tick recent-demand observation used by learned policies
- #54 Optuna study infrastructure used for DQN hyperparameter search
- common SB3-to-Controller adapter
- MLflow params/metrics/artifacts/model logging
- training/validation only; no held-out test use
- short CI smoke training plus separate real training run

### Exit condition

DQN training is reproducible from config/container, produces an MLflow run and model artifact, and can be evaluated through the same controller harness as baselines.

## October 12–18 — M3/M4 PPO + benchmark harness

### Must have

- #16 PPO training using the same run contract and #54 Optuna tuning infrastructure
- #65 stochastic dynamics + delayed-telemetry robustness scenarios after DQN/PPO are available
- #19 multi-seed evaluation/statistical summaries across nominal/robustness conditions
- identical controller evaluation schema
- model/environment compatibility checks

### Exit condition

Static, tuned threshold, predictive, DQN, and PPO can be evaluated automatically on identical frozen workloads/configs/seeds. Q-learning is included if #47 is completed.

## October 19–25 — M4 methodology freeze + real-trace analysis

### Must have

- #78 freeze constrained cost-aware model selection after #19 exposed the SLA-first full-fleet failure mode — done: `selection-v2-cost-under-sla` committed at `benchmarks/v1/selection-v2-cost-under-sla.json`; v1 selectors and #19 results unchanged
- #79 compare existing `Discrete(3)` codes 0/1/2 (effects -1/0/+1) with direct desired-replica actions and freeze the final action contract
- #80 preserve existing `predictive-v1` / `forecast-plus-backlog-v1` lineage and add a stronger cloud-style proactive predictive baseline
- #81 add a separate seeded startup-delay robustness extension while preserving robustness-v1
- #20 reward-function ablation on the final #79 action contract using the frozen #78 selection rule
- #72 freeze canonical controller artifacts and sim-to-real protocol before held-out outcomes can influence them
- #65 robustness-v1 remains frozen and unchanged
- #46 held-out Azure trace comparison under the finalized action/baseline/reward contract and predeclared robustness conditions
- latency/SLA/cost/queue/churn analysis
- per-seed raw results and dispersion
- MLflow run IDs for reported results
- honest failure-case analysis and conclusion

### Exit condition

All methodology choices are frozen using train/validation evidence only, then reproducible held-out evidence answers the research question on controlled synthetic workloads and selected real production traces.

## October 26–31 — dashboard + portfolio release

### Must have

- polished README with final tables, limitations, dataset citation, and reproduction commands
- architecture/results diagrams
- polish the earlier #37 Scenario Lab City View and #56 Live City playback
- #38 controller replay/comparison if time permits
- #59 unified ScaleRL platform navigation + native Optuna/MLflow integration
- #39 ScaleRL Experiment Hub / MLflow-backed saved Results Explorer if time permits
- tagged v1.0 release

### Production-inspired follow-ups

- #22 FastAPI shadow-mode service
- #23 safety/fallback controller
- #24 inference/demo container
- #32 AWS Lambda real-cloud validation with cost guardrails

These are valuable but must not compromise the experiment quality above.

## Definition of v1.0 done

ScaleRL v1.0 is portfolio-ready when:

1. the simulator is deterministic and tested;
2. static, reactive, and both short-history and stronger proactive predictive baselines are implemented fairly;
3. training/validation/test workloads are explicitly separated before final tuning;
4. DQN and PPO train reproducibly;
5. training/evaluation runs are tracked in MLflow with config/model artifacts;
6. Docker and CI reproduce the relevant smoke pipeline on a clean machine, with ScaleRL, Optuna Dashboard, and MLflow available through the documented local platform stack;
7. final evaluation uses the action semantics frozen by #79 and held-out workloads/multiple seeds, with frozen capacity-jitter, telemetry-delay, and startup-delay robustness checks;
8. at least part of the final evaluation uses attributed real Azure production traces;
9. results report raw systems metrics rather than only RL return;
10. another developer can reproduce the principal benchmark from documented commands.
