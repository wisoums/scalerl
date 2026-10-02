# ScaleRL Roadmap

ScaleRL is an **ongoing student learning project**, not a claim of scientific completeness or production readiness.

The roadmap is intentionally iterative: each phase adds a little more realism or a harder generalization question, while preserving earlier results instead of rewriting them when they are inconvenient.

## Current checkpoint

The v1 experimental stack is complete enough to expose a real failure mode:

- deterministic simulator and Gymnasium environment;
- synthetic workloads + selected Azure Functions 2021 traces;
- Threshold and predictive baselines;
- DQN/PPO training with Stable-Baselines3;
- Optuna tuning;
- MLflow lineage;
- multi-seed evaluation;
- frozen action/reward/robustness methodology;
- held-out Azure evaluation (#46);
- Docker/CI and Scenario Lab.

The frozen #46 result showed that the v1 DQN/PPO policies overprovisioned badly on very-low-load Azure TEST traces.

That result is preserved.

It does **not** answer whether RL can be useful when trained over a much broader workload distribution.

## Current critical path

~~~text
#46  frozen v1 evidence
  ↓
#122 documentation/roadmap sync
  ↓
#115 freeze benchmark-v2 generalization methodology
  ↓
#116 heterogeneous datasets/provenance
  ↓
#117 workload taxonomy + balanced episode sampler
  ↓
#118 train/select generalist-workload DQN/PPO
  ↓
#119 fresh untouched benchmark-v2 TEST evaluation
  ↓
#26  final synthesis of what improved and what did not
~~~

The new phase must not use benchmark-v2 TEST outcomes to redesign training.

## Phase A — benchmark-v2 workload generalization

### #115 — Freeze the methodology first

Before new generalist training begins, define:

- eligible dataset sources;
- train/validation/test grouping;
- application/window leakage rules;
- traffic-intensity strata;
- traffic-shape taxonomy;
- domain/source labels;
- training-sampler weighting;
- model-selection rule;
- seeds;
- metrics;
- untouched TEST tiers.

The benchmark-v2 TEST suite must be frozen before #118 training.

### #116 — Expand dataset coverage

Implement only the sources approved by #115.

Candidate sources currently include:

- Azure Functions 2019;
- broader Azure Functions 2021 windows/apps;
- Alibaba microservice traces;
- synthetic generators used deliberately to cover missing regimes.

Google ClusterData and SeBS may contribute different kinds of system/application evidence, but they must not be misrepresented as ordinary request-rate traces.

### #117 — Build workload taxonomy + episode sampler

Create reproducible training coverage across:

- near-idle/low/moderate/high/overload intensity;
- steady/ramp/drop/spike/burst/seasonal/noisy/regime-switching shapes;
- multiple source/application domains.

The training sampler should avoid one large source dominating by accident.

### #118 — Train the generalist policies

Train DQN/PPO using the benchmark-v2 TRAIN sampler.

Keep the Stage-A simulator contract fixed so the experiment mainly changes **workload diversity**, not every system assumption at once.

Use TRAIN for learning and VALIDATION for model/hyperparameter selection.

Freeze canonical artifacts before touching benchmark-v2 TEST.

### #119 — Open fresh TEST

Evaluate the frozen generalist models on untouched benchmark-v2 TEST data.

Planned evaluation tiers include:

- unseen apps/windows;
- unseen workload combinations;
- source/domain holdout where feasible.

Compare against strong Threshold/Predictive baselines under matched conditions.

Negative results remain valid.

## Phase B — non-stationarity and robustness

#21 extends benchmark-v2 with explicitly versioned scenarios such as:

- quiet -> flash crowd;
- high load -> sudden collapse;
- burst intensity outside TRAIN range;
- capacity loss/degradation;
- additional startup/telemetry disturbances.

These scenarios must keep clear TRAIN/VALIDATION/TEST roles.

Historical #65/#81 robustness artifacts remain unchanged.

## Phase C — environment generalization

#120 studies whether a single learned policy can generalize across a wider **system** space, not only workload shapes.

Potential dimensions:

- fleet size;
- per-replica capacity;
- startup delay;
- control cadence;
- SLA target;
- initial replica count;
- service characteristics.

This may require a future fixed-dimensional observation-v2 and fleet-size-independent action-v2.

Those contracts are not selected yet.

This phase does not block Stage-A #118/#119.

## Phase D — simulator-to-real validation

Controller-independent infrastructure can proceed in parallel:

- #73 local Knative testbed;
- #74 generic deterministic HTTP replay harness;
- #23 safety/fallback layer.

After #119:

~~~text
#119
  ↓
#121 freeze sim-to-real-v2
  ↓
#75 run frozen generalist controllers on Knative
  ↓
#76 publish simulator-vs-Knative comparison
~~~

#72 remains the historical v1 sim-to-real protocol and is not rewritten.

The live experiment should include workload regimes that actually exercise scaling where local hardware allows it.

## Secondary engineering work

### #111 — Compute efficiency

After #118, benchmark representative generalist DQN/PPO training and inference:

- wall time;
- steps/sec;
- CPU/GPU utilization;
- RAM/VRAM;
- inference p50/p95.

This is engineering evidence, not controller-selection evidence.

### #47 — Tabular Q-learning

Optional educational baseline.

It is useful for learning, but it is not required for the main DQN/PPO generalization result.

## Final synthesis

#26 combines:

- historical v1 methodology and #46 result;
- benchmark-v2 design;
- generalist training;
- fresh #119 TEST evidence;
- optional #21/#111/#120 results;
- optional #76 sim-to-real evidence.

The final write-up should answer narrowly:

- what changed when training diversity increased;
- where RL helped, tied, or lost;
- what failed to generalize;
- what conclusions remain unsupported by the simulator.

It should **not** claim a universal cloud-autoscaling result.

## Learning / dashboard / project polish

These tracks make the project easier to understand and use, but they should not drive scientific choices.

### Learning and extension

- #85 umbrella
- #86 beginner learning path
- #87 Guided Learn mode
- #88 stable Bring-Your-Own Controller API
- #89 researcher extension CLI/toolkit

### Dashboard / presentation

- #25 dashboard umbrella
- #38 controller comparison/playback
- #39 saved Results Explorer
- #59 navigation + native Optuna/MLflow integration
- #91 presentation umbrella
- #92 branding/assets
- #93 demo GIF/video
- #94 final README redesign
- #95 MkDocs site
- #96 ADRs
- #97 architecture/result visuals
- #98 repository/community polish
- #103 public Streamlit demo

The final README redesign (#94) should wait until #119 so it can distinguish:

- v1 narrow-training evidence;
- benchmark-v2 generalist evidence;
- any later sim-to-real evidence.

## v1.0.0

#99 is a **packaging/presentation milestone**, not a claim that ScaleRL has modeled every factor relevant to production autoscaling.

Before v1.0.0, the project should at minimum have:

- #119 fresh benchmark-v2 evidence;
- #26 final synthesis;
- reproducible commands;
- honest limitations;
- stable user-facing documentation.

Environment-generalization and additional real-system work may continue after v1.0.0.

ScaleRL is expected to remain an evolving personal learning project rather than a finished scientific model of cloud autoscaling.
