# ScaleRL Benchmark v2 — generalization protocol

- **Protocol:** [`generalization-protocol-v1.json`](generalization-protocol-v1.json) (`benchmark-v2-generalization-protocol-v1`)
- **Code:** [`src/scalerl/evaluation/generalization_protocol.py`](../../src/scalerl/evaluation/generalization_protocol.py)
- **Issue:** #115, frozen before any generalist model is trained and before any benchmark-v2 TEST data is opened

```bash
python -m scalerl.evaluation.generalization_protocol check    # verify upstream + artifact + content ID
python -m scalerl.evaluation.generalization_protocol show-id  # print the protocol ID
python -m scalerl.evaluation.generalization_protocol write    # regenerate (must be byte-identical)
```

This directory freezes **rules only**. It deliberately contains no concrete future workload IDs. #116 prepares the approved sources. #117 then applies these rules and commits the content-addressed `benchmark-v2-workloads-v1` manifest, which must happen **before #118's first training run**. TEST membership is immutable from that point.

Every rule has a tested reference implementation in the module (role assignment, compositional-cell choice, evaluation sampling, the TRAIN episode sampler, validation selection, the canonical seed and the clustered bootstrap). #117–#119 must use these functions rather than re-implement the prose.

## Research question and stages

> Within ScaleRL's simplified simulator and the unchanged v1 system contract, does training DQN/PPO over a broader, predeclared workload distribution improve cost/SLA generalization to unseen windows, applications, taxonomy combinations and a held-out trace source, relative to fixed reactive and predictive baselines?

**Stage A (`generalist-workload-v1`) changes only the training workload distribution.** Everything else is the exact #46 configuration, and the check verifies it against `heldout-evaluation-v1.json`:

- `desired-replicas-v1` and `scalerl-observation-v1`;
- 1–10 replicas, starting at 1;
- 50 rps per replica;
- 60 s fixed startup delay;
- 30 s control interval and 3600 s episodes (120 ticks);
- 0.5 s SLA;
- `reward-contract-v1` / `full-cost-low-v1`.

A contract change is allowed only for a demonstrated correctness bug. It needs a new protocol version, made before #118. Stage B (observation-v2/action-v2, varying fleet, capacity, cadence and SLA) belongs to #120 and is never mixed in.

The 12 observation features contain no source, domain, dataset or taxonomy information. The taxonomy labels are used for sampling, stratified selection and reporting only.

## Sources

Licenses and semantics were checked against the official pages on 2026-10-02.

| Source | Stage-A role | License | Notes |
|---|---|---|---|
| Azure Functions 2019 | TRAIN, VALIDATION, TEST-A, TEST-B | CC-BY-4.0 | Workload = `HashApp`; **leakage group = `HashOwner`**, which links possibly related apps. Per-minute counts are recorded **after execution**, so they bin completions and serve only as an arrival proxy. Each minute rate is repeated for its two 30 s ticks. Citation: Shahrad et al., ATC 2020. |
| Azure Functions 2021 | **TEST-C only** | CC-BY-4.0 | Arrivals are `end_timestamp - duration`, per app. Windows that overlap any v1 Azure hour are excluded, which removes both #46 TEST hours from benchmark-v2. Citation: Zhang et al., SOSP 2021. |
| synthetic-v2 | TRAIN and VALIDATION supplement, never TEST | MIT (generated) | Fills TRAIN cells with fewer than 20 Azure 2019 windows. Seeded, frozen in the #117 manifest. |
| Alibaba microservices 2021 | **excluded** | none found | No LICENSE file and no usage terms anywhere in the repository. The trace is only 12 h long, with per-container MCR at 60 s. Adding it later needs a new protocol version before #118. |
| Google ClusterData 2019 | excluded (possible #120 evidence) | CC-BY-4.0 | Resource requests/usage, not a request trace. It is never converted into one. |
| SeBS | excluded (possible sim-to-real-v2 calibration) | BSD-3-Clause | An application benchmark suite, not a trace. |

Real traces are never rescaled. Windows with zero requests, and windows whose mean exceeds 1000 rps (2× fleet capacity), are excluded and counted.

## Windows and split (`grouped-temporal-split-v1`)

Windows are aligned one-hour slices, 336 per Azure source (14 days). For Azure 2019, window `(day d, hour h)` has index `24(d−1)+h`. For Azure 2021, window `k` covers `[3600k, 3600(k+1))` in trace-relative seconds.

| Period | Window indices | Days |
|---|---|---|
| development | 0–263 | 1–11 |
| embargo | 264–287 | 12 |
| temporal-test | 288–335 | 13–14 |

Each Azure 2019 owner is assigned by `SHA-256("scalerl-benchmark-v2-v1|group|azure-functions-2019|<HashOwner>")`: 80% development, 10% validation, 10% application-test.

| Owner group | development period | embargo | temporal-test period |
|---|---|---|---|
| development | **TRAIN** | reserved | **TEST-A** |
| validation | **VALIDATION** | reserved | reserved |
| application-test | **TEST-B application** | reserved | reserved |

- **Neighbouring hours never straddle TRAIN and TEST.** TEST-A starts at least 24 hours after the last TRAIN hour of the same owner. The pre-review draft instead hash-split windows *within* an app, which #115 forbids.
- **TEST-B compositional:** up to two (intensity, shape) cells are chosen from Azure 2019 candidate-TRAIN counts. A cell needs at least 20 windows from at least 5 owners, and its intensity and shape must each still be supported elsewhere. Cells are ranked by SHA-256, and the second cell must differ in both factors. The chosen cells are removed from TRAIN and VALIDATION for **every** source, synthetic included.
- **TEST-C:** the whole Azure 2021 source. This is a same-provider shift (different year, app population, sampling and binning), **not** a cross-provider domain shift.
- Reserved windows are never opened. A window removed by a rule is never moved into another role.

VALIDATION and TEST sets are capped stratified samples. Within each role and each (source, intensity, shape) cell, windows are taken in SHA-256 rank order, at most one per group, up to **4** per cell for VALIDATION and **8** for TEST.

## Taxonomy (`workload-taxonomy-v1`)

**Intensity** uses the mean rps, with inclusive upper bounds:

| Stratum | Mean rps |
|---|---|
| near-idle | ≤ 5 |
| low | ≤ 25 |
| single-replica | ≤ 50 |
| multi-replica-moderate | ≤ 150 |
| multi-replica-high | ≤ 400 |
| near-saturation | ≤ 500 |
| overload | ≤ 1000 |

A separate peak-pressure label (from the 30 s peak) is kept for analysis only.

**Shape** (`traffic-shape-minute-v1`) is classified on one-minute means, which is the native Azure 2019 resolution. The first matching rule wins:

1. steady
2. ramp-up
3. ramp-down
4. periodic (an ACF peak after a dip)
5. spike (one contiguous excursion above 2× the median)
6. regime-switch (one change point explains ≥ 60% of the variance)
7. bursty
8. noisy-other

The exact thresholds are in the JSON. These are deterministic strata, not semantic truth; for example, a single smooth one-hour wave falls into regime-switch.

## Training, selection, seeds

- **Sampler** (`hierarchical-balanced-sampler-v1`, TRAIN only): it draws the source with weights Azure 2019 = 0.75 and synthetic = 0.25, then a uniform intensity stratum, then a uniform shape, then a uniform window. The RNG is `PCG64(SeedSequence(training_seed, spawn_key=(0x73616D70,)))`, so the episode sequence is exactly reproducible. A non-TRAIN window raises an error.
- **Seeds:** training seeds 0–4. Screening uses seed 0, and that run is reused as the seed-0 member. Evaluation and dynamics seeds are 0, under nominal conditions.
- **Selection** (`benchmark-v2-cost-under-sla-v1`, within one family, VALIDATION only):
  - a candidate is feasible if every window's SLA violation rate is ≤ frozen `threshold-v1` + 0.05, and every (source, intensity) group's balanced SLA is ≤ Threshold + 0.02;
  - among feasible candidates, the lowest balanced normalized cost wins (equal weight per source, then intensity, then shape, then window);
  - ties break on queue pressure, then churn, then SLA, then candidate ID;
  - if no candidate is feasible, nothing is selected (a diagnostic fallback is reported).
  - This adapts #78's per-workload rule to hundreds of heterogeneous windows. SLA stays a constraint and cost stays the objective.
- #118 must commit candidate spaces, trial counts and timesteps before its first result.

## Evaluation and reporting (#119)

- Baselines keep their frozen v1 configurations and are **not re-tuned**: `threshold-v1`, `predictive-v1`, and `predictive-seasonal-v1`. The seasonal baseline runs only where the app has a prior day of demand history, and is reported as a separate subset.
- Metrics are the #46 primary metrics, plus `overprovision_ratio` and the fractions of ticks at the minimum and maximum fleet. Reward is secondary.
- All five seeds are reported, with no winner score.
- Results are reported separately per tier, source, intensity, shape, and low-load vs high-load strata.
- Paired deltas vs Threshold come with a group-clustered percentile bootstrap 95% interval (10,000 resamples).
- #21 scenarios are never used for TRAIN, are diagnostic-only on VALIDATION, and form a TEST stress tier only if #21 freezes them before #118.

## Relationship to #46

#46 remains frozen v1 evidence and is not rerun or reinterpreted. Its TEST hours are excluded from benchmark-v2, and no benchmark-v2 item was chosen using #46 outcomes. Low-load coverage comes from Azure 2019 near-idle and low windows and from synthetic data. #119 may show #46 next to benchmark-v2 results, but labelled as v1 evidence.
