# Reward Design

The reward should encode the real systems trade-off instead of rewarding replica count or CPU utilization directly.

A starting formulation is:

```text
reward = -(
    w_latency * normalized_latency
    + w_cost * normalized_cost
    + w_sla * sla_violation
    + w_churn * scaling_action_cost
    + w_queue * normalized_queue
)
```

## Objectives

The policy should learn to:

- maintain application latency within an SLA target;
- avoid sustained request queues;
- minimize infrastructure spend;
- avoid unnecessary scale-up/scale-down oscillation;
- account for delayed consequences such as replica startup time.

## Guardrails

Reward optimization is not allowed to bypass hard safety constraints. The environment enforces minimum/maximum replicas and valid actions independently of learned policy behavior.

## Reward-engineering experiments

#20 ran these experiments as the predeclared `reward-ablation-v1` study (spec `benchmarks/v1/reward-ablation-v1.json`), frozen before any run of record. It tested eight variants: four component variants and four cost/SLA sensitivity variants.

| Variant | latency | cost | SLA | queue | churn |
|---|---|---|---|---|---|
| `latency-cost-v1` | 1.0 | 1.0 | 0 | 0 | 0 |
| `latency-cost-sla-v1` | 1.0 | 1.0 | 1.0 | 0 | 0 |
| `latency-cost-sla-queue-v1` | 1.0 | 1.0 | 1.0 | 1.0 | 0 |
| `full-default-v1` (the previous default) | 1.0 | 1.0 | 1.0 | 1.0 | 0.1 |
| `full-cost-low-v1` | 1.0 | 0.5 | 1.0 | 1.0 | 0.1 |
| `full-cost-high-v1` | 1.0 | 2.0 | 1.0 | 1.0 | 0.1 |
| `full-sla-low-v1` | 1.0 | 1.0 | 0.5 | 1.0 | 0.1 |
| `full-sla-high-v1` | 1.0 | 1.0 | 2.0 | 1.0 | 0.1 |

Only component inclusion and weights varied; the normalized penalty terms were unchanged. The final weights were chosen by the predeclared reward-level rule on validation system metrics (SLA feasibility, then cost, queue, churn and SLA), never by training or episodic reward. The design and full results are in [EXPERIMENTS.md](EXPERIMENTS.md#reward-ablation-20).

## Reward ablation result (#20)

The predeclared ablation (`reward-ablation-v1`, see [EXPERIMENTS.md](EXPERIMENTS.md#reward-ablation-20)) froze **`full-cost-low-v1`** in `benchmarks/v1/reward-contract-v1.json`:

```text
latency 1.0, cost 0.5, SLA 1.0, queue 1.0, churn 0.1
```

It was the only variant for which both learned families (the #78-selected DQN, retrained on 5 seeds, and the fixed `ppo-c08` on 5 seeds) met the Threshold SLA on every validation workload. Two caveats:
- the DQN margin on `syn-val-bursty` is exactly zero;
- PPO under this reward runs a near-full fleet.

The component normalization is unchanged, and `RewardWeights()` keeps its historical defaults (cost 1.0) for reproducibility. Final experiments request the frozen weights explicitly.

## Churn counts scaling events, not replica magnitude (#79 note)

The implemented churn term is `weights.churn * (applied_replica_change != 0)`: one penalty per tick whose applied replica change is non-zero, whatever its size. Under the historical `delta-v1` contract every scaling tick moves exactly one replica, so events and magnitude coincide. Under `desired-replicas-v1` one decision can move several replicas (`+6` in one tick costs the same churn penalty as `+1`).

#79 deliberately does **not** change the reward: the action-contract comparison must not be confounded with a reward change. Instead it reports magnitude separately as `action.*` diagnostics (`total_absolute_replica_change`, `mean_absolute_replica_change_when_scaling`, `max_absolute_replica_change_in_one_tick`, `max_pending_replicas`) next to the event-based `scaling_actions`/`churn_rate`. Fewer scaling ticks therefore do not imply less total scaling.

#20 kept these semantics. `reward-ablation-v1` left every component's normalization, including the event-based churn term, unchanged and varied only component inclusion and weights. Magnitude-sensitive churn was not part of `reward-ablation-v1`; investigating it would require a separately versioned future study.
