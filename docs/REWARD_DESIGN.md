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

The project should include ablations for:

1. latency + cost only;
2. latency + cost + SLA penalty;
3. latency + cost + SLA + scaling-churn penalty;
4. sensitivity to each reward weight.

The final weights should be justified empirically rather than selected only because they produce the highest training reward.

## Churn counts scaling events, not replica magnitude (#79 note)

The implemented churn term is `weights.churn * (applied_replica_change != 0)`: one penalty per tick whose applied replica change is non-zero, whatever its size. Under the historical `delta-v1` contract every scaling tick moves exactly one replica, so events and magnitude coincide. Under `desired-replicas-v1` one decision can move several replicas (`+6` in one tick costs the same churn penalty as `+1`).

#79 deliberately does **not** change the reward: the action-contract comparison must not be confounded with a reward change. Instead it reports magnitude separately as `action.*` diagnostics (`total_absolute_replica_change`, `mean_absolute_replica_change_when_scaling`, `max_absolute_replica_change_in_one_tick`, `max_pending_replicas`) next to the event-based `scaling_actions`/`churn_rate`. Fewer scaling ticks therefore do not imply less total scaling. Whether churn should become magnitude-sensitive is a reward-design question for #20.
