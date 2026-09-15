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
