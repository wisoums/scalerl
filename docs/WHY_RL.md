# Why Reinforcement Learning?

ScaleRL is a **personal student learning project**. RL is used because autoscaling is a good place to learn sequential decision-making, delayed effects, and ML-system evaluation — not because this repository assumes RL should replace real production autoscalers.

The current project is intentionally simplified. It does not model every parameter that could matter in a production cluster, and its conclusions only apply to the exact simulator/workload contracts used in each experiment.

## The narrow question

The current question is:

> **Within ScaleRL's simplified simulator, can DQN or PPO learn a useful cost/SLA scaling policy across diverse workload patterns compared with strong reactive and predictive baselines?**

That is much narrower than:

> "Is RL the best way to autoscale arbitrary cloud systems?"

ScaleRL is not designed to prove the second statement.

## Why RL is interesting to learn here

Autoscaling is sequential:

- a decision is repeated every control interval;
- scaling actions change future capacity;
- new replicas take time to become ready;
- scaling down can increase future queueing;
- overprovisioning protects service quality but wastes resources;
- short-term choices can create longer-term churn and cost.

That makes RL a reasonable learning tool because the action changes the future state and future reward.

## Why not supervised learning alone?

Supervised learning is useful for forecasting demand, latency, or required capacity.

But forecasting answers:

> "What is likely to happen?"

Control asks:

> "What should I do now, given what may happen and the future consequences of this action?"

ScaleRL therefore keeps predictive controllers as important baselines instead of treating forecasting as a competing idea that must disappear.

## Why not simpler controllers?

They may be better.

Threshold, predictive, and classical-control approaches can be easier to understand, cheaper to run, and more robust than RL.

A useful ScaleRL result can therefore be:

> "The simpler controller handled this regime just as well or better."

The project is not considered a failure when RL loses.

## What the current simulator actually contains

The v1 environment currently reasons about a limited set of signals such as:

- recent demand pressure;
- active/pending replicas;
- utilization/capacity pressure;
- queue pressure;
- latency pressure;
- a simplified cost signal;
- startup delay;
- a fixed control cadence;
- a fixed fleet range and SLA contract.

It does **not** currently model the full set of factors that may matter in real systems, such as richer CPU/memory/I/O behavior, network/storage effects, heterogeneous services, request classes, resource requests/limits, multi-service dependencies, provider-specific schedulers, richer failures, or arbitrary deployment sizes.

That is why the project should be read as a learning environment, not a production autoscaling claim.

## What v1 taught us

The first frozen held-out Azure experiment (#46) exposed a real limitation.

The canonical v1 DQN/PPO policies were trained under a narrow workload distribution. On two very-low-load Azure TEST hours, they overprovisioned badly while Threshold and Predictive stayed at one replica.

That does not prove RL is generally bad at autoscaling.

It shows:

> **The v1 learned policies did not generalize economically to this low-load regime.**

The project keeps that result unchanged.

## Next hypothesis: workload generalization

The next phase, benchmark-v2 (#115–#119), tests a more precise question:

> **If DQN/PPO are trained across a broader, predeclared workload distribution, does their held-out cost/SLA behavior generalize better?**

Stage A intentionally changes workload diversity while keeping the main simulator/action/observation world fixed.

That lets us study whether training-distribution coverage itself matters.

## Later question: environment generalization

A separate future study (#120) asks whether a single policy could sensibly transfer across different:

- fleet sizes;
- service capacities;
- startup delays;
- control intervals;
- SLA targets;
- initial capacity;
- service characteristics.

That may require a future observation-v2/action-v2.

Those designs are not implemented or selected yet.

## Falsifiable outcomes

The benchmark-v2 hypothesis is weakened if, on fresh untouched TEST data, RL:

- still overprovisions at low load;
- cannot maintain SLA under harder traffic;
- produces worse cost/service trade-offs than simpler baselines;
- becomes unstable or churn-heavy;
- only looks good after test-driven tuning;
- fails on unseen workload domains.

The hypothesis gains support if a frozen generalist RL policy improves meaningful cost/SLA/churn trade-offs on fresh unseen workloads without relying on post-hoc test changes.

Either outcome is useful for learning.

## Experimental implication

ScaleRL compares:

1. static capacity;
2. random policy as a sanity check;
3. threshold / target-tracking control;
4. predictive autoscaling;
5. DQN;
6. PPO.

The important outputs are raw system metrics such as:

- SLA violation;
- p95 latency;
- infrastructure cost;
- queue/backlog;
- dropped/failed work;
- scaling churn/movement.

Reward is secondary.

## What this project is really for

The main goal is to practice:

- defining a model/control problem;
- building baselines;
- training RL agents;
- designing fair experiments;
- preventing train/test leakage;
- tracking model provenance;
- diagnosing failure;
- adding realism gradually instead of pretending the first simulator is complete.

That is the reason RL belongs in ScaleRL.
