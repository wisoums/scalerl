# Why Reinforcement Learning?

ScaleRL uses reinforcement learning as the **primary learning hypothesis**, not as a predetermined winner.

The central claim being tested is that cloud autoscaling is not only a prediction problem. It is also a **sequential decision problem with delayed consequences**: a scaling action changes future capacity, queueing, latency, cost, and the set of actions that will be useful later. Replica startup delay makes this especially important because an action taken now may not affect service capacity until several control intervals later.

## Problem structure

At each control step, an autoscaler observes a system state, chooses a scaling action, and then sees the system evolve.

- **State:** request rate and trend, CPU/load, queue depth, p95 latency, active and pending replicas, recent scaling history, and cost-related signals.
- **Action:** scale down, hold, or scale up.
- **Transition:** workload and infrastructure dynamics determine the next state. Scaling actions can have delayed effects because replicas take time to start or terminate.
- **Reward/objective:** balance latency, SLA violations, infrastructure cost, dropped work, and scaling churn over time.

This maps naturally to a Markov decision process or an approximate partially observed decision process.

## Why not supervised learning alone?

Supervised learning is well suited to predicting a target from labeled examples. For autoscaling, it could predict future request rate, latency, or required capacity.

The difficulty is that there is usually no ground-truth label for the **optimal action** at every state. A dataset may tell us what a previous controller did, but training on those actions would largely imitate that controller rather than discover a better policy.

Supervised forecasting can still be useful inside ScaleRL. A demand forecast can be part of the observation or power a predictive baseline. The distinction is:

- forecasting answers **"what is likely to happen?"**
- control answers **"what should I do now, given what may happen and the future consequences of this action?"**

ScaleRL evaluates both.

## Why not unsupervised learning?

Unsupervised learning is useful for discovering structure without labels: clustering workload regimes, learning representations, or detecting anomalies. Those are potentially useful supporting components, but unsupervised learning does not directly specify a policy that optimizes a sequence of scaling actions against latency/cost/SLA objectives.

It is therefore not the primary controller-learning paradigm for this project.

## Why not a contextual bandit?

A contextual bandit chooses an action from the current context and receives an immediate reward, but it generally assumes the current action does not meaningfully change the future state distribution that must be optimized.

Autoscaling violates that simplification. Scaling up now can create capacity only after a startup delay; scaling down can increase future queueing; repeated scale-up/scale-down decisions can create churn. Because actions affect later states and later rewards, a sequential RL formulation is more appropriate to investigate.

## What about control theory and model predictive control?

Classical feedback control and model predictive control (MPC) are strong alternatives and may outperform RL when the system dynamics are sufficiently known, stable, and modelable. ScaleRL should not frame them as obsolete.

The practical comparison in v1 focuses on strong reactive and predictive autoscaling baselines. A future extension can add an MPC controller if time permits.

RL is most interesting when:

- dynamics are nonlinear or difficult to model exactly;
- workloads are stochastic or non-stationary;
- several competing objectives must be traded off;
- actions have delayed effects;
- a policy must adapt behavior across many operating regimes.

## Why RL is worth evaluating here

The project has the characteristics that make RL scientifically reasonable to test:

1. **Sequential decisions:** scaling is repeated over time rather than performed once.
2. **Delayed effects:** startup and shutdown delays separate action time from capacity impact.
3. **Long-term trade-offs:** saving money now can cause SLA violations later; over-provisioning can protect latency at unnecessary cost.
4. **No direct optimal-action labels:** the simulator provides outcomes and rewards, not a dataset of known optimal actions.
5. **Multi-objective control:** the controller must balance latency, SLA compliance, cost, dropped work, and scaling stability.
6. **Interaction:** actions change the environment the agent will observe next.

These properties motivate RL, but they do not prove RL will be better.

## Falsifiable hypothesis

ScaleRL tests the hypothesis:

> A reinforcement-learning policy can learn scaling behavior that improves the latency/cost/SLA/stability trade-off relative to well-tuned reactive and forecasting-based baselines, particularly under workloads with delayed capacity effects and unpredictable changes.

The hypothesis is weakened or rejected if, on held-out workloads and multiple random seeds, RL:

- does not improve the Pareto trade-off between service quality and cost;
- is consistently less stable than tuned reactive/predictive controllers;
- only wins after excessive hyperparameter tuning on evaluation workloads;
- fails to generalize to traffic patterns outside its training distribution; or
- provides gains too small to justify its complexity.

A result showing that a simpler controller wins is still a valid and useful outcome.

## Experimental implication

The project therefore compares the following controller families under identical simulated workloads:

1. static capacity;
2. random policy as a sanity check;
3. threshold / target-tracking control;
4. predictive autoscaling based on demand forecasting;
5. DQN;
6. PPO.

All controllers are evaluated using the same raw system metrics, held-out workload traces, and multiple seeds. Episodic reward alone is never sufficient evidence that RL is better.
