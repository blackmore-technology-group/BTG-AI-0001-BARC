# BARC v1.0.0 Algorithm Contract

BARC controls the mixture between broad uniform experience replay and prioritized experience replay (PER).

## State

The controller maintains exponentially weighted moving averages:

- `S_t`: normalized surprise / learning-error pressure;
- `N_t`: normalized novelty;
- `R_t`: normalized risk pressure.

For each newly observed experience and EMA rate `a = 0.05`:

```
S_t = (1-a) S_(t-1) + a * clamp(priority, 0, 1)
N_t = (1-a) N_(t-1) + a * clamp(novelty, 0, 1)
R_t = (1-a) R_(t-1) + a * clamp(risk, 0, 1)
```

## Priority-share controller

The locked v1.0 controller computes:

```
surprise_pressure = min(1, S_t / 0.35)
pressure = 0.70 * surprise_pressure + 0.30 * N_t

lambda_raw = 0.15 + 0.25 * pressure - 0.10 * R_t
lambda = clamp(lambda_raw, 0.15, 0.40)
```

Thus the reference controller always preserves at least 60% broad replay and always permits at least 15% PER pressure.

The negative risk term is deliberately conservative: high caller-reported risk pressure shifts replay toward broader coverage rather than allowing a small set of high-priority transitions to dominate the buffer.

## PER component

For record `i` with nonnegative priority `p_i` and `alpha = 0.60`:

```
P_PER(i) = p_i^alpha / sum_j p_j^alpha
```

## BARC mixture

For `N` retained experiences:

```
P_BARC(i) = lambda * P_PER(i) + (1-lambda) / N
```

This probability is used directly for sampling.

## Importance correction

For `beta = 0.40`:

```
w_i = (N * P_BARC(i))^(-beta)
```

The returned batch weights are normalized by the largest weight in that batch, so every returned importance weight lies in `(0, 1]`.

## Capacity

The reference buffer is FIFO at its capacity boundary. BARC v1.0 controls replay scheduling; it does not claim a novel eviction policy.

## Task/context transitions

`reset_controller()` can be called at a known task or context boundary. It resets only the controller's operating EMAs and replay-share telemetry. Stored experiences and their priorities are not deleted.

## Signal semantics

BARC does not prescribe how callers compute novelty or risk. Both are normalized inputs in `[0,1]`.

A risk value is **not** a safety guarantee or certified hazard probability. It is an operating signal used by the replay scheduler.

## Determinism

Given identical buffer operations, priorities, signals, configuration and random seed, the reference sampler is deterministic.

## Complexity

With `N` retained records and batch size `B`, the reference implementation constructs the current distribution in `O(N)`. Python's weighted sampling then draws the batch from that distribution. Memory is `O(N)`.

Production implementations may use trees/alias structures without changing the normative probability equation.
