# Qualification evidence

BARC was developed under an evidence-first rule: rejected variants remain part of the engineering history rather than being relabelled as successful results.

## Internal causal qualification

BTG used a custom continual hazard-navigation testbed to study replay behavior under:

- task switching;
- rare adverse events;
- degraded observations;
- retention/forgetting pressure;
- equal-capacity replay buffers;
- identical fixed transition streams.

Important rejected directions included multi-axis-only replay, dedicated mission channels, fixed safety reservations, a bootstrapped risk critic and an empirical action gate.

The surviving design emerged from Uniform/PER mixture experiments and was converted into a closed-loop replay-balance controller.

### Locked internal holdout

On a ten-seed locked internal holdout:

| Method | Task A success | Task B success | A forgetting | A hazards/episode | B hazards/episode |
|---|---:|---:|---:|---:|---:|
| Uniform | 0.853 | 0.810 | 0.067 | 0.337 | 0.723 |
| PER | 0.883 | 0.793 | 0.003 | 0.290 | 0.653 |
| BARC adaptive | 0.887 | 0.790 | 0.000 | 0.280 | 0.657 |

The locked adaptive controller was essentially at parity with PER on that holdout. It did not justify a universal-superiority claim.

## Standard external environment gate

The locked controller was then run, without controller retuning, on Gymnasium Classic Control environments.

Environment source: Gymnasium 1.3.0.

A common bounded tile-coded Q learner was used for Uniform, PER and BARC. The qualification profile used:

- memory capacity: 500;
- PER alpha: 0.60;
- importance beta: 0.40;
- BARC priority-share bounds: 0.15–0.40;
- three fixed seeds per environment;
- four tile-coding tilings;
- replay batch 4;
- replay every eighth transition;
- 60 training episodes for CartPole;
- 120 training episodes for MountainCar and Acrobot;
- 10 evaluation episodes per run.

These budgets were intentionally modest so the test evaluates replay behavior in a common learner. They are not intended as state-of-the-art environment solutions.

### External results

| Environment | Method | Mean eval return | Success rate | Train-tail return |
|---|---|---:|---:|---:|
| CartPole-v1 | Uniform | 118.833 | 0.033 | 61.647 |
| CartPole-v1 | PER | 22.567 | 0.000 | 22.100 |
| CartPole-v1 | BARC | 78.733 | 0.000 | 50.067 |
| MountainCar-v0 | Uniform | -187.433 | 1.000 | -191.407 |
| MountainCar-v0 | PER | -184.767 | 0.500 | -186.820 |
| MountainCar-v0 | BARC | -189.133 | 0.467 | -188.733 |
| Acrobot-v1 | Uniform | -305.133 | 0.933 | -319.193 |
| Acrobot-v1 | PER | -312.667 | 0.967 | -332.527 |
| Acrobot-v1 | BARC | -313.500 | 0.933 | -343.207 |

During this external gate BARC selected a priority share ranging from approximately **0.294 to 0.368**, with mean approximately **0.327**.

## Interpretation

The external evidence supports four claims:

1. the controller operates on environments not designed by BTG;
2. the replay share actually adapts within its locked bounds;
3. exact mixed-distribution importance correction can be maintained while it adapts;
4. BARC is not universally dominant—environment and learner regime matter.

In this bounded qualification BARC materially outperformed PER on CartPole return, but it did not beat the strongest baseline in every MountainCar or Acrobot measure.

That mixed result is part of the release evidence, not something hidden by the project.

## Not established

This evidence does not establish:

- state-of-the-art reinforcement-learning performance;
- universal superiority over PER or uniform replay;
- deep-neural-network performance;
- physical-robot performance;
- safety certification;
- patentability;
- commercial value;
- statistical significance beyond the disclosed seeds/budgets.

## Reproduction

The core tests require only Python. The external Gymnasium benchmark additionally requires Gymnasium and NumPy.

The sealed evidence manifest records the benchmark version, controller constants, environment names and claim boundary.
