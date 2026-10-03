# BARC v1.0.0

BTG-AI-0001 is the first public release of the Blackmore Adaptive Replay Controller.

## Included

- framework-independent adaptive replay buffer;
- bounded Uniform/PER mixture controller;
- exact mixed sampling probability;
- normalized importance-sampling correction;
- auditable controller state;
- deterministic seeded sampling;
- 13 deterministic contract tests;
- exact external Gymnasium qualification harness and raw evidence;
- prior-art and claim-boundary documentation.

## Qualification

The exact public package passed 13/13 local contract tests.

The locked controller was evaluated on CartPole-v1, MountainCar-v0 and Acrobot-v1 without controller retuning. Results are intentionally published even where BARC did not beat the strongest baseline.

## Release boundary

v1.0.0 is a research-grade reference implementation. It is not a robot safety system, not a safety certification, and not a claim of universal or state-of-the-art RL performance.

## ENTITY

This release is eligible for controlled ENTITY v3.4.3 provenance/economic registration. ENTITY records are separate from Apache-2.0 permissions granted by this repository.
