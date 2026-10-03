# Blackmore Adaptive Replay Controller (BARC)

**BTG Algorithm ID:** BTG-AI-0001  
**Release:** v1.0.0  
**Developer:** Blackmore Technology Group Limited (BTG), Canada  
**Licence:** Apache-2.0  
**Status:** Open-source research-grade reference implementation

BARC is a framework-independent replay controller for reinforcement learning and continual learning. It dynamically blends broad **uniform experience replay** with **prioritized experience replay (PER)** instead of forcing a fixed sampling policy for the entire life of a learner.

The design target is the stability–plasticity problem: a learner needs enough prioritization to revisit high-learning-value experiences, but excessive prioritization can reduce coverage, amplify stale/high-error transitions, and contribute to interference or forgetting.

## Blackmore technology signature

BARC observes three normalized operating signals:

- **surprise** — current learning error / TD-error pressure;
- **novelty** — how unfamiliar the incoming experience is;
- **risk pressure** — caller-supplied adverse-outcome significance.

It maintains exponentially weighted operating state and chooses a PER mixture share `λ` inside a bounded range. Sampling is then:

```
P_BARC(i) = λ · P_PER(i) + (1 - λ) · 1/N
```

Every sampled item carries its exact probability and a normalized importance weight:

```
w(i) ∝ (N · P_BARC(i))^(-β)
```

The reference v1.0.0 controller is deliberately bounded: it can never eliminate broad replay and can never become pure PER.

## Why it exists

Classic uniform replay provides coverage. PER focuses computation on high-error transitions. BARC treats the **amount of prioritization itself as a controlled variable**.

This is not a claim that uniform replay or PER is obsolete. In BTG's qualification work, each wins in some regimes. BARC exists to give a system an auditable middle layer that can adapt between those regimes while retaining exact sampling probabilities.

## Evidence

BTG developed the controller through internal continual-learning/hazard-navigation tests, fixed-stream causal comparisons, ablations, and holdouts. The final internal locked holdout showed the adaptive controller operating at roughly 15–30% PER share and reaching near-PER task performance with improved retention/safety metrics in that custom test.

The locked controller was then run without controller retuning on standard Gymnasium Classic Control environments:

| Environment | Uniform eval return | PER eval return | BARC eval return |
|---|---:|---:|---:|
| CartPole-v1 | 118.833 | 22.567 | 78.733 |
| MountainCar-v0 | -187.433 | -184.767 | -189.133 |
| Acrobot-v1 | -305.133 | -312.667 | -313.500 |

BARC's mean PER share in that external gate was about **32.7%**, ranging from about **29.4% to 36.8%**.

These results establish reproducible operation across external benchmark environments. They **do not establish universal superiority**. The bounded tile-coded learner used for this qualification also is not a state-of-the-art deep-RL benchmark. Full evidence and limitations are in `docs/BENCHMARKS.md`.

Gymnasium documents CartPole-v1, MountainCar-v0 and Acrobot-v1 as standard Classic Control environments. The original PER work is Schaul et al., *Prioritized Experience Replay* (2015).

## Install

```bash
pip install -e .
```

BARC's core has no third-party runtime dependency.

## Minimal use

```python
from btg_barc import BARCReplayBuffer

buffer = BARCReplayBuffer(capacity=10_000, seed=42)

record_id = buffer.add(
    payload={"state": "...", "action": 1},
    priority=0.72,   # e.g. absolute TD error
    novelty=0.40,    # normalized [0,1]
    risk=0.10,       # normalized [0,1]
)

batch = buffer.sample(32)

for sample in batch:
    transition = sample.payload
    importance_weight = sample.importance_weight
    # learner_update(transition, importance_weight)
    new_td_error = 0.31
    buffer.update_priority(sample.record_id, new_td_error)
```

Inspect the live controller:

```python
print(buffer.controller_state())
```

## Integration contract

BARC does **not** prescribe:

- a neural-network architecture;
- Q-learning vs actor–critic;
- a particular robot;
- how novelty is estimated;
- how risk is defined;
- a reward function;
- a simulator.

The caller supplies the experience payload and normalized novelty/risk signals. BARC owns replay scheduling, exact mixture probability, and importance correction.

## Prior-art boundary

Experience replay, prioritized experience replay, adaptive replay buffers, continual-learning replay, safety/risk-aware replay, and stability–plasticity methods predate this project.

BTG's contribution is the specific BARC controller and reference implementation—not ownership of those underlying research concepts. See `docs/PRIOR_ART.md`.

## Safety boundary

BARC is **not** a robot safety controller, collision-avoidance system, safety certificate, or guarantee of safe behavior. A risk input changes replay scheduling; it does not enforce physical-system constraints.

## ENTITY relationship

The source code is open source. ENTITY lineage/economic records are separate provenance and rights objects; registering BARC in ENTITY does not revoke Apache-2.0 permissions and does not imply ownership of third-party prior art.

## Repository map

- `src/btg_barc/` — reusable controller and replay buffer
- `tests/` — deterministic contract tests
- `docs/ALGORITHM.md` — equations and controller state
- `docs/BENCHMARKS.md` — qualification evidence and limitations
- `docs/PRIOR_ART.md` — claim boundary
- `evidence/` — sealed qualification summaries/manifests
- `examples/` — minimal integration

## Citation

See `CITATION.cff`.

Copyright © 2026 Blackmore Technology Group Limited.
