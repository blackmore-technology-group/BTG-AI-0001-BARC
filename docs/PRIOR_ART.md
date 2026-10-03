# Prior-art and claim boundary

BARC is intentionally documented with a narrow technical claim boundary.

## Established foundations

The following concepts predate BTG-AI-0001 and are not claimed as BTG inventions:

- experience replay;
- uniform replay buffers;
- prioritized experience replay (PER);
- TD-error-based replay priority;
- importance-sampling correction for non-uniform replay;
- continual-learning replay;
- stability–plasticity methods;
- risk-aware / safety-aware reinforcement-learning methods;
- adaptive replay-buffer and replay-scheduling research.

A central reference for PER is:

Schaul, T., Quan, J., Antonoglou, I., & Silver, D. (2015). *Prioritized Experience Replay*. arXiv:1511.05952.

Research after PER has explored many ways of adapting replay, including uncertainty, task context, age, offline/online mixture, learned replay policies and continual-learning replay.

## BTG contribution

BTG-AI-0001 is the specific **Blackmore Adaptive Replay Controller (BARC)** architecture and implementation released in this repository:

1. a continuously controlled mixture of uniform replay and PER;
2. a hard lower and upper bound on prioritization share;
3. a locked controller driven by exponentially smoothed surprise, novelty and caller-supplied risk pressure;
4. exact probability accounting for the mixed distribution;
5. normalized importance correction derived from that exact mixed probability;
6. exposed controller telemetry for auditability;
7. the associated engineering, negative-result history, holdout qualification and external benchmark evidence.

This repository does **not** claim ownership of PER, experience replay, adaptive replay as a broad concept, or reinforcement learning.

## Performance claim boundary

The published evidence shows that BARC operates reproducibly and adapts its replay mixture. Results vary by environment.

BTG does not claim that BARC universally outperforms uniform replay, PER, modern deep-RL replay methods, or every continual-learning method.

## IP and licence boundary

Apache-2.0 governs the source released here. ENTITY provenance/economic registration is separate from the open-source licence and does not retract permissions granted by Apache-2.0.

Third-party concepts, publications, Gymnasium environments and implementations retain their own provenance and applicable licences.
