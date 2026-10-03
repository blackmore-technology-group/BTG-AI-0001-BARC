"""Blackmore Adaptive Replay Controller (BARC).

Copyright 2026 Blackmore Technology Group Limited.
Licensed under the Apache License, Version 2.0.

BARC blends uniform and prioritized experience replay with an adaptive,
bounded prioritization share. The controller is framework-independent.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Any, Dict, List, Optional


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


@dataclass(frozen=True)
class BARCConfig:
    """Configuration for the locked BARC v1.0 controller."""

    capacity: int = 10_000
    priority_alpha: float = 0.60
    importance_beta: float = 0.40
    ema_rate: float = 0.05
    min_priority_share: float = 0.15
    max_priority_share: float = 0.40
    initial_surprise_ema: float = 0.20
    initial_novelty_ema: float = 0.50
    initial_risk_ema: float = 0.05
    probability_epsilon: float = 1e-12

    def __post_init__(self) -> None:
        if self.capacity < 1:
            raise ValueError("capacity must be >= 1")
        if not 0.0 <= self.priority_alpha <= 1.0:
            raise ValueError("priority_alpha must be in [0,1]")
        if not 0.0 <= self.importance_beta <= 1.0:
            raise ValueError("importance_beta must be in [0,1]")
        if not 0.0 < self.ema_rate <= 1.0:
            raise ValueError("ema_rate must be in (0,1]")
        if not 0.0 <= self.min_priority_share <= self.max_priority_share <= 1.0:
            raise ValueError("priority share bounds must satisfy 0 <= min <= max <= 1")
        for name in (
            "initial_surprise_ema",
            "initial_novelty_ema",
            "initial_risk_ema",
        ):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must be in [0,1]")
        if self.probability_epsilon <= 0.0:
            raise ValueError("probability_epsilon must be positive")


@dataclass
class _Record:
    record_id: int
    payload: Any
    priority: float
    novelty: float
    risk: float


@dataclass(frozen=True)
class BARCSample:
    """One replay sample with an auditable sampling probability."""

    record_id: int
    payload: Any
    probability: float
    importance_weight: float
    priority_share: float


class BARCReplayBuffer:
    """Bounded adaptive mixture of uniform replay and PER.

    For record i:

        P(i) = lambda * P_PER(i) + (1 - lambda) / N

    lambda is selected by the locked BARC controller from exponentially
    smoothed surprise, novelty and risk-pressure signals.

    Returned importance weights are normalized within the sampled batch:

        w(i) ∝ (N * P(i)) ** (-beta)
    """

    def __init__(
        self,
        config: Optional[BARCConfig] = None,
        *,
        seed: int = 0,
    ) -> None:
        self.config = config or BARCConfig()
        self._rng = random.Random(seed)
        self._records: List[_Record] = []
        self._by_id: Dict[int, _Record] = {}
        self._next_id = 1
        self.reset_controller()

    def __len__(self) -> int:
        return len(self._records)

    def reset_controller(
        self,
        *,
        surprise: Optional[float] = None,
        novelty: Optional[float] = None,
        risk: Optional[float] = None,
    ) -> None:
        """Reset operating state, e.g. at a known task/context boundary."""

        c = self.config
        self._surprise_ema = _clamp01(
            c.initial_surprise_ema if surprise is None else surprise
        )
        self._novelty_ema = _clamp01(
            c.initial_novelty_ema if novelty is None else novelty
        )
        self._risk_ema = _clamp01(
            c.initial_risk_ema if risk is None else risk
        )
        self._sample_calls = 0
        self._share_sum = 0.0
        self._share_min = 1.0
        self._share_max = 0.0

    def _observe(self, priority: float, novelty: float, risk: float) -> None:
        a = self.config.ema_rate
        # Priority is a learner-defined nonnegative quantity such as |TD error|.
        # v1.0 maps values >=1 to maximal normalized surprise.
        surprise = _clamp01(priority)
        self._surprise_ema = (1.0 - a) * self._surprise_ema + a * surprise
        self._novelty_ema = (
            (1.0 - a) * self._novelty_ema + a * _clamp01(novelty)
        )
        self._risk_ema = (
            (1.0 - a) * self._risk_ema + a * _clamp01(risk)
        )

    def priority_share(self) -> float:
        """Return the current PER share selected by the locked controller."""

        surprise_pressure = min(1.0, self._surprise_ema / 0.35)
        pressure = 0.70 * surprise_pressure + 0.30 * self._novelty_ema
        raw = 0.15 + 0.25 * pressure - 0.10 * self._risk_ema
        return max(
            self.config.min_priority_share,
            min(self.config.max_priority_share, raw),
        )

    def add(
        self,
        payload: Any,
        *,
        priority: float,
        novelty: float = 0.0,
        risk: float = 0.0,
    ) -> int:
        """Add one experience and return its stable in-buffer record id."""

        p = float(priority)
        if not math.isfinite(p) or p < 0.0:
            raise ValueError("priority must be a finite nonnegative number")
        n = _clamp01(novelty)
        r = _clamp01(risk)

        record = _Record(
            record_id=self._next_id,
            payload=payload,
            priority=max(self.config.probability_epsilon, p),
            novelty=n,
            risk=r,
        )
        self._next_id += 1
        self._records.append(record)
        self._by_id[record.record_id] = record

        while len(self._records) > self.config.capacity:
            victim = self._records.pop(0)
            self._by_id.pop(victim.record_id, None)

        self._observe(p, n, r)
        return record.record_id

    def update_priority(self, record_id: int, priority: float) -> None:
        """Update learning priority after replay without double-counting signals."""

        p = float(priority)
        if not math.isfinite(p) or p < 0.0:
            raise ValueError("priority must be a finite nonnegative number")
        try:
            record = self._by_id[int(record_id)]
        except KeyError as exc:
            raise KeyError("record_id is not currently retained") from exc
        record.priority = max(self.config.probability_epsilon, p)

    def get(self, record_id: int) -> Any:
        try:
            return self._by_id[int(record_id)].payload
        except KeyError as exc:
            raise KeyError("record_id is not currently retained") from exc

    def _distribution(self) -> tuple[List[float], float]:
        if not self._records:
            return [], self.priority_share()

        alpha = self.config.priority_alpha
        masses = [
            max(self.config.probability_epsilon, rec.priority) ** alpha
            for rec in self._records
        ]
        total = sum(masses)
        per = [m / total for m in masses]
        count = len(self._records)
        share = self.priority_share()
        probabilities = [
            share * p + (1.0 - share) / count
            for p in per
        ]
        # Floating arithmetic can drift by a few ulps. Normalize once so the
        # probabilities used for both selection and evidence sum exactly to 1.
        norm = sum(probabilities)
        return [p / norm for p in probabilities], share

    def probabilities(self) -> Dict[int, float]:
        """Return the current exact sampling distribution by record id."""

        probs, _ = self._distribution()
        return {
            record.record_id: probability
            for record, probability in zip(self._records, probs)
        }

    def sample(self, batch_size: int) -> List[BARCSample]:
        """Sample with replacement and return exact probability + IS weight."""

        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if not self._records:
            return []

        probabilities, share = self._distribution()
        indices = self._rng.choices(
            range(len(self._records)),
            weights=probabilities,
            k=int(batch_size),
        )

        count = float(len(self._records))
        beta = self.config.importance_beta
        raw_weights = [
            (count * probabilities[index]) ** (-beta)
            for index in indices
        ]
        peak = max(raw_weights) if raw_weights else 1.0

        self._sample_calls += 1
        self._share_sum += share
        self._share_min = min(self._share_min, share)
        self._share_max = max(self._share_max, share)

        return [
            BARCSample(
                record_id=self._records[index].record_id,
                payload=self._records[index].payload,
                probability=probabilities[index],
                importance_weight=weight / peak,
                priority_share=share,
            )
            for index, weight in zip(indices, raw_weights)
        ]

    def controller_state(self) -> Dict[str, float | int | None]:
        """Expose the auditable operating state of the controller."""

        return {
            "surprise_ema": self._surprise_ema,
            "novelty_ema": self._novelty_ema,
            "risk_ema": self._risk_ema,
            "current_priority_share": self.priority_share(),
            "sample_calls": self._sample_calls,
            "mean_sampled_priority_share": (
                self._share_sum / self._sample_calls
                if self._sample_calls else None
            ),
            "min_sampled_priority_share": (
                self._share_min if self._sample_calls else None
            ),
            "max_sampled_priority_share": (
                self._share_max if self._sample_calls else None
            ),
            "retained_records": len(self._records),
            "capacity": self.config.capacity,
        }
