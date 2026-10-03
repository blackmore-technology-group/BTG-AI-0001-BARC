"""BTG-AI-0001 v0.26 bounded standardized Gymnasium qualification.

Standard external environments:
- CartPole-v1
- MountainCar-v0
- Acrobot-v1

Uniform, classic PER, and the locked BTG adaptive replay-balance controller use
the same tile-coded linear Q learner, training/evaluation budget, memory limit,
and seeds. No tuning is performed per method or per holdout result.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
import random
import statistics
from typing import Dict, List, Sequence, Tuple

import gymnasium as gym
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVIDENCE = os.path.join(ROOT, "evidence", "external")
os.makedirs(EVIDENCE, exist_ok=True)

CAPACITY = 500
WARMUP = 64
REPLAY_BATCH = 4
GAMMA = 0.97
PER_ALPHA = 0.60
PER_BETA = 0.40

# Locked from v0.21/v0.22. These are not tuned on Gymnasium.
EMA_RATE = 0.05
MIN_PER = 0.15
MAX_PER = 0.40

SEEDS = [935101, 935102, 935103]

ENVIRONMENTS = {
    "CartPole-v1": {
        "episodes": 60,
        "eval_episodes": 10,
        "bins": 8,
        "tilings": 4,
        "alpha": 0.20,
        "epsilon_start": 0.25,
        "epsilon_end": 0.02,
        "clip_low": [-2.4, -3.0, -0.2095, -3.5],
        "clip_high": [2.4, 3.0, 0.2095, 3.5],
    },
    "MountainCar-v0": {
        "episodes": 120,
        "eval_episodes": 10,
        "bins": 10,
        "tilings": 4,
        "alpha": 0.18,
        "epsilon_start": 0.25,
        "epsilon_end": 0.02,
        "clip_low": [-1.2, -0.07],
        "clip_high": [0.6, 0.07],
    },
    "Acrobot-v1": {
        "episodes": 120,
        "eval_episodes": 10,
        "bins": 6,
        "tilings": 4,
        "alpha": 0.16,
        "epsilon_start": 0.25,
        "epsilon_end": 0.02,
        "clip_low": [-1.0, -1.0, -1.0, -1.0, -12.566, -28.274],
        "clip_high": [1.0, 1.0, 1.0, 1.0, 12.566, 28.274],
    },
}


class TileCoder:
    def __init__(self, low, high, bins: int, tilings: int):
        self.low = np.asarray(low, dtype=np.float64)
        self.high = np.asarray(high, dtype=np.float64)
        self.span = np.maximum(1e-9, self.high - self.low)
        self.bins = int(bins)
        self.tilings = int(tilings)

    def encode(self, observation) -> Tuple[Tuple[int, ...], ...]:
        obs = np.asarray(observation, dtype=np.float64)
        scaled = np.clip((obs - self.low) / self.span, 0.0, 0.999999)
        features = []
        for tiling in range(self.tilings):
            offset = (tiling / self.tilings) / self.bins
            coords = np.floor((scaled + offset) * self.bins).astype(int)
            coords = np.clip(coords, 0, self.bins)
            features.append((tiling, *[int(x) for x in coords]))
        return tuple(features)

    def signature(self, observation) -> Tuple[int, ...]:
        obs = np.asarray(observation, dtype=np.float64)
        scaled = np.clip((obs - self.low) / self.span, 0.0, 0.999999)
        coords = np.floor(scaled * self.bins).astype(int)
        coords = np.clip(coords, 0, self.bins - 1)
        return tuple(int(x) for x in coords)


@dataclass
class Transition:
    key: int
    state: Tuple[Tuple[int, ...], ...]
    state_sig: Tuple[int, ...]
    action: int
    reward: float
    next_state: Tuple[Tuple[int, ...], ...]
    next_sig: Tuple[int, ...]
    terminated: bool
    truncated: bool
    risk: float
    novelty: float


class LinearTileQ:
    def __init__(self, actions: int, alpha: float, tilings: int, seed: int):
        self.actions = int(actions)
        self.alpha = float(alpha)
        self.tilings = int(tilings)
        self.rng = random.Random(seed)
        self.weights: Dict[Tuple[int, Tuple[int, ...]], float] = {}

    def q(self, features, action: int) -> float:
        return sum(self.weights.get((action, feat), 0.0) for feat in features)

    def greedy_action(self, features) -> int:
        values = [self.q(features, a) for a in range(self.actions)]
        peak = max(values)
        choices = [a for a, v in enumerate(values) if abs(v - peak) < 1e-12]
        return self.rng.choice(choices)

    def action(self, features, epsilon: float) -> int:
        if self.rng.random() < epsilon:
            return self.rng.randrange(self.actions)
        return self.greedy_action(features)

    def td_error(self, t: Transition) -> float:
        current = self.q(t.state, t.action)
        future = 0.0 if t.terminated else max(
            self.q(t.next_state, a) for a in range(self.actions)
        )
        target = t.reward + GAMMA * future
        return target - current

    def update(self, t: Transition, importance: float = 1.0) -> Tuple[float, float]:
        before = abs(self.td_error(t))
        td = self.td_error(t)
        step = (self.alpha / self.tilings) * max(0.0, min(1.0, importance)) * td
        for feat in t.state:
            key = (t.action, feat)
            self.weights[key] = self.weights.get(key, 0.0) + step
        after = abs(self.td_error(t))
        return before, after


class UniformReplay:
    def __init__(self, seed: int):
        self.rng = random.Random(seed)
        self.items: List[Transition] = []

    def __len__(self):
        return len(self.items)

    def add(self, transition: Transition, priority: float):
        self.items.append(transition)
        if len(self.items) > CAPACITY:
            self.items.pop(0)

    def sample(self, batch: int):
        if not self.items:
            return []
        k = min(batch, len(self.items))
        return [(t, 1.0, None) for t in self.rng.sample(self.items, k)]

    def outcome(self, token, after: float):
        return

    def controller_state(self):
        return None


class PERReplay:
    def __init__(self, seed: int):
        self.rng = random.Random(seed)
        self.items: List[Transition] = []
        self.priorities: List[float] = []

    def __len__(self):
        return len(self.items)

    def add(self, transition: Transition, priority: float):
        self.items.append(transition)
        self.priorities.append(max(1e-5, float(priority)))
        if len(self.items) > CAPACITY:
            self.items.pop(0)
            self.priorities.pop(0)

    def _probabilities(self):
        masses = [max(1e-8, p) ** PER_ALPHA for p in self.priorities]
        total = sum(masses)
        return [m / total for m in masses]

    def sample(self, batch: int):
        if not self.items:
            return []
        probs = self._probabilities()
        indices = self.rng.choices(range(len(probs)), weights=probs, k=batch)
        raw = [(idx, probs[idx]) for idx in indices]
        n = float(len(self.items))
        weights = [(n * p) ** (-PER_BETA) for _, p in raw]
        peak = max(weights) if weights else 1.0
        return [
            (self.items[idx], weight / peak, idx)
            for (idx, _), weight in zip(raw, weights)
        ]

    def outcome(self, token, after: float):
        if token is not None and 0 <= token < len(self.priorities):
            self.priorities[token] = max(1e-5, float(after))

    def controller_state(self):
        return None


class AdaptiveReplay(PERReplay):
    """Locked BTG replay-balance controller from v0.21/v0.22."""

    def __init__(self, seed: int):
        super().__init__(seed)
        self.surprise_ema = 0.20
        self.novelty_ema = 0.50
        self.risk_ema = 0.05
        self.mix_history: List[float] = []

    def add(self, transition: Transition, priority: float):
        super().add(transition, priority)
        a = EMA_RATE
        surprise = min(1.0, float(priority) / 1.0)
        self.surprise_ema = (1.0 - a) * self.surprise_ema + a * surprise
        self.novelty_ema = (
            (1.0 - a) * self.novelty_ema + a * float(transition.novelty)
        )
        self.risk_ema = (
            (1.0 - a) * self.risk_ema + a * float(transition.risk)
        )

    def per_share(self) -> float:
        surprise = min(1.0, self.surprise_ema / 0.35)
        pressure = 0.70 * surprise + 0.30 * self.novelty_ema
        value = 0.15 + 0.25 * pressure - 0.10 * self.risk_ema
        return max(MIN_PER, min(MAX_PER, value))

    def sample(self, batch: int):
        if not self.items:
            return []
        per_probs = self._probabilities()
        n = len(self.items)
        lam = self.per_share()
        probs = [
            lam * per_probs[i] + (1.0 - lam) / n
            for i in range(n)
        ]
        self.mix_history.append(lam)
        raw = []
        for _ in range(batch):
            needle = self.rng.random()
            running = 0.0
            idx = n - 1
            for i, p in enumerate(probs):
                running += p
                if running >= needle:
                    idx = i
                    break
            raw.append((idx, probs[idx]))
        weights = [(n * p) ** (-PER_BETA) for _, p in raw]
        peak = max(weights) if weights else 1.0
        return [
            (self.items[idx], weight / peak, idx)
            for (idx, _), weight in zip(raw, weights)
        ]

    def controller_state(self):
        if not self.mix_history:
            return None
        return {
            "mean_per_share": statistics.mean(self.mix_history),
            "min_per_share": min(self.mix_history),
            "max_per_share": max(self.mix_history),
            "final_surprise_ema": self.surprise_ema,
            "final_novelty_ema": self.novelty_ema,
            "final_risk_ema": self.risk_ema,
        }


def make_replay(method: str, seed: int):
    if method == "uniform":
        return UniformReplay(seed)
    if method == "per":
        return PERReplay(seed)
    if method == "adaptive":
        return AdaptiveReplay(seed)
    raise ValueError(method)


def risk_metric(env_id: str, obs, terminated: bool, truncated: bool) -> float:
    if env_id == "CartPole-v1":
        x, _, theta, _ = [float(v) for v in obs]
        boundary = max(abs(x) / 2.4, abs(theta) / 0.2095)
        return max(0.0, min(1.0, boundary))
    if truncated and not terminated:
        return 1.0
    return 0.0


def success_metric(env_id: str, terminated: bool, truncated: bool, episode_return: float) -> bool:
    if env_id == "CartPole-v1":
        return episode_return >= 475.0
    return bool(terminated and not truncated)


def epsilon_for(episode: int, total: int, start: float, end: float) -> float:
    progress = episode / max(1, total - 1)
    return start * (1.0 - progress) + end * progress


def evaluate(env_id: str, config, coder: TileCoder, agent: LinearTileQ, seed: int):
    env = gym.make(env_id)
    returns = []
    successes = 0
    lengths = []
    for episode in range(config["eval_episodes"]):
        obs, _ = env.reset(seed=seed + episode * 31)
        total = 0.0
        steps = 0
        terminated = truncated = False
        while not (terminated or truncated):
            features = coder.encode(obs)
            action = agent.greedy_action(features)
            obs, reward, terminated, truncated, _ = env.step(action)
            total += float(reward)
            steps += 1
        returns.append(total)
        lengths.append(steps)
        successes += int(success_metric(env_id, terminated, truncated, total))
    env.close()
    return {
        "mean_return": statistics.mean(returns),
        "std_return": statistics.stdev(returns) if len(returns) > 1 else 0.0,
        "success_rate": successes / len(returns),
        "mean_length": statistics.mean(lengths),
    }


def train_one(env_id: str, method: str, seed: int):
    config = ENVIRONMENTS[env_id]
    env = gym.make(env_id)
    coder = TileCoder(
        config["clip_low"],
        config["clip_high"],
        config["bins"],
        config["tilings"],
    )
    agent = LinearTileQ(
        env.action_space.n,
        config["alpha"],
        config["tilings"],
        seed + 1,
    )
    replay = make_replay(method, seed + 2)
    visits: Dict[Tuple[int, ...], int] = {}
    training_returns = []
    transition_key = 0

    for episode in range(config["episodes"]):
        obs, _ = env.reset(seed=seed + episode * 17)
        total = 0.0
        terminated = truncated = False
        epsilon = epsilon_for(
            episode,
            config["episodes"],
            config["epsilon_start"],
            config["epsilon_end"],
        )

        while not (terminated or truncated):
            state = coder.encode(obs)
            sig = coder.signature(obs)
            visits[sig] = visits.get(sig, 0) + 1
            action = agent.action(state, epsilon)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            next_state = coder.encode(next_obs)
            next_sig = coder.signature(next_obs)
            novelty = min(1.0, 1.0 / math.sqrt(visits[sig]))
            transition_key += 1
            t = Transition(
                key=transition_key,
                state=state,
                state_sig=sig,
                action=action,
                reward=float(reward),
                next_state=next_state,
                next_sig=next_sig,
                terminated=bool(terminated),
                truncated=bool(truncated),
                risk=risk_metric(env_id, next_obs, terminated, truncated),
                novelty=novelty,
            )

            priority = abs(agent.td_error(t))
            replay.add(t, priority)
            agent.update(t, 1.0)

            if len(replay) >= WARMUP and transition_key % 8 == 0:
                for sampled, weight, token in replay.sample(REPLAY_BATCH):
                    _, after = agent.update(sampled, weight)
                    replay.outcome(token, after)

            total += float(reward)
            obs = next_obs

        training_returns.append(total)

    env.close()
    evaluation = evaluate(
        env_id,
        config,
        coder,
        agent,
        seed + 500000,
    )
    tail = training_returns[-50:] if len(training_returns) >= 50 else training_returns
    return {
        "env": env_id,
        "method": method,
        "seed": seed,
        "train_tail_mean_return": statistics.mean(tail),
        "evaluation": evaluation,
        "controller": replay.controller_state(),
        "transitions": transition_key,
    }


def summarize(rows):
    result = {}
    for env_id in ENVIRONMENTS:
        result[env_id] = {}
        for method in ("uniform", "per", "adaptive"):
            subset = [r for r in rows if r["env"] == env_id and r["method"] == method]
            result[env_id][method] = {}
            for field, getter in {
                "eval_return": lambda r: r["evaluation"]["mean_return"],
                "success_rate": lambda r: r["evaluation"]["success_rate"],
                "eval_length": lambda r: r["evaluation"]["mean_length"],
                "train_tail_return": lambda r: r["train_tail_mean_return"],
            }.items():
                values = [float(getter(r)) for r in subset]
                result[env_id][method][field] = {
                    "mean": statistics.mean(values),
                    "std": statistics.stdev(values) if len(values) > 1 else 0.0,
                }
    return result


def main():
    rows = []
    print("BTG-AI-0001 v0.26 BOUNDED STANDARDIZED GYMNASIUM QUALIFICATION", flush=True)
    for env_id in ENVIRONMENTS:
        for seed in SEEDS:
            for method in ("uniform", "per", "adaptive"):
                print(f"run env={env_id} method={method} seed={seed}", flush=True)
                rows.append(train_one(env_id, method, seed))

    summary = summarize(rows)
    controller_states = [
        r["controller"] for r in rows
        if r["method"] == "adaptive" and r["controller"]
    ]
    controller_summary = {}
    for key in (
        "mean_per_share",
        "min_per_share",
        "max_per_share",
        "final_surprise_ema",
        "final_novelty_ema",
        "final_risk_ema",
    ):
        values = [float(x[key]) for x in controller_states]
        controller_summary[key] = statistics.mean(values)

    payload = {
        "version": "0.26",
        "gymnasium_version": gym.__version__,
        "environments": list(ENVIRONMENTS),
        "seeds": SEEDS,
        "capacity": CAPACITY,
        "replay_batch": REPLAY_BATCH,
        "locked_controller": {
            "ema_rate": EMA_RATE,
            "min_per": MIN_PER,
            "max_per": MAX_PER,
        },
        "summary": summary,
        "controller_summary": controller_summary,
        "runs": rows,
        "boundary": (
            "Standard Gymnasium Classic Control environments with a common "
            "tile-coded Q learner. This is external benchmark evidence for the "
            "replay controller, not a state-of-the-art RL claim or robot safety certification."
        ),
    }
    out_path = os.path.join(EVIDENCE, "GYMNASIUM_LOCKED_v0.26.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)

    print("\nSUMMARY")
    for env_id in ENVIRONMENTS:
        print(env_id)
        for method in ("uniform", "per", "adaptive"):
            s = summary[env_id][method]
            print(
                f"  {method:8s} "
                f"return={s['eval_return']['mean']:.3f} "
                f"success={s['success_rate']['mean']:.3f} "
                f"tail={s['train_tail_return']['mean']:.3f}"
            )
    print("controller=" + json.dumps(controller_summary, sort_keys=True))
    print("evidence=" + out_path)


if __name__ == "__main__":
    main()



