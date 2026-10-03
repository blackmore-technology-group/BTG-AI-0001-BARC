import math
import unittest

from btg_barc import BARCConfig, BARCReplayBuffer


class TestBARC(unittest.TestCase):
    def test_capacity_is_hard_bounded(self):
        b = BARCReplayBuffer(BARCConfig(capacity=3), seed=1)
        ids = [b.add(i, priority=0.1) for i in range(5)]
        self.assertEqual(len(b), 3)
        with self.assertRaises(KeyError):
            b.get(ids[0])
        self.assertEqual(b.get(ids[-1]), 4)

    def test_distribution_sums_to_one(self):
        b = BARCReplayBuffer(seed=2)
        for i in range(20):
            b.add(i, priority=0.05 + i / 20, novelty=i / 20)
        probs = b.probabilities()
        self.assertAlmostEqual(sum(probs.values()), 1.0, places=12)
        self.assertTrue(all(p > 0.0 for p in probs.values()))

    def test_uniform_floor_keeps_low_priority_sampleable(self):
        b = BARCReplayBuffer(seed=3)
        low = b.add("low", priority=0.0)
        b.add("high", priority=100.0)
        probs = b.probabilities()
        self.assertGreater(probs[low], 0.0)

    def test_importance_weights_are_normalized(self):
        b = BARCReplayBuffer(seed=4)
        for i in range(40):
            b.add(i, priority=i + 1)
        samples = b.sample(64)
        self.assertEqual(len(samples), 64)
        self.assertTrue(all(0.0 < s.importance_weight <= 1.0 for s in samples))
        self.assertAlmostEqual(max(s.importance_weight for s in samples), 1.0)

    def test_share_respects_locked_bounds(self):
        b = BARCReplayBuffer(seed=5)
        for _ in range(500):
            b.add(None, priority=100.0, novelty=1.0, risk=0.0)
        self.assertLessEqual(b.priority_share(), 0.40)
        b.reset_controller(surprise=0.0, novelty=0.0, risk=1.0)
        self.assertGreaterEqual(b.priority_share(), 0.15)

    def test_surprise_increases_priority_share(self):
        low = BARCReplayBuffer(seed=6)
        low.reset_controller(surprise=0.0, novelty=0.0, risk=0.0)
        low_share = low.priority_share()
        high = BARCReplayBuffer(seed=6)
        high.reset_controller(surprise=1.0, novelty=0.0, risk=0.0)
        self.assertGreater(high.priority_share(), low_share)

    def test_novelty_increases_priority_share(self):
        b = BARCReplayBuffer(seed=7)
        b.reset_controller(surprise=0.1, novelty=0.0, risk=0.0)
        low = b.priority_share()
        b.reset_controller(surprise=0.1, novelty=1.0, risk=0.0)
        self.assertGreater(b.priority_share(), low)

    def test_risk_pressure_reduces_priority_share(self):
        b = BARCReplayBuffer(seed=8)
        b.reset_controller(surprise=0.3, novelty=0.5, risk=0.0)
        no_risk = b.priority_share()
        b.reset_controller(surprise=0.3, novelty=0.5, risk=1.0)
        self.assertLess(b.priority_share(), no_risk)

    def test_priority_update_changes_distribution(self):
        b = BARCReplayBuffer(seed=9)
        a = b.add("a", priority=0.1)
        z = b.add("z", priority=0.1)
        before = b.probabilities()[a]
        b.update_priority(a, 10.0)
        after = b.probabilities()[a]
        self.assertGreater(after, before)
        self.assertLess(b.probabilities()[z], after)

    def test_sampling_is_seed_deterministic(self):
        def draws(seed):
            b = BARCReplayBuffer(seed=seed)
            for i in range(15):
                b.add(i, priority=i + 1, novelty=0.2, risk=0.1)
            return [s.record_id for s in b.sample(30)]
        self.assertEqual(draws(10), draws(10))
        self.assertNotEqual(draws(10), draws(11))

    def test_record_id_is_stable_until_eviction(self):
        b = BARCReplayBuffer(BARCConfig(capacity=2), seed=12)
        rid = b.add({"x": 1}, priority=1.0)
        self.assertEqual(b.get(rid), {"x": 1})
        b.update_priority(rid, 0.3)
        self.assertEqual(b.get(rid), {"x": 1})

    def test_invalid_priority_rejected(self):
        b = BARCReplayBuffer()
        for value in (-1.0, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                b.add(None, priority=value)

    def test_controller_state_is_auditable(self):
        b = BARCReplayBuffer(seed=13)
        for i in range(10):
            b.add(i, priority=0.2, novelty=0.3, risk=0.1)
        b.sample(5)
        state = b.controller_state()
        self.assertEqual(state["sample_calls"], 1)
        self.assertEqual(state["retained_records"], 10)
        self.assertIsNotNone(state["mean_sampled_priority_share"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
