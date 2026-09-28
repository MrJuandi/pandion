import unittest

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from nightjar.ppo import returns_and_advantages, train_smoke


@unittest.skipUnless(HAS_TORCH, "PyTorch is optional")
class PPOTests(unittest.TestCase):
    def test_returns_and_advantages(self):
        returns, advantages = returns_and_advantages([1.0, 2.0], [0.0, 0.0], gamma=1.0)
        self.assertEqual(returns.tolist(), [3.0, 2.0])
        self.assertAlmostEqual(float(advantages.mean()), 0.0, places=6)

    def test_smoke_uses_valid_actions_on_development(self):
        result = train_smoke(updates=1, episodes_per_update=1, seed=3)
        self.assertEqual(result["split"], "development")
        self.assertTrue(result["synthetic_no_physical_claim"])
        self.assertEqual(result["metrics"][0]["rejected_orders"], 0)
        self.assertGreater(result["metrics"][0]["action_count"], 0)


if __name__ == "__main__":
    unittest.main()
