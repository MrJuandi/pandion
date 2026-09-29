"""Unit tests for geographic PPO training and evaluation."""

import unittest

from nightjar.geographic_family import (
    build_synthetic_cems_fixture,
    generate_geographic_scenario,
)
from nightjar.geographic_ppo import (
    HAS_TORCH,
    load_geographic_split,
    ppo_update_geographic,
    returns_and_advantages,
    rollout_geographic,
    train_geographic_ppo,
)
from nightjar.model import SpatialActorCritic, TACTICAL_ACTION_NAMES

if HAS_TORCH:
    import torch


class TestGeographicPPO(unittest.TestCase):
    def setUp(self):
        if not HAS_TORCH:
            self.skipTest("PyTorch not installed")
        fixture = build_synthetic_cems_fixture("EMSR578", utm_zone=31, rows=25, cols=25)
        self.scenario = generate_geographic_scenario(
            fixture,
            index=0,
            split="development",
            window_rows=15,
            window_cols=15,
            horizon_s=300.0,
        )

    def test_load_geographic_split(self):
        scenarios = load_geographic_split("development")
        self.assertGreater(len(scenarios), 0)
        self.assertIn("grid", scenarios[0])
        self.assertIn("fire", scenarios[0])

    def test_returns_and_advantages(self):
        rewards = [-1.0, -0.5, 2.0]
        values = [-0.8, -0.4, 1.5]
        ret, adv = returns_and_advantages(rewards, values, gamma=0.9)
        self.assertEqual(len(ret), 3)
        self.assertEqual(len(adv), 3)
        self.assertTrue(torch.isfinite(ret).all())
        self.assertTrue(torch.isfinite(adv).all())

    def test_rollout_and_ppo_update(self):
        model = SpatialActorCritic(
            grid_rows=15,
            grid_cols=15,
            action_dim=len(TACTICAL_ACTION_NAMES),
            recurrent=True,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        records, metrics = rollout_geographic(model, self.scenario, deterministic=False)
        self.assertGreater(len(records), 0)
        self.assertEqual(metrics["rejected_orders"], 0)
        self.assertIn("final_burned_ha", metrics)

        loss = ppo_update_geographic(model, optimizer, records)
        self.assertTrue(torch.isfinite(torch.tensor(loss)))

    def test_train_geographic_ppo_single_update(self):
        model, losses = train_geographic_ppo(
            [self.scenario],
            updates=1,
            episodes_per_update=1,
            recurrent=True,
        )
        self.assertEqual(len(losses), 1)
        self.assertTrue(torch.isfinite(torch.tensor(losses[0])))


if __name__ == "__main__":
    unittest.main()
