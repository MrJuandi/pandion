"""Weighted drawing of training fires (D-103)."""
import copy
import random
import unittest
from unittest.mock import patch

from pandion.geographic_ppo import HAS_TORCH

if HAS_TORCH:
    import torch
    from pandion.geographic_ppo import train_geographic_ppo
    from tests.test_vine import _scenario


@unittest.skipUnless(HAS_TORCH, "torch not installed")
class TestScenarioWeights(unittest.TestCase):
    def setUp(self):
        random.seed(0)
        torch.manual_seed(0)
        self.a = _scenario()
        self.b = copy.deepcopy(self.a)
        self.a["scenario_id"], self.b["scenario_id"] = "fire-a", "fire-b"

    def _drawn(self, weights):
        drawn = []
        real = random.choices

        def spy(population, **kwargs):
            picked = real(population, **kwargs)
            drawn.extend(s["scenario_id"] for s in picked)
            return picked
        with patch("pandion.geographic_ppo.random.choices", side_effect=spy):
            train_geographic_ppo([self.a, self.b], updates=2, episodes_per_update=3, input_mode="cells",
                                 action_mode="parametric", scenario_weights=weights)
        return drawn

    def test_zero_weight_fire_is_never_drawn(self):
        drawn = self._drawn({"fire-a": 0.0})
        self.assertEqual(len(drawn), 6)
        self.assertEqual(set(drawn), {"fire-b"})

    def test_without_weights_uniform_draw_is_used(self):
        self.assertEqual(self._drawn(None), [])


if __name__ == "__main__":
    unittest.main()
