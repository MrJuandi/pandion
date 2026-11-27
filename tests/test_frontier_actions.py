"""Any-cell frontier actions (D-104)."""
import math
import random
import unittest

from pandion.model import HAS_TORCH, ActionMapper
from pandion.parametric_actions import MAX_FRONTIER, FrontierActionMapper, ParametricActionMapper
from pandion.policies import candidate_cells
from tests.test_parametric_actions import BIG, observation

if HAS_TORCH:
    import torch
    from pandion.geographic_ppo import train_geographic_ppo
    from pandion.image_model import ImageActorCritic
    from tests.test_vine import _scenario


class FrontierMapperTests(unittest.TestCase):
    def test_every_candidate_is_an_action_in_row_major_order(self):
        obs = observation(BIG | {(9, 9)})
        mapper = FrontierActionMapper(100.0)
        cands = sorted(tuple(c) for c in candidate_cells(obs))
        self.assertEqual(mapper.num_actions, 2 + MAX_FRONTIER)
        self.assertEqual(list(mapper.targets(obs).values()), cands)
        self.assertEqual(list(mapper.targets(obs)), list(range(2, 2 + len(cands))))
        mask = mapper.compute_mask(obs)
        self.assertEqual(sum(mask[2:]), len(cands))
        self.assertTrue(mask[0])

    def test_it_reaches_cells_the_four_aspects_cannot(self):
        obs = observation(BIG)
        four = set(ParametricActionMapper(100.0).targets(obs).values())
        anyc = set(FrontierActionMapper(100.0).targets(obs).values())
        self.assertTrue(four < anyc)
        self.assertEqual(anyc, {tuple(c) for c in candidate_cells(obs)})

    def test_orders_drop_on_the_cell_and_the_env_mapper_matches(self):
        obs = observation(BIG)
        mapper = ActionMapper(12, 12, 100.0, action_mode="frontier")
        r, c = FrontierActionMapper(100.0).targets(obs)[5]
        self.assertEqual(mapper.to_order(5, obs), {"type": "DROP", "target_m": [c * 100.0, r * 100.0]})
        self.assertEqual(mapper.compute_mask(obs), FrontierActionMapper(100.0).compute_mask(obs))
        with self.assertRaises(ValueError):
            mapper.to_order(2 + MAX_FRONTIER - 1, obs)

    def test_outward_step_points_away_from_the_fire_downwind_first(self):
        obs = observation({(5, 5)}, wind=(2.0, 0.0))  # wind towards +col (east)
        details = FrontierActionMapper(100.0).target_details(obs)
        by_cell = {(r, c): (orow, ocol) for r, c, orow, ocol, _ in details.values()}
        self.assertEqual(by_cell[(5, 6)], (0, 1))
        self.assertEqual(by_cell[(5, 4)], (0, -1))
        self.assertTrue(all(d == 0 for *_, d in details.values()))

    def test_only_distance_zero(self):
        with self.assertRaises(ValueError):
            FrontierActionMapper(100.0, (0, 1))

    def test_no_targets_while_flying_or_empty(self):
        obs = observation(BIG, phase="flying")
        mask = FrontierActionMapper(100.0).compute_mask(obs)
        self.assertEqual(mask[:2], [True, False])
        self.assertFalse(any(mask[2:]))


@unittest.skipUnless(HAS_TORCH, "torch not installed")
class FrontierModelTests(unittest.TestCase):
    def _model(self):
        torch.manual_seed(0)
        return ImageActorCritic(cell_channels=17, vector_dim=15, px_per_cell=1, grid_rows=6, grid_cols=6, action_mode="frontier",
                                use_image=False, action_feature_dim=22, action_skip=True).eval()

    def test_needs_target_features(self):
        with self.assertRaisesRegex(ValueError, "target features"):
            ImageActorCritic(cell_channels=17, vector_dim=15, px_per_cell=1, grid_rows=6, grid_cols=6, action_mode="frontier", use_image=False)

    def test_starts_uniform_over_valid_actions(self):
        model = self._model()
        self.assertEqual(model.action_dim, 2 + MAX_FRONTIER)
        actions = torch.zeros(1, MAX_FRONTIER, 23)
        actions[0, :7, 0] = torch.arange(7).float()
        actions[0, :7, 1] = 1.0
        actions[0, :7, 2:] = torch.rand(7, 21)
        mask = torch.zeros(1, model.action_dim, dtype=torch.bool)
        mask[0, :9] = True
        with torch.no_grad():
            logits, values, _ = model(torch.rand(1, 17, 6, 6), torch.rand(1, 15), actions, state=None)
            p = torch.softmax(model._masked(logits, mask), -1)
        self.assertEqual(logits.shape, (1, 2 + MAX_FRONTIER))
        torch.testing.assert_close(p[0, :9], torch.full((9,), 1 / 9))
        self.assertEqual(float(p[0, 9:].sum()), 0.0)

    def test_trains_with_branched_samples(self):
        random.seed(0)
        torch.manual_seed(0)
        _, losses = train_geographic_ppo([_scenario()], updates=1, episodes_per_update=1, input_mode="cells",
                                         action_mode="frontier", action_features=True, vine_states=1, vine_actions=2)
        self.assertTrue(math.isfinite(losses[0]))


if __name__ == "__main__":
    unittest.main()
