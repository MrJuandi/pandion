"""Tests for parameterised focus x aspect actions (D-61)."""

import unittest

import numpy as np

from nightjar.model import HAS_TORCH, ActionMapper
from nightjar.parametric_actions import (
    ASPECTS,
    MAX_SLOTS,
    NUM_ACTIONS,
    ParametricActionMapper,
    action_names,
    fire_slots,
    num_actions,
    slot_masks,
)
from nightjar.policies import candidate_cells


def observation(burning, burned=(), rows=12, cols=12, wind=(2.0, 0.0), water=6137, phase="idle", pos=(0.0, 0.0)):
    cells = []
    for r in range(rows):
        line = []
        for c in range(cols):
            if (r, c) in burning:
                line.append({"fire_detected": True, "fire_state": "burning"})
            elif (r, c) in burned:
                line.append({"fire_detected": True, "fire_state": "burned"})
            else:
                line.append({"fire_detected": False, "fire_state": "unburned"})
        cells.append(line)
    return {"fire": {"cells": cells}, "fuel_model": [["102"] * cols for _ in range(rows)],
            "known_treated_cells": [], "midflame_wind_to_m_s": list(wind),
            "aircraft": {"position_m": list(pos), "water_l": water, "phase": phase}}


BIG = {(r, c) for r in range(2, 5) for c in range(2, 5)}
SMALL = {(9, 9)}


class ParametricActionTests(unittest.TestCase):
    def test_slots_are_fire_fronts_ordered_by_length(self):
        obs = observation(BIG | SMALL, burned={(0, 0)})
        slots = fire_slots(obs)
        # 3x3 block: 8 border cells (the centre has no attackable neighbour); the
        # isolated burning cell and the burned corner cell are fronts of length 1.
        self.assertEqual([len(s) for s in slots], [8, 1, 1])
        self.assertNotIn((3, 3), slots[0])
        masks = np.asarray(slot_masks(obs))
        self.assertEqual(masks.shape, (MAX_SLOTS, 12, 12))
        self.assertEqual(masks[0].sum(), 8)
        self.assertEqual(masks[:, 0, 0].sum(), 1)  # a burned cell next to fuel is still a front
        self.assertEqual(masks[3].sum(), 0)

    def test_every_candidate_borders_some_front(self):
        obs = observation(BIG | SMALL, burned={(0, 0), (6, 6), (6, 7)})
        fronts = {cell for slot in fire_slots(obs) for cell in slot}
        for r, c in candidate_cells(obs):
            self.assertTrue(any((r + dr, c + dc) in fronts for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))))

    def test_mask_enables_existing_slots_only(self):
        mapper = ParametricActionMapper(100.0)
        mask = mapper.compute_mask(observation(BIG | SMALL, water=3000))
        self.assertEqual(len(mask), NUM_ACTIONS)
        self.assertTrue(mask[0] and mask[1])
        self.assertTrue(all(mask[2:2 + 2 * len(ASPECTS)]))
        self.assertFalse(any(mask[2 + 2 * len(ASPECTS):]))  # two fronts, two valid slots
        self.assertEqual(mapper.compute_mask(observation(BIG, phase="transit")), [True] + [False] * (NUM_ACTIONS - 1))
        self.assertEqual(mapper.compute_mask(observation(BIG, water=0))[1], True)

    def test_aspects_pick_wind_relative_perimeter_cells(self):
        mapper = ParametricActionMapper(100.0)
        obs = observation(BIG, wind=(2.0, 0.0))  # wind toward +x (east, increasing column)
        candidates = {tuple(c) for c in candidate_cells(obs)}
        head = mapper.target_cell(2 + 0, obs)
        rear = mapper.target_cell(2 + 3, obs)
        self.assertIn(head, candidates)
        self.assertIn(rear, candidates)
        self.assertEqual(head[1], 5)  # east of the cluster
        self.assertEqual(rear[1], 1)  # west of the cluster
        left = mapper.target_cell(2 + 1, obs)
        right = mapper.target_cell(2 + 2, obs)
        self.assertNotEqual(left[0], right[0])  # opposite flanks
        with self.assertRaises(ValueError):
            mapper.target_cell(2 + len(ASPECTS), obs)  # slot 1 absent

    def test_action_mapper_delegates_parametric_mode(self):
        mapper = ActionMapper(12, 12, 100.0, action_mode="parametric")
        obs = observation(BIG)
        self.assertEqual(mapper.num_actions, NUM_ACTIONS)
        order = mapper.to_order(2, obs)
        self.assertEqual(order["type"], "DROP")
        self.assertIn([int(order["target_m"][1] / 100), int(order["target_m"][0] / 100)], candidate_cells(obs))

    def test_zero_distance_is_the_d61_action_set(self):
        obs = observation(BIG | SMALL, water=3000)
        plain, zero = ParametricActionMapper(100.0), ParametricActionMapper(100.0, (0,))
        self.assertEqual(zero.num_actions, NUM_ACTIONS)
        self.assertEqual(zero.compute_mask(obs), plain.compute_mask(obs))
        for index in range(2, NUM_ACTIONS):
            if plain.compute_mask(obs)[index]:
                self.assertEqual(zero.target_cell(index, obs), plain.target_cell(index, obs))

    def test_drop_distances_step_away_from_the_fire(self):
        # D-79: index 2 + (4k + a) * D + j aims distances[j] cells beyond the front cell.
        mapper = ParametricActionMapper(100.0, (0, 1, 2))
        self.assertEqual(mapper.num_actions, num_actions((0, 1, 2)))
        self.assertEqual(mapper.num_actions, 2 + MAX_SLOTS * len(ASPECTS) * 3)
        self.assertEqual(action_names((0, 1, 2))[2:5], ("ATTACK_F0_HEAD_D0", "ATTACK_F0_HEAD_D1", "ATTACK_F0_HEAD_D2"))
        obs = observation(BIG, wind=(2.0, 0.0))  # cluster columns 2-4, head to the east
        head = [mapper.target_cell(2 + j, obs) for j in range(3)]
        self.assertEqual([c for _, c in head], [5, 6, 7])
        self.assertEqual(len({r for r, _ in head}), 1)
        rear = [mapper.target_cell(2 + 3 * 3 + j, obs) for j in range(2)]
        self.assertEqual(rear[0][1], 1)
        self.assertEqual(rear[1][1], 0)
        # Two cells west of column 1 is off the grid: that action is masked.
        mask = mapper.compute_mask(obs)
        self.assertFalse(mask[2 + 3 * 3 + 2])
        with self.assertRaises(ValueError):
            mapper.target_cell(2 + 3 * 3 + 2, obs)

    def test_target_details_agree_with_targets(self):
        mapper = ParametricActionMapper(100.0, (0, 1, 2))
        obs = observation(BIG | SMALL, burned={(0, 0)}, wind=(2.0, 1.0))
        details = mapper.target_details(obs)
        self.assertEqual(mapper.targets(obs), {i: (r, c) for i, (r, c, *_) in details.items()})
        checked = 0
        for i, (r, c, orow, ocol, d) in details.items():
            if d == 0:
                continue
            j = (i - 2) % len(mapper.distances)  # distance index; its d=0 sibling is i - j
            base = details.get(i - j)
            if base is None:
                continue
            checked += 1
            self.assertEqual((r, c), (base[0] + d * orow, base[1] + d * ocol))
            self.assertEqual((orow, ocol), (base[2], base[3]))
        self.assertGreater(checked, 0)

    def test_offset_targets_must_be_untreated_unburned_fuel(self):
        mapper = ParametricActionMapper(100.0, (0, 1))
        obs = observation(BIG, wind=(2.0, 0.0))
        r, c = mapper.target_cell(2 + 1, obs)  # head, one cell beyond the front
        obs["known_treated_cells"] = [[r, c]]
        self.assertFalse(mapper.compute_mask(obs)[2 + 1])
        obs = observation(BIG, wind=(2.0, 0.0))
        obs["fuel_model"][r][c] = "0"
        self.assertFalse(mapper.compute_mask(obs)[2 + 1])
        self.assertTrue(mapper.compute_mask(obs)[2 + 0])

    def test_environment_executes_a_drop_beyond_the_front(self):
        import sys, os
        sys.path.insert(0, os.path.dirname(__file__))
        from test_geographic_integrated import growing_fire_scenario
        from nightjar.geographic_training_env import GeographicTrainingEnv
        env = GeographicTrainingEnv(growing_fire_scenario(), action_mode="parametric", drop_distances=(0, 1, 2))
        obs = env.reset()
        mapper = env.action_mapper()
        while not any(mapper.compute_mask(obs)[2:]):
            obs, _, done, _ = env.step_discrete(0)
            self.assertFalse(done)
        far = next(i for i, ok in enumerate(mapper.compute_mask(obs)) if ok and i >= 2 and (i - 2) % 3 == 2)
        target = mapper._tactical.target_cell(far, obs)
        self.assertNotIn(list(target), candidate_cells(obs))  # two cells out: not fire-adjacent
        obs, _, done, _ = env.step_discrete(far)
        while not done:  # hold, or refill when empty, until the horizon
            obs, _, done, _ = env.step_discrete(mapper.compute_mask(obs).index(True))
        drops = [(round(d["target_local_m"][1] / 100), round(d["target_local_m"][0] / 100)) for d in env.replay["drops"]]
        self.assertIn(target, drops)

    @unittest.skipUnless(HAS_TORCH, "PyTorch not installed")
    def test_head_outputs_one_logit_per_aspect_and_distance(self):
        import torch
        from nightjar.image_model import ImageActorCritic
        model = ImageActorCritic(cell_channels=2 + MAX_SLOTS, vector_dim=3, px_per_cell=1, grid_rows=6,
                                 grid_cols=6, action_mode="parametric", use_image=False, recurrent=False,
                                 width=8, hidden_dim=16, drop_distances=(0, 1, 2))
        logits, _ = model(torch.zeros(1, 2 + MAX_SLOTS, 6, 6), torch.zeros(1, 3))
        self.assertEqual(logits.shape, (1, num_actions((0, 1, 2))))
        self.assertEqual(model.config["drop_distances"], (0, 1, 2))

    @unittest.skipUnless(HAS_TORCH, "PyTorch not installed")
    def test_slot_logits_follow_slot_masks(self):
        import torch
        from nightjar.image_model import ImageActorCritic
        torch.manual_seed(0)
        model = ImageActorCritic(cell_channels=2 + MAX_SLOTS, vector_dim=3, px_per_cell=1, grid_rows=6,
                                 grid_cols=6, action_mode="parametric", use_image=False, recurrent=False,
                                 width=8, hidden_dim=16)
        cells = torch.zeros(1, 2 + MAX_SLOTS, 6, 6)
        cells[0, 2, 1, 1] = 1.0  # slot 0 mask
        vector = torch.zeros(1, 3)
        logits, _ = model(cells, vector)
        self.assertEqual(logits.shape, (1, NUM_ACTIONS))
        # Empty slots share identical logits (no mask), present slot differs from them.
        empty = logits[0, 2 + len(ASPECTS):].view(MAX_SLOTS - 1, len(ASPECTS))
        self.assertTrue(torch.allclose(empty[0], empty[1]))
        self.assertFalse(torch.allclose(logits[0, 2:2 + len(ASPECTS)], empty[0]))

    @unittest.skipUnless(HAS_TORCH, "PyTorch not installed")
    def test_slot_raw_features_see_slot_averages(self):
        import torch
        from nightjar.image_model import ImageActorCritic
        torch.manual_seed(0)
        model = ImageActorCritic(cell_channels=2 + MAX_SLOTS, vector_dim=3, px_per_cell=1, grid_rows=6,
                                 grid_cols=6, action_mode="parametric", use_image=False, recurrent=False,
                                 width=8, hidden_dim=16, slot_raw_features=True)
        self.assertEqual(model.slot_actor_head[0].in_features, 8 + 16 + 1 + (2 + MAX_SLOTS) + 1)
        cells = torch.zeros(1, 2 + MAX_SLOTS, 6, 6)
        cells[0, 2, 1, 1] = 1.0
        logits, _ = model(cells, torch.zeros(1, 3))
        changed = cells.clone()
        changed[0, 0, 1, 1] = 3.0  # raw channel value inside slot 0 only
        changed_logits, _ = model(changed, torch.zeros(1, 3))
        self.assertFalse(torch.allclose(logits[0, 2:6], changed_logits[0, 2:6]))


if __name__ == "__main__":
    unittest.main()
