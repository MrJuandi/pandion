"""Tests for parameterised focus x aspect actions (D-61)."""

import unittest

import numpy as np

from nightjar.model import HAS_TORCH, ActionMapper
from nightjar.parametric_actions import (
    ASPECTS,
    MAX_SLOTS,
    NUM_ACTIONS,
    ParametricActionMapper,
    fire_slots,
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
    def test_slots_ordered_by_size_and_masks_match(self):
        obs = observation(BIG | SMALL, burned={(0, 0)})
        slots = fire_slots(obs)
        self.assertEqual([len(s) for s in slots], [9, 1])
        masks = np.asarray(slot_masks(obs))
        self.assertEqual(masks.shape, (MAX_SLOTS, 12, 12))
        self.assertEqual(masks[0].sum(), 9)
        self.assertEqual(masks[1][9, 9], 1.0)
        self.assertEqual(masks[2:].sum(), 0)
        self.assertEqual(masks[:, 0, 0].sum(), 0)  # burned cells are not a focus

    def test_mask_enables_existing_slots_only(self):
        mapper = ParametricActionMapper(100.0)
        mask = mapper.compute_mask(observation(BIG | SMALL, water=3000))
        self.assertEqual(len(mask), NUM_ACTIONS)
        self.assertTrue(mask[0] and mask[1])
        self.assertTrue(all(mask[2:2 + 2 * len(ASPECTS)]))
        self.assertFalse(any(mask[2 + 2 * len(ASPECTS):]))
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


if __name__ == "__main__":
    unittest.main()
