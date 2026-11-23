"""Tests for per-action parametric head inputs (D-82)."""

from math import atan2, degrees, hypot
import unittest

import numpy as np

from pandion.action_features import (
    ACTION_FEATURE_NAMES,
    NUM_ACTION_FEATURES,
    ObservedFireBehavior,
    action_feature_array,
)
from pandion.environment import MOISTURE_KEYS
from pandion.model import HAS_TORCH
from pandion.parametric_actions import ParametricActionMapper, num_actions

PX = 4
MOISTURE = {"dead_1h": 0.06, "dead_10h": 0.07, "dead_100h": 0.08,
            "dead_herbaceous": 0.30, "live_herbaceous": 0.60, "live_woody": 0.90}


def observation(burning, burned=(), rows=12, cols=12, size=30.0, wind=(2.0, 0.0), water=6137,
                phase="idle", pos=(0.0, 0.0), code="102", slope=(0.0, 0.0), treated=(), limit=3):
    """Small realtime controller observation: fuel code, terrain slope (m per cell east, north)."""
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
    elevation = [[slope[0] * c + slope[1] * r for c in range(cols)] for r in range(rows)]
    return {"observation_mode": "realtime", "time_s": 600.0, "fire": {"cells": cells},
            "fuel_model": [[code] * cols for _ in range(rows)], "elevation_m": elevation,
            "cell_size_m": size, "midflame_wind_to_m_s": list(wind), "fuel_moisture": dict(MOISTURE),
            "known_treated_cells": [list(cell) for cell in treated], "max_stacked_drops": limit,
            "aircraft": {"position_m": list(pos), "water_l": water, "phase": phase},
            "fleet": [{"index": 0, "position_m": list(pos), "water_l": water, "phase": phase,
                       "busy_remaining_s": 0.0},
                      {"index": 1, "position_m": [rows * size, cols * size], "water_l": 6137,
                       "phase": "idle", "busy_remaining_s": 0.0}],
            "aircraft_index": 0}


def direct_intensity(obs, r, c, dr, dc):
    """Independent recomputation of the simulator's fireline intensity (geographic_integrated)."""
    from pyretechnics import fuel_models, surface_fire
    elevation = obs["elevation_m"]
    rows, cols, size = len(elevation), len(elevation[0]), obs["cell_size_m"]
    west, east = max(c - 1, 0), min(c + 1, cols - 1)
    south, north = max(r - 1, 0), min(r + 1, rows - 1)
    gx = (elevation[r][east] - elevation[r][west]) / ((east - west) * size)
    gy = (elevation[north][c] - elevation[south][c]) / ((north - south) * size)
    w_e, w_n = (float(v) for v in obs["midflame_wind_to_m_s"])
    upwind = degrees(atan2(-w_e, -w_n)) % 360 if (w_e, w_n) != (0.0, 0.0) else 0.0
    moisture = tuple(float(obs["fuel_moisture"][k]) for k in MOISTURE_KEYS)
    model = fuel_models.moisturize(fuel_models.get_fuel_model(int(obs["fuel_model"][r][c])), moisture)
    base = surface_fire.calc_surface_fire_behavior_no_wind_no_slope(model)
    slope = hypot(gx, gy)
    aspect = degrees(atan2(-gx, -gy)) % 360 if slope > 0 else 0.0
    behavior = surface_fire.calc_surface_fire_behavior_max(base, hypot(w_e, w_n) * 60.0, upwind, slope, aspect)
    norm = hypot(dc, dr, gx * dc + gy * dr)
    direction = (dc / norm, dr / norm, (gx * dc + gy * dr) / norm)
    return float(surface_fire.calc_surface_fire_behavior_in_direction(behavior, direction)["fireline_intensity"])


BIG = {(r, c) for r in range(4, 7) for c in range(4, 7)}


class ActionFeatureArrayTests(unittest.TestCase):
    def test_shape_for_default_and_multi_distance_sets(self):
        obs = observation(BIG, wind=(2.0, 0.0))
        behavior = ObservedFireBehavior()
        for distances in ((0,), (0, 1, 2)):
            mapper = ParametricActionMapper(30.0, distances)
            arr = action_feature_array(obs, mapper, behavior)
            self.assertEqual(arr.dtype, np.float32)
            self.assertEqual(arr.shape, (mapper.num_actions - 2, 1 + NUM_ACTION_FEATURES))
            self.assertEqual(mapper.num_actions, num_actions(distances))
        self.assertEqual(NUM_ACTION_FEATURES, len(ACTION_FEATURE_NAMES))

    def test_valid_column_matches_compute_mask_and_targets(self):
        obs = observation(BIG | {(10, 10)}, burned={(0, 0)}, wind=(2.0, 1.0))
        mapper = ParametricActionMapper(30.0, (0, 1, 2))
        arr = action_feature_array(obs, mapper, ObservedFireBehavior())
        mask = np.asarray(mapper.compute_mask(obs)[2:])
        np.testing.assert_array_equal(arr[:, 1] == 1, mask)
        self.assertTrue(mask.any())
        cols = len(obs["fuel_model"][0])
        for index, (r, c) in mapper.targets(obs).items():
            self.assertEqual(arr[index - 2, 0], r * cols + c)
        self.assertTrue(np.all(arr[~mask] == 0))

    def test_aircraft_that_cannot_drop_gives_zero_rows(self):
        mapper = ParametricActionMapper(30.0, (0, 1))
        busy = observation(BIG, phase="transit_to_drop")
        arr = action_feature_array(busy, mapper, ObservedFireBehavior())
        self.assertFalse(arr.any())
        empty = observation(BIG, water=0)
        self.assertFalse(action_feature_array(empty, mapper, ObservedFireBehavior()).any())

    def test_need_is_one_hot_and_stacking_is_reflected(self):
        obs = observation(BIG, wind=(2.0, 0.0), limit=4)
        mapper = ParametricActionMapper(30.0, (0, 1))
        behavior = ObservedFireBehavior()
        arr = action_feature_array(obs, mapper, behavior)
        valid = arr[:, 1] == 1
        self.assertTrue(valid.any())
        onehot = arr[:, 3] + arr[:, 4] + arr[:, 5]
        np.testing.assert_array_equal(onehot[valid], 1.0)
        self.assertTrue(np.all(onehot[~valid] == 0))
        # Pick a valid row with a finite need; stacking enough drops there makes it sufficient.
        finite = [i for i in np.flatnonzero(valid) if arr[i, 5] == 0]
        self.assertTrue(finite)
        i = finite[0]
        index = int(i) + 2
        r, c = mapper.targets(obs)[index]
        need = 1.0 if arr[i, 3] == 1 else 3.0
        stacked = observation(BIG, wind=(2.0, 0.0), limit=4, treated=[(r, c)] * int(need))
        after = action_feature_array(stacked, mapper, behavior)
        self.assertEqual(after[i, 6], 0.0)  # remaining
        self.assertEqual(after[i, 7], 1.0)  # sufficient
        self.assertGreater(after[i, 2], 0.0)  # stacked fraction

    def test_column_names_align_with_features(self):
        self.assertEqual(ACTION_FEATURE_NAMES[-1], "offset")
        obs = observation(BIG, wind=(2.0, 0.0))
        mapper = ParametricActionMapper(30.0, (0, 2))
        arr = action_feature_array(obs, mapper, ObservedFireBehavior())
        for i, (r, c, orow, ocol, d) in mapper.target_details(obs).items():
            row = arr[i - 2]
            self.assertAlmostEqual(row[-1], d / 2.0)
            self.assertAlmostEqual(row[1 + ACTION_FEATURE_NAMES.index("water_distance")],
                                   min(hypot(c * 30.0, r * 30.0) / 20000.0, 1.5), places=6)
            self.assertLessEqual(row[1 + ACTION_FEATURE_NAMES.index("reached_neighbours")], 1.0)


class ObservedFireBehaviorTests(unittest.TestCase):
    def test_intensity_is_non_negative_and_larger_downwind(self):
        obs = observation(BIG, wind=(3.0, 0.0))
        behavior = ObservedFireBehavior()
        downwind = behavior.intensity(obs, 5, 5, 0, 1)
        upwind = behavior.intensity(obs, 5, 5, 0, -1)
        self.assertGreaterEqual(upwind, 0.0)
        self.assertGreater(downwind, upwind)

    def test_nonburnable_cell_has_zero_intensity(self):
        obs = observation(BIG, wind=(3.0, 0.0))
        obs["fuel_model"][5][5] = "0"
        self.assertEqual(ObservedFireBehavior().intensity(obs, 5, 5, 0, 1), 0.0)

    def test_repeated_calls_hit_the_cache_unchanged(self):
        obs = observation(BIG, wind=(3.0, 1.0), slope=(3.0, 2.0))
        behavior = ObservedFireBehavior()
        first = behavior.intensity(obs, 5, 5, 1, 0)
        self.assertEqual(behavior.intensity(obs, 5, 5, 1, 0), first)
        self.assertEqual(behavior.intensity(obs, 5, 5, 1, 0), first)

    def test_missing_moisture_is_an_error(self):
        obs = observation(BIG)
        del obs["fuel_moisture"]
        with self.assertRaises(ValueError):
            ObservedFireBehavior().intensity(obs, 5, 5, 0, 1)

    def test_matches_direct_simulator_computation(self):
        obs = observation(BIG, wind=(2.0, 1.0), slope=(3.0, 2.0))
        behavior = ObservedFireBehavior()
        for (r, c), (dr, dc) in (((5, 5), (0, 1)), ((3, 6), (1, 0)), ((7, 2), (-1, -1))):
            self.assertAlmostEqual(behavior.intensity(obs, r, c, dr, dc),
                                   direct_intensity(obs, r, c, dr, dc), delta=1e-9)

    def test_needed_at_uses_neighbour_fire_intensity(self):
        obs = observation(BIG, wind=(2.0, 0.0))
        behavior = ObservedFireBehavior()
        self.assertEqual(behavior.needed_at(obs, 0, 0), 0.0)  # no burning neighbour
        need = behavior.needed_at(obs, 3, 5)  # unburned cell north of the block edge
        self.assertIn(need, (1.0, 3.0, float("inf")))


@unittest.skipUnless(HAS_TORCH, "PyTorch not installed")
class ActionFeatureModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from pandion.geographic_family import build_synthetic_cems_fixture, generate_geographic_scenario
        from pandion.geographic_integrated import run_geographic_integrated
        from pandion.policy_inputs import GeographicInputTensorizer
        cls.torch = torch
        fixture = build_synthetic_cems_fixture("EMSR578", utm_zone=31, rows=40, cols=40)
        cls.scenario = generate_geographic_scenario(fixture, index=0, split="development",
                                                    window_rows=12, window_cols=12, horizon_s=900.0)
        cls.rows = int(cls.scenario["grid"]["rows"])
        cls.distances = (0, 1)
        cls.mapper = ParametricActionMapper(float(cls.scenario["grid"]["cell_size_m"]), cls.distances)
        decisions = [d["observation"] for d in run_geographic_integrated(cls.scenario, policy="heuristic")["decisions"]]
        cls.observations = [o for o in decisions if any(cls.mapper.compute_mask(o)[2:])]
        if not cls.observations:
            cls.observations = decisions
        cls.tz = GeographicInputTensorizer(cls.scenario, None, explicit_fire=True, focus_slots=True,
                                           action_features=True, drop_distances=cls.distances)

    def make_model(self, **overrides):
        from pandion.image_model import ImageActorCritic
        kwargs = dict(use_image=False, action_mode="parametric", drop_distances=self.distances,
                      action_feature_dim=NUM_ACTION_FEATURES, cell_channels=self.tz.num_cell_channels,
                      vector_dim=self.tz.num_vector_features, px_per_cell=1, grid_rows=self.rows,
                      grid_cols=self.rows, width=8, hidden_dim=16, lstm_hidden_dim=16)
        kwargs.update(overrides)
        self.torch.manual_seed(0)
        return ImageActorCritic(**kwargs)

    def sequence(self, count):
        picks = [self.observations[min(i, len(self.observations) - 1)] for i in range(count)]
        return picks

    def test_get_action_and_evaluate_actions_run_on_tensorizer_outputs(self):
        torch = self.torch
        model = self.make_model()
        obs = self.observations[-1]
        inputs = self.tz.to_torch(obs)
        self.assertEqual(len(inputs), 3)
        self.assertEqual(inputs[2].dtype, torch.float32)
        self.assertEqual(tuple(inputs[2].shape), (1, self.mapper.num_actions - 2, 1 + NUM_ACTION_FEATURES))
        mask = self.mapper.to_mask_torch(obs)
        actions, logp, entropy, values, state = model.get_action(*inputs, mask, state=None)
        self.assertEqual(tuple(actions.shape), (1,))
        self.assertTrue(bool(mask[0, int(actions[0])]))

        seq = self.sequence(3)
        stacked = [torch.cat([self.tz.to_torch(o)[i] for o in seq]) for i in range(3)]
        seq_mask = torch.cat([self.mapper.to_mask_torch(o) for o in seq])
        chosen = torch.tensor([int(np.flatnonzero(seq_mask[t].numpy())[0]) for t in range(3)])
        log_probs, entropies, values = model.evaluate_actions(*stacked, seq_mask, chosen)
        self.assertEqual(tuple(log_probs.shape), (3,))
        self.assertEqual(tuple(values.shape), (3,))
        self.assertTrue(torch.isfinite(log_probs).all())

    def test_zero_init_head_ignores_action_feature_values(self):
        torch = self.torch
        model = self.make_model()
        inputs = self.tz.to_torch(self.observations[-1])
        with torch.no_grad():
            base_logits, _, _ = model.forward(*inputs)
            torch.manual_seed(7)
            noisy_actions = torch.cat([inputs[2][..., :1], torch.randn_like(inputs[2][..., 1:])], dim=-1)
            noisy_logits, _, _ = model.forward(inputs[0], inputs[1], noisy_actions)
        self.assertTrue(torch.equal(base_logits, noisy_logits))

    def test_invalid_actions_stay_unchanged_when_head_is_nonzero(self):
        torch = self.torch
        from torch import nn
        model = self.make_model()
        obs = next((o for o in self.observations if any(self.mapper.compute_mask(o)[2:])), None)
        self.assertIsNotNone(obs)
        inputs = self.tz.to_torch(obs)
        with torch.no_grad():
            zero_logits, _, _ = model.forward(*inputs)
            nn.init.constant_(model.target_actor_head[-1].weight, 0.1)
            changed_logits, _, _ = model.forward(*inputs)
        valid = inputs[2][0, :, 1].numpy() == 1
        invalid = torch.from_numpy(~valid)
        valid_t = torch.from_numpy(valid)
        self.assertTrue(valid_t.any())
        self.assertTrue(torch.equal(zero_logits[0, 2:][invalid], changed_logits[0, 2:][invalid]))
        self.assertFalse(torch.equal(zero_logits[0, 2:][valid_t], changed_logits[0, 2:][valid_t]))

    def test_wrong_number_of_inputs_raises(self):
        model = self.make_model()
        inputs = self.tz.to_torch(self.observations[-1])
        with self.assertRaises(ValueError):
            model.forward(inputs[0], inputs[1])
        with self.assertRaises(ValueError):
            model.forward(*inputs, inputs[2])

    def test_action_features_require_parametric_mode(self):
        from pandion.image_model import ImageActorCritic
        with self.assertRaises(ValueError):
            ImageActorCritic(use_image=False, action_mode="tactical", action_feature_dim=NUM_ACTION_FEATURES,
                             cell_channels=self.tz.num_cell_channels, vector_dim=self.tz.num_vector_features,
                             px_per_cell=1, grid_rows=self.rows, grid_cols=self.rows)


class ActionFeatureTensorizerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from pandion.geographic_family import build_synthetic_cems_fixture, generate_geographic_scenario
        from pandion.geographic_integrated import run_geographic_integrated
        fixture = build_synthetic_cems_fixture("EMSR578", utm_zone=31, rows=40, cols=40)
        cls.scenario = generate_geographic_scenario(fixture, index=0, split="development",
                                                    window_rows=12, window_cols=12, horizon_s=900.0)
        cls.rows = int(cls.scenario["grid"]["rows"])
        cls.observation = run_geographic_integrated(cls.scenario, policy="heuristic")["decisions"][-1]["observation"]
        cls.base = np.full((cls.rows * PX, cls.rows * PX, 3), 80, dtype=np.uint8)

    def test_action_features_add_one_trailing_float_array(self):
        from pandion.policy_inputs import GeographicInputTensorizer
        tz = GeographicInputTensorizer(self.scenario, self.base, focus_slots=True,
                                       action_features=True, drop_distances=(0, 1))
        arrays = tz.extract_arrays(self.observation)
        self.assertEqual(len(arrays), 4)
        rgb, cells, vector, actions = arrays
        self.assertEqual(actions.dtype, np.float32)
        self.assertEqual(actions.shape, (num_actions((0, 1)) - 2, 1 + NUM_ACTION_FEATURES))
        self.assertEqual(rgb.shape[0], 3)

    def test_action_features_need_focus_slots(self):
        from pandion.policy_inputs import GeographicInputTensorizer
        with self.assertRaises(ValueError):
            GeographicInputTensorizer(self.scenario, self.base, focus_slots=False, action_features=True)

    def test_default_tensorizer_output_is_unchanged(self):
        from pandion.policy_inputs import GeographicInputTensorizer
        tz = GeographicInputTensorizer(self.scenario, self.base, focus_slots=True)
        self.assertEqual(len(tz.extract_arrays(self.observation)), 3)
        self.assertEqual(tz.num_action_features, 0)
        plain = GeographicInputTensorizer(self.scenario, self.base)
        self.assertEqual(len(plain.extract(self.observation)), 3)


if __name__ == "__main__":
    unittest.main()
