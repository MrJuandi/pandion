"""Tests for the per-step RGB view composite."""

import unittest

import numpy as np

from nightjar.geographic_family import build_synthetic_cems_fixture, generate_geographic_scenario
from nightjar.geographic_integrated import run_geographic_integrated
from nightjar.imagery_composite import (
    BURNED_RGB,
    BURNING_RGB,
    THERMAL_BURNED_PEAK,
    compose_view,
    state_masks,
    thermal_view,
)


def observation(states, treated=(), arrivals=None, time_s=0.0):
    arrivals = arrivals or [[None] * len(row) for row in states]
    return {"time_s": time_s,
            "fire": {"cells": [[{"fire_state": s, "arrival_time_s": a} for s, a in zip(row, arow)]
                               for row, arow in zip(states, arrivals)]},
            "known_treated_cells": [list(c) for c in treated]}


class ImageryCompositeTests(unittest.TestCase):
    def setUp(self):
        self.base = np.full((4, 6, 3), 100, dtype=np.uint8)  # 2x3 cells, 2 px per cell

    def test_unburned_untreated_cells_keep_base_pixels(self):
        obs = observation([["unburned"] * 3, ["unburned"] * 3])
        np.testing.assert_array_equal(compose_view(self.base, obs, 2), self.base)

    def test_states_tint_whole_cell_blocks_in_grid_orientation(self):
        obs = observation([["burning", "burned", "unburned"], ["unburned", "unburned", "unburned"]],
                          treated=[(1, 2)])
        view = compose_view(self.base, obs, 2)
        burning, burned, treated, clean = view[0:2, 0:2], view[0:2, 2:4], view[2:4, 4:6], view[2:4, 0:2]
        self.assertTrue((burning == burning[0, 0]).all())
        self.assertTrue((burned == burned[0, 0]).all())
        # Burning moves toward flame orange: red up, blue down.
        self.assertGreater(int(burning[0, 0, 0]), 200)
        self.assertLess(int(burning[0, 0, 2]), 40)
        # Burned moves toward char: darker than base on every channel.
        self.assertTrue((burned[0, 0] < 100).all())
        self.assertTrue((burned[0, 0] > np.asarray(BURNED_RGB) - 1).all())
        # Treated is blue-tinted.
        self.assertGreater(int(treated[0, 0, 2]), int(treated[0, 0, 0]))
        np.testing.assert_array_equal(clean, self.base[2:4, 0:2])
        self.assertEqual(view.dtype, np.uint8)
        self.assertNotEqual(tuple(burning[0, 0]), BURNING_RGB)  # blended, not replaced

    def test_rejects_delayed_observations_and_mismatched_base(self):
        delayed = {"fire": {"cells": [[{"fire_detected": True}]]}}
        with self.assertRaises(ValueError):
            state_masks(delayed)
        with self.assertRaises(ValueError):
            compose_view(self.base, observation([["unburned"] * 3]), 2)

    def test_composes_realtime_simulator_observations(self):
        fixture = build_synthetic_cems_fixture("EMSR578", utm_zone=31, rows=40, cols=40)
        scenario = generate_geographic_scenario(fixture, index=0, split="development",
                                                window_rows=30, window_cols=30, horizon_s=1800.0)
        replay = run_geographic_integrated(scenario, policy="heuristic")
        base = np.full((30 * 8, 30 * 8, 3), 90, dtype=np.uint8)
        last = replay["decisions"][-1]["observation"]
        view = compose_view(base, last, 8)
        masks = state_masks(last)
        changed = (view != base).any(axis=-1).reshape(30, 8, 30, 8).any(axis=(1, 3))
        expected = masks["burning"] | masks["burned"] | masks["treated"]
        np.testing.assert_array_equal(changed, expected)
        self.assertTrue(masks["treated"].any())

    def test_thermal_band_burning_hot_halo_decays_burned_cools(self):
        states = [["unburned"] * 9 for _ in range(9)]
        states[4][4] = "burning"
        states[0][0] = "burned"
        arrivals = [[None] * 9 for _ in range(9)]
        arrivals[0][0] = 0.0
        early = thermal_view(observation(states, arrivals=arrivals, time_s=60.0), 4, 100.0)
        late = thermal_view(observation(states, arrivals=arrivals, time_s=1800.0), 4, 100.0)
        self.assertEqual(early.shape, (36, 36))
        self.assertTrue((early[16:20, 16:20] == 1.0).all())
        row = early[17, 20:]
        self.assertTrue((np.diff(row) <= 1e-6).all())  # halo decays away from the fire
        self.assertLess(row[-1], 0.05)
        self.assertGreater(early[1, 1], late[1, 1])  # burned cell cools
        self.assertLessEqual(early[1, 1], THERMAL_BURNED_PEAK + 1e-6)
        self.assertTrue(((early >= 0) & (early <= 1)).all())

    def test_thermal_band_is_cold_without_fire(self):
        cold = thermal_view(observation([["unburned"] * 3] * 2), 2, 100.0)
        self.assertTrue((cold == 0).all())


if __name__ == "__main__":
    unittest.main()
