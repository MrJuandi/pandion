"""Tests for D4 geometric augmentation of geographic scenarios (D-65)."""

import unittest

import numpy as np

from pandion.augmentation import (
    NUM_TRANSFORMS,
    augment_image,
    augment_scenario,
    transform_grid,
    transform_point,
    transform_vector,
)
from pandion.geographic_family import build_synthetic_cems_fixture, canonical_hash, generate_geographic_scenario
from pandion.geographic_integrated import run_geographic_integrated


def scenario():
    fixture = build_synthetic_cems_fixture("EMSR578", utm_zone=31, rows=30, cols=30)
    return generate_geographic_scenario(fixture, index=0, split="development",
                                        window_rows=20, window_cols=20, horizon_s=1200.0)


def arrivals(replay):
    return np.array([[np.nan if v is None else v for v in row] for row in replay["arrival_time_s"]], float)


class AugmentationTests(unittest.TestCase):
    def test_quarter_turn_is_counter_clockwise_and_group_closes(self):
        grid = np.zeros((5, 5), int)
        grid[2, 4] = 1  # east edge (row 0 = south)
        self.assertEqual(tuple(np.argwhere(transform_grid(grid, 1))[0]), (4, 2))  # north edge
        self.assertEqual(transform_vector((1.0, 0.0), 1), (0.0, 1.0))
        values = np.arange(25).reshape(5, 5)
        for t in range(NUM_TRANSFORMS):
            seen = {transform_grid(values, t).tobytes() for t in range(NUM_TRANSFORMS)}
        self.assertEqual(len(seen), NUM_TRANSFORMS)  # eight distinct symmetries

    def test_points_and_grids_transform_consistently(self):
        values = np.arange(16).reshape(4, 4)
        for t in range(NUM_TRANSFORMS):
            moved = transform_grid(values, t)
            for r in range(4):
                for c in range(4):
                    # Simulator convention: cell (r, c) at (c, r) * size, points map by rounding.
                    x, y = transform_point(c * 100.0, r * 100.0, t, 300, 300)
                    self.assertEqual(moved[int(round(y / 100)), int(round(x / 100))], values[r, c])

    def test_simulated_fire_is_equivariant(self):
        base = scenario()
        reference = run_geographic_integrated(base, "no_intervention")
        for t in range(NUM_TRANSFORMS):
            replay = run_geographic_integrated(augment_scenario(base, t), "no_intervention")
            self.assertEqual(replay["outcomes"]["final_burned_ha"], reference["outcomes"]["final_burned_ha"])
            expected = transform_grid(arrivals(reference), t)
            got = arrivals(replay)
            np.testing.assert_array_equal(np.isnan(got), np.isnan(expected))
            self.assertLess(np.nanmax(np.abs(got - expected)), 0.05)

    def test_augmented_scenario_is_marked_and_identity_is_unchanged(self):
        base = scenario()
        self.assertIs(augment_scenario(base, 0), base)
        moved = augment_scenario(base, 5)
        self.assertEqual(moved["augmentation"], {"transform": 5})
        self.assertIn("augmentation", moved["provenance"])
        self.assertEqual(moved["provenance"]["augmentation"]["source_scenario_hash"], base["scenario_hash"])
        self.assertEqual(moved["scenario_hash"],
                         canonical_hash({k: v for k, v in moved.items() if k != "scenario_hash"}))
        self.assertNotEqual(moved["aircraft"]["initial_position_utm"], base["aircraft"]["initial_position_utm"])
        self.assertEqual(sum(map(sum, moved["fire"]["burn_mask"])), sum(map(sum, base["fire"]["burn_mask"])))

    def test_aircraft_and_water_stay_on_their_transformed_cells(self):
        base = scenario()
        size = float(base["grid"]["cell_size_m"])
        x0, y0 = float(base["grid"]["origin_utm_easting_m"]), float(base["grid"]["origin_utm_northing_m"])
        rows, cols = int(base["grid"]["rows"]), int(base["grid"]["cols"])

        def cell(point):
            return int(round((point[1] - y0) / size)), int(round((point[0] - x0) / size))

        for t in range(NUM_TRANSFORMS):
            moved = augment_scenario(base, t)
            for key in ("initial_position_utm", "water_scooping_utm"):
                if key not in base["aircraft"]:
                    continue
                r, c = cell(base["aircraft"][key])
                if not (0 <= r < rows and 0 <= c < cols):
                    continue  # outside the window: no cell to compare
                marker = np.zeros((rows, cols), int)
                marker[r, c] = 1
                expected = tuple(np.argwhere(transform_grid(marker, t))[0])
                self.assertEqual(cell(moved["aircraft"][key]), expected, (key, t))

    def test_image_follows_the_grid(self):
        image = np.zeros((8, 8, 3), np.uint8)
        image[0:2, 6:8] = 255  # south-east cell at 2 px per cell
        moved = augment_image(image, 1)  # quarter turn: south-east -> north-east
        self.assertTrue((moved[6:8, 6:8] == 255).all())
        self.assertEqual(int(moved.sum()), int(image.sum()))


if __name__ == "__main__":
    unittest.main()
