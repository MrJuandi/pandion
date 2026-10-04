"""Tests for D4 geometric augmentation of geographic scenarios (D-65)."""

import unittest

import numpy as np

from nightjar.augmentation import (
    NUM_TRANSFORMS,
    augment_image,
    augment_scenario,
    transform_grid,
    transform_point,
    transform_vector,
)
from nightjar.geographic_family import build_synthetic_cems_fixture, generate_geographic_scenario
from nightjar.geographic_integrated import run_geographic_integrated


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
                    x, y = transform_point((c + 0.5) * 100, (r + 0.5) * 100, t, 400, 400)
                    self.assertEqual(moved[int(y // 100), int(x // 100)], values[r, c])

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
        self.assertNotEqual(moved["aircraft"]["initial_position_utm"], base["aircraft"]["initial_position_utm"])
        self.assertEqual(sum(map(sum, moved["fire"]["burn_mask"])), sum(map(sum, base["fire"]["burn_mask"])))

    def test_image_follows_the_grid(self):
        image = np.zeros((8, 8, 3), np.uint8)
        image[0:2, 6:8] = 255  # south-east cell at 2 px per cell
        moved = augment_image(image, 1)  # quarter turn: south-east -> north-east
        self.assertTrue((moved[6:8, 6:8] == 255).all())
        self.assertEqual(int(moved.sum()), int(image.sum()))


if __name__ == "__main__":
    unittest.main()
