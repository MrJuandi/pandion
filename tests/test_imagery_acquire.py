"""Offline tests for Sentinel-2 imagery selection, compositing and loading."""

from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from nightjar.imagery_acquire import (
    SCHEMA_VERSION,
    best_available_pixel,
    buffer_pixels,
    cems_code,
    family_grid,
    match_to_reference,
    load_activation_image,
    north_up_to_grid,
    obscured_mask,
    obscured_fraction,
    rank_scenes,
    search_window,
    target_grid,
)


def scenario(activation="EMSR578", rows=2, cols=3, x0=1000.0, y0=2000.0):
    return {"activation_id": activation, "scenario_id": f"{activation}-x",
            "crs": {"code": "EPSG:25831", "utm_zone": 31},
            "grid": {"rows": rows, "cols": cols, "cell_size_m": 100.0,
                     "origin_utm_easting_m": x0, "origin_utm_northing_m": y0}}


def item(item_id, date, cloud=10.0):
    return {"id": item_id, "properties": {"datetime": date, "eo:cloud_cover": cloud}}


class ImageryAcquireTests(unittest.TestCase):
    def test_cems_code_strips_aoi_suffix(self):
        self.assertEqual(cems_code("EMSR602_AOI02"), "EMSR602")
        self.assertEqual(cems_code("EMSR578"), "EMSR578")

    def test_search_window_ends_margin_before_activation(self):
        start, end = search_window("2022-06-16T13:04:00", 14, 75)
        self.assertEqual(end.isoformat(), "2022-06-02T13:04:00+00:00")
        self.assertEqual((end - start).days, 75)
        with self.assertRaises(ValueError):
            search_window("2022-06-16T13:04:00", -1, 75)

    def test_target_grid_is_north_up_and_covers_cells(self):
        grid = target_grid(40, 30, 100.0, 5000.0, 7000.0, 8)
        self.assertEqual((grid["height"], grid["width"]), (320, 240))
        self.assertEqual(grid["pixel_size_m"], 12.5)
        self.assertEqual(grid["transform"], [12.5, 0.0, 5000.0, 0.0, -12.5, 11000.0])

    def test_obscured_fraction_counts_cloud_shadow_cirrus_and_nodata(self):
        scl = np.array([[4, 5, 3, 8], [9, 10, 0, 7]])
        self.assertAlmostEqual(obscured_fraction(scl), 5 / 8)

    def test_obscured_mask_buffers_cloud_edges(self):
        scl = np.full((5, 5), 4)
        scl[2, 2] = 9
        self.assertEqual(obscured_mask(scl).sum(), 1)
        buffered = obscured_mask(scl, 1)
        self.assertEqual(buffered.sum(), 9)
        self.assertTrue(buffered[1:4, 1:4].all())
        self.assertEqual(buffer_pixels(12.5), 8)
        self.assertEqual(buffer_pixels(100.0), 1)

    def test_rank_scenes_uses_crop_screening_not_tile_cloud(self):
        cloudy_tile = item("a", "2022-05-01T10:00:00Z", cloud=80.0)
        clear_tile = item("b", "2022-05-06T10:00:00Z", cloud=5.0)
        crops = {"a": np.full((4, 4), 4), "b": np.full((4, 4), 9)}
        ranked = rank_scenes([clear_tile, cloudy_tile], lambda it: crops[it["id"]])
        self.assertEqual([it["id"] for it, _ in ranked], ["a", "b"])
        self.assertEqual([f for _, f in ranked], [0.0, 1.0])

    def test_rank_scenes_prefers_newest_on_ties(self):
        old, new = item("old", "2022-04-01T10:00:00Z"), item("new", "2022-05-01T10:00:00Z")
        ranked = rank_scenes([old, new], lambda it: np.full((2, 2), 4))
        self.assertEqual(ranked[0][0]["id"], "new")

    def test_best_available_pixel_fills_in_priority_order(self):
        first = np.full((2, 2, 3), 10, dtype=np.uint8)
        second = np.full((2, 2, 3), 20, dtype=np.uint8)
        clear_first = np.array([[True, False], [True, True]])
        clear_second = np.array([[True, True], [False, True]])
        composite, shares, unfilled = best_available_pixel([(first, clear_first), (second, clear_second)])
        self.assertEqual(composite[0, 1, 0], 20)
        self.assertTrue((composite[[0, 1, 1], [0, 0, 1], 0] == 10).all())
        self.assertEqual(shares, [0.75, 0.25])
        self.assertEqual(unfilled, 0.0)

    def test_best_available_pixel_reports_unfilled(self):
        image = np.ones((2, 2, 3), dtype=np.uint8)
        _, _, unfilled = best_available_pixel([(image, np.array([[True, False], [False, False]]))])
        self.assertEqual(unfilled, 0.75)

    def test_match_to_reference_aligns_shared_statistics(self):
        rng = np.random.default_rng(0)
        reference = rng.integers(50, 150, size=(40, 40, 3)).astype(np.uint8)
        hazy = np.clip(reference.astype(int) * 0.5 + 100, 0, 255).astype(np.uint8)
        clear = np.ones((40, 40), dtype=bool)
        matched = match_to_reference(hazy, clear, reference, clear)
        self.assertLess(np.abs(matched.astype(int) - reference.astype(int)).mean(), 1.5)
        small = np.zeros((40, 40), dtype=bool)
        np.testing.assert_array_equal(match_to_reference(hazy, small, reference, small), hazy)

    def test_family_grid_rejects_disagreeing_variants(self):
        self.assertEqual(family_grid([scenario(), scenario()])[:2], (2, 3))
        with self.assertRaises(ValueError):
            family_grid([scenario(), scenario(x0=1500.0)])

    def test_north_up_to_grid_puts_south_in_row_zero(self):
        north_up = np.zeros((2, 1, 3), dtype=np.uint8)
        north_up[0] = 255  # northern row
        grid = north_up_to_grid(north_up)
        self.assertEqual(grid[1, 0, 0], 255)
        self.assertEqual(grid[0, 0, 0], 0)

    def _write_imagery(self, directory: Path, image: np.ndarray, *, px=2, grid_of=None):
        grid_of = grid_of or scenario()
        np.save(directory / "emsr578.npy", image)
        g = grid_of["grid"]
        manifest = {"schema_version": SCHEMA_VERSION, "activations": {"EMSR578": {
            "crs": "EPSG:25831", "px_per_cell": px, "array": "emsr578.npy",
            "array_sha256": sha256((directory / "emsr578.npy").read_bytes()).hexdigest(),
            "grid": {"rows": g["rows"], "cols": g["cols"], "cell_size_m": g["cell_size_m"],
                     "origin_utm_easting_m": g["origin_utm_easting_m"],
                     "origin_utm_northing_m": g["origin_utm_northing_m"]}}}}
        (directory / "sources.json").write_text(json.dumps(manifest))

    def test_load_activation_image_validates_grid_hash_and_shape(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            image = np.arange(4 * 6 * 3, dtype=np.uint8).reshape(4, 6, 3)
            self._write_imagery(directory, image)
            np.testing.assert_array_equal(load_activation_image(directory, scenario()), image)
            with self.assertRaises(ValueError):
                load_activation_image(directory, scenario(x0=1100.0))
            with self.assertRaises(KeyError):
                load_activation_image(directory, scenario(activation="EMSR590"))
            np.save(directory / "emsr578.npy", image[::-1])
            with self.assertRaises(ValueError):
                load_activation_image(directory, scenario())

    def test_load_activation_image_rejects_wrong_shape(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            self._write_imagery(directory, np.zeros((3, 6, 3), dtype=np.uint8))
            with self.assertRaises(ValueError):
                load_activation_image(directory, scenario())


if __name__ == "__main__":
    unittest.main()
