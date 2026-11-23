"""Unit tests for pilot_emsr578 module (geodesy, fuel crosswalk, and spatial metrics)."""

from __future__ import annotations

import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from zipfile import ZipFile

import numpy as np

from pandion import pilot_emsr578


class GeodeticProjectionTests(unittest.TestCase):
    def test_wgs84_to_utm31n_known_point(self):
        # Reference coordinates near Artesa de Segre: lon 1.0°E, lat 41.9°N
        lon, lat = 1.0, 41.9
        x, y = pilot_emsr578.wgs84_to_utm31n(lon, lat)
        self.assertAlmostEqual(x, 334101.978, delta=0.5)
        self.assertAlmostEqual(y, 4640607.540, delta=0.5)

    def test_roundtrip_conversion_precision(self):
        test_points = [
            (1.0, 41.9),
            (1.045, 41.93),
            (0.95, 42.0),
            (1.15, 41.85),
        ]
        for lon, lat in test_points:
            x, y = pilot_emsr578.wgs84_to_utm31n(lon, lat)
            lon_back, lat_back = pilot_emsr578.utm31n_to_wgs84(x, y)
            self.assertAlmostEqual(lon, lon_back, places=7)
            self.assertAlmostEqual(lat, lat_back, places=7)

    def test_invalid_coordinates_raise_value_error(self):
        with self.assertRaises(ValueError):
            pilot_emsr578.wgs84_to_utm31n(float("nan"), 41.9)
        with self.assertRaises(ValueError):
            pilot_emsr578.wgs84_to_utm31n(1.0, 95.0)
        with self.assertRaises(ValueError):
            pilot_emsr578.utm31n_to_wgs84(float("inf"), 4640607.0)


class FuelCrosswalkTests(unittest.TestCase):
    def test_all_13_corine_classes_present(self):
        expected_classes = {
            "211", "212", "221", "222", "242", "243",
            "311", "312", "313", "321", "323", "324", "998"
        }
        self.assertEqual(set(pilot_emsr578.CORINE_TO_SCOTT_BURGAN.keys()), expected_classes)

    def test_crosswalk_entries_completeness(self):
        required_fields = {
            "clc_code", "clc_name", "sb_model", "sb_code",
            "description", "confidence", "rationale", "alternatives"
        }
        for code, entry in pilot_emsr578.CORINE_TO_SCOTT_BURGAN.items():
            self.assertEqual(set(entry.keys()), required_fields, f"Missing fields in {code}")
            self.assertIsInstance(entry["sb_code"], int)
            self.assertIn(entry["confidence"], {"High", "Medium", "Low"})
            self.assertIsInstance(entry["alternatives"], list)

    def test_clc_to_scott_burgan_mapping(self):
        self.assertEqual(pilot_emsr578.clc_to_scott_burgan("323"), 145)  # SH5
        self.assertEqual(pilot_emsr578.clc_to_scott_burgan("312"), 165)  # TU5
        self.assertEqual(pilot_emsr578.clc_to_scott_burgan("311"), 186)  # TL6
        self.assertEqual(pilot_emsr578.clc_to_scott_burgan("211"), 101)  # GR1
        self.assertEqual(pilot_emsr578.clc_to_scott_burgan("212"), 0)    # Nonburnable
        self.assertEqual(pilot_emsr578.clc_to_scott_burgan("323-Sclerophyllous"), 145)
        self.assertEqual(pilot_emsr578.clc_to_scott_burgan("99999-Unknown"), 0)


class SpatialOverlapMetricsTests(unittest.TestCase):
    def test_identical_masks(self):
        mask = np.array([[True, False], [True, True]])
        metrics = pilot_emsr578.evaluate_spatial_overlap(mask, mask, cell_area_ha=1.0)
        self.assertEqual(metrics["iou"], 1.0)
        self.assertEqual(metrics["dice"], 1.0)
        self.assertEqual(metrics["commission_rate"], 0.0)
        self.assertEqual(metrics["omission_rate"], 0.0)
        self.assertEqual(metrics["area_error_ha"], 0.0)
        self.assertEqual(metrics["counts"]["true_positive_cells"], 3)
        self.assertEqual(metrics["counts"]["false_positive_cells"], 0)
        self.assertEqual(metrics["counts"]["false_negative_cells"], 0)
        self.assertEqual(metrics["counts"]["true_negative_cells"], 1)

    def test_disjoint_masks(self):
        pred = np.array([[True, False], [False, False]])
        obs = np.array([[False, False], [False, True]])
        metrics = pilot_emsr578.evaluate_spatial_overlap(pred, obs, cell_area_ha=2.5)
        self.assertEqual(metrics["iou"], 0.0)
        self.assertEqual(metrics["dice"], 0.0)
        self.assertEqual(metrics["commission_rate"], 1.0)
        self.assertEqual(metrics["omission_rate"], 1.0)
        self.assertEqual(metrics["intersection_area_ha"], 0.0)
        self.assertEqual(metrics["predicted_area_ha"], 2.5)
        self.assertEqual(metrics["observed_area_ha"], 2.5)

    def test_partial_overlap(self):
        # 4 cells:
        # pred: [1, 1, 0, 0] (2 cells)
        # obs:  [0, 1, 1, 0] (2 cells)
        # TP = 1 (cell 1)
        # FP = 1 (cell 0)
        # FN = 1 (cell 2)
        # TN = 1 (cell 3)
        # IoU = 1 / 3 = 0.333333
        # Dice = 2*1 / (2+2) = 0.5
        pred = np.array([[True, True], [False, False]])
        obs = np.array([[False, True], [True, False]])
        metrics = pilot_emsr578.evaluate_spatial_overlap(pred, obs, cell_area_ha=1.0)
        self.assertAlmostEqual(metrics["iou"], 1.0 / 3.0, places=5)
        self.assertAlmostEqual(metrics["dice"], 0.5, places=5)
        self.assertAlmostEqual(metrics["commission_rate"], 0.5, places=5)
        self.assertAlmostEqual(metrics["omission_rate"], 0.5, places=5)

    def test_shape_mismatch_raises_value_error(self):
        pred = np.zeros((2, 2))
        obs = np.zeros((3, 3))
        with self.assertRaises(ValueError):
            pilot_emsr578.evaluate_spatial_overlap(pred, obs, cell_area_ha=1.0)


class ScenarioManifestBuilderTests(unittest.TestCase):
    def test_build_scenario_with_mock_zip(self):
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            del_zip = temp_path / "del.zip"

            # Create mock vector package with minimal valid geometries
            aoi_geojson = {
                "type": "FeatureCollection",
                "features": [{
                    "type": "Feature",
                    "properties": {"emsr_id": "EMSR578", "locality": "Artesa De Segre"},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[[1.0, 41.9], [1.02, 41.9], [1.02, 41.92], [1.0, 41.92], [1.0, 41.9]]],
                    },
                }],
            }
            landuse_geojson = {
                "type": "FeatureCollection",
                "features": [{
                    "type": "Feature",
                    "properties": {"info": "323-Sclerophyllous vegetation"},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[[1.002, 41.902], [1.018, 41.902], [1.018, 41.918], [1.002, 41.918], [1.002, 41.902]]],
                    },
                }],
            }
            hydro_geojson = {
                "type": "FeatureCollection",
                "features": [{
                    "type": "Feature",
                    "properties": {"obj_type": "BH130-Reservoir"},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[[1.005, 41.905], [1.010, 41.905], [1.010, 41.910], [1.005, 41.910], [1.005, 41.905]]],
                    },
                }],
            }
            obs_geojson = {
                "type": "FeatureCollection",
                "features": [{
                    "type": "Feature",
                    "properties": {"notation": "Burnt area", "area": 10.0},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[[1.004, 41.904], [1.008, 41.904], [1.008, 41.908], [1.004, 41.908], [1.004, 41.904]]],
                    },
                }],
            }

            with ZipFile(del_zip, "w") as z:
                z.writestr("EMSR578_AOI01_DEL_PRODUCT_areaOfInterestA_r1_v1.json", json.dumps(aoi_geojson))
                z.writestr("EMSR578_AOI01_DEL_PRODUCT_naturalLandUseA_r1_v1.json", json.dumps(landuse_geojson))
                z.writestr("EMSR578_AOI01_DEL_PRODUCT_hydrographyA_r1_v1.json", json.dumps(hydro_geojson))
                z.writestr("EMSR578_AOI01_DEL_PRODUCT_observedEventA_r1_v1.json", json.dumps(obs_geojson))

            scenario = pilot_emsr578.build_emsr578_scenario(temp_path, cell_size_m=200.0)

            self.assertEqual(scenario["schema_version"], "nightjar-emsr578-pilot-0.1")
            self.assertEqual(scenario["spatial_reference"]["crs"], "EPSG:25831")
            self.assertGreater(scenario["grid"]["rows"], 0)
            self.assertGreater(scenario["grid"]["cols"], 0)
            self.assertEqual(len(scenario["water_sources"]), 1)
            self.assertEqual(scenario["water_sources"][0]["type"], "BH130-Reservoir")

            fuel_arr = np.array(scenario["fuel_model"])
            # Fuel code 145 (SH5) must be present from CLC 323
            self.assertIn(145, fuel_arr)
            # Water nonburnable 0 must be present
            self.assertIn(0, fuel_arr)

    def test_build_scenario_with_real_emsr578_if_available(self):
        real_del = Path("data/raw/emsr578/del.zip")
        if not real_del.exists():
            self.skipTest("data/raw/emsr578/del.zip not found on disk")

        scenario = pilot_emsr578.build_emsr578_scenario(Path("data/raw/emsr578"), cell_size_m=200.0)
        self.assertEqual(scenario["schema_version"], "nightjar-emsr578-pilot-0.1")
        self.assertEqual(scenario["spatial_reference"]["crs"], "EPSG:25831")
        self.assertGreater(scenario["initial_fire"]["raster_burned_cells"], 0)
        self.assertAlmostEqual(
            scenario["initial_fire"]["raster_burned_area_ha"],
            scenario["initial_fire"]["reported_burned_area_ha"],
            delta=50.0,  # 200m grid resolution discretization tolerance
        )


if __name__ == "__main__":
    unittest.main()
