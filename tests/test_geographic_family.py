"""Unit tests for the real-world geographic scenario family generator."""

from pathlib import Path
import tempfile
import unittest

from nightjar.geographic_family import (
    FAMILY_SCHEMA_VERSION,
    build_geographic_family,
    build_synthetic_cems_fixture,
    canonical_hash,
    generate_geographic_scenario,
    slice_tactical_aoi,
)


class TestGeographicFamily(unittest.TestCase):
    def setUp(self):
        self.fixture_31 = build_synthetic_cems_fixture("EMSR578", utm_zone=31, rows=60, cols=60)
        self.fixture_30 = build_synthetic_cems_fixture("EMSR590", utm_zone=30, rows=60, cols=60)

    def test_fixture_integrity(self):
        self.assertEqual(self.fixture_31["schema_version"], "nightjar-cems-scenario-0.1")
        self.assertEqual(self.fixture_31["crs"]["utm_zone"], 31)
        self.assertEqual(self.fixture_30["crs"]["utm_zone"], 30)
        self.assertEqual(len(self.fixture_31["fuel_grid"]), 60)
        self.assertEqual(len(self.fixture_31["fuel_grid"][0]), 60)

    def test_slice_tactical_aoi(self):
        sliced = slice_tactical_aoi(self.fixture_31, window_rows=30, window_cols=30)
        self.assertEqual(sliced["rows"], 30)
        self.assertEqual(sliced["cols"], 30)
        self.assertEqual(len(sliced["fuel_grid"]), 30)
        self.assertEqual(len(sliced["fuel_grid"][0]), 30)
        self.assertGreaterEqual(sliced["origin_utm_easting_m"], self.fixture_31["grid"]["origin_utm_easting_m"])
        self.assertGreaterEqual(sliced["origin_utm_northing_m"], self.fixture_31["grid"]["origin_utm_northing_m"])

    def test_generate_geographic_scenario_deterministic(self):
        scen1 = generate_geographic_scenario(self.fixture_31, index=0, split="development", window_rows=30, window_cols=30)
        scen2 = generate_geographic_scenario(self.fixture_31, index=0, split="development", window_rows=30, window_cols=30)
        self.assertEqual(canonical_hash(scen1), canonical_hash(scen2))
        self.assertEqual(scen1["scenario_id"], "geographic-dev-emsr578-00")
        self.assertEqual(scen1["crs"]["utm_zone"], 31)
        self.assertIn("fuel_grid", scen1["fire"])
        self.assertIn("burn_mask", scen1["fire"])
        self.assertIn("wind_to_m_s", scen1["fire"])
        self.assertIn("moisture", scen1["fire"])
        self.assertEqual(scen1["aircraft"]["model"], "CL-415")
        self.assertEqual(scen1["aircraft"]["tank_capacity_l"], 6137)
        self.assertGreater(len(scen1["observation_schedule"]), 0)

    def test_scenarios_differ_across_indices_and_splits(self):
        scen_dev0 = generate_geographic_scenario(self.fixture_31, index=0, split="development")
        scen_dev1 = generate_geographic_scenario(self.fixture_31, index=1, split="development")
        scen_val0 = generate_geographic_scenario(self.fixture_30, index=0, split="validation")
        self.assertNotEqual(canonical_hash(scen_dev0), canonical_hash(scen_dev1))
        self.assertNotEqual(canonical_hash(scen_dev0), canonical_hash(scen_val0))
        self.assertEqual(scen_dev0["crs"]["utm_zone"], 31)
        self.assertEqual(scen_val0["crs"]["utm_zone"], 30)

    def test_build_geographic_family_manifest(self):
        cems_bases = {
            "EMSR578": self.fixture_31,
            "EMSR590": self.fixture_30,
        }
        family = build_geographic_family(cems_bases, num_dev=4, num_val=4, num_res=4)
        manifest = family["manifest"]
        self.assertEqual(manifest["schema_version"], FAMILY_SCHEMA_VERSION)
        self.assertEqual(manifest["counts"]["total"], 12)
        self.assertEqual(manifest["counts"]["development"], 4)
        self.assertEqual(manifest["counts"]["validation"], 4)
        self.assertEqual(manifest["counts"]["reserved"], 4)

        # Ensure all scenario IDs and hashes are distinct
        all_ids = set()
        all_hashes = set()
        for split, scens in manifest["scenarios"].items():
            for item in scens:
                all_ids.add(item["scenario_id"])
                all_hashes.add(item["hash"])
        self.assertEqual(len(all_ids), 12)
        self.assertEqual(len(all_hashes), 12)


if __name__ == "__main__":
    unittest.main()
