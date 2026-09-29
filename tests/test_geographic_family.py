"""Unit tests for the real-world geographic scenario family generator."""

from pathlib import Path
import tempfile
import unittest
from contextlib import redirect_stderr
from io import StringIO


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
        self.assertEqual(self.fixture_31["source_kind"], "synthetic_fixture")
        self.assertNotIn("license", self.fixture_31["provenance"])
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
        self.assertEqual(scen1["source_kind"], "synthetic_fixture")
        self.assertEqual(scen1["provenance"]["assumptions"]["topography"], "flat")
        self.assertIn("crop", scen1["provenance"])
        self.assertEqual(scen1["scenario_hash"], canonical_hash({k: v for k, v in scen1.items() if k != "scenario_hash"}))
        self.assertIn("fuel_grid", scen1["fire"])
        self.assertIn("burn_mask", scen1["fire"])
        self.assertIn("wind_to_m_s", scen1["fire"])
        self.assertIn("moisture", scen1["fire"])
        self.assertEqual(scen1["aircraft"]["model"], "CL-415")
        self.assertEqual(scen1["aircraft"]["tank_capacity_l"], 6137)
        self.assertGreater(len(scen1["observation_schedule"]), 0)

    def test_family_parent_hash_matches_unmodified_source_base(self):
        from contextlib import ExitStack
        from unittest.mock import patch
        from nightjar.cems_catalog import ACTIVATIONS

        bases = {}
        expected_hashes = {}
        test_digest = "a" * 64
        with ExitStack() as patches:
            for activation_id in ("EMSR578", "EMSR590", "EMSR592", "EMSR600"):
                base = build_synthetic_cems_fixture(activation_id, 31 if activation_id == "EMSR578" else 30)
                base.pop("source_kind")
                base["activation_id"] = activation_id
                base["scenario_id"] = f"cems-{activation_id.lower()}-del-0.1"
                product = ACTIVATIONS[activation_id]["products"]["del"]
                patches.enter_context(patch.dict(product, {"sha256": test_digest}))
                base["provenance"] = {
                    "source": "synthetic test base with mocked catalog digest",
                    "source_zip_sha256": test_digest,
                    "source_stage": "del",
                    "source_product_code": product["product_code"],
                    "source_product_filename": product["vector_filename"],
                    "image_acquired_at_utc": product["acquired_at_utc"],
                }
                bases[activation_id] = base
                expected_hashes[activation_id] = canonical_hash(base)

            family = build_geographic_family(bases, num_dev=1, num_val=1, num_res=2)

        generated = family["scenarios"]["development"][0]
        self.assertEqual(generated["provenance"]["parent_scenario_hash"], expected_hashes["EMSR578"])
        self.assertNotIn("source_kind", bases["EMSR578"])

    def test_real_source_provenance_survives_scenario_generation(self):
        base = dict(self.fixture_31)
        base["source_kind"] = "copernicus_cems"
        base["provenance"] = {"source_zip_sha256": "a" * 64, "source_stage": "del",
                              "source_product_code": "PRODUCT", "source_product_filename": "source.zip",
                              "source_url": "https://example.invalid/source.zip",
                              "image_acquired_at_utc": "2022-06-17T10:06:00Z",
                              "units": {"coordinates": "m"}, "resolution_m": 100}
        scenario = generate_geographic_scenario(base, 0, "development")
        self.assertEqual(scenario["source_kind"], "copernicus_cems")
        self.assertEqual(scenario["provenance"]["source_zip_sha256"], "a" * 64)
        self.assertEqual(scenario["provenance"]["image_acquired_at_utc"], "2022-06-17T10:06:00Z")
        self.assertEqual(scenario["provenance"]["crop"]["rows"], scenario["grid"]["rows"])
        self.assertEqual(scenario["provenance"]["crop"]["crs"], scenario["crs"])

    def test_scenarios_differ_across_indices_and_splits(self):
        scen_dev0 = generate_geographic_scenario(self.fixture_31, index=0, split="development")
        scen_dev1 = generate_geographic_scenario(self.fixture_31, index=1, split="development")
        scen_val0 = generate_geographic_scenario(self.fixture_30, index=0, split="validation")
        self.assertNotEqual(canonical_hash(scen_dev0), canonical_hash(scen_dev1))
        self.assertNotEqual(canonical_hash(scen_dev0), canonical_hash(scen_val0))
        self.assertEqual(scen_dev0["crs"]["utm_zone"], 31)
        self.assertEqual(scen_val0["crs"]["utm_zone"], 30)

    def test_real_build_requires_all_four_incidents_and_counts(self):
        with self.assertRaisesRegex(ValueError, "EMSR578.*EMSR590.*EMSR592.*EMSR600"):
            build_geographic_family()
        bases = {key: build_synthetic_cems_fixture(key, 31 if key == "EMSR578" else 30)
                 for key in ("EMSR578", "EMSR590", "EMSR592", "EMSR600")}
        with self.assertRaisesRegex(ValueError, "num_res must be at least 2"):
            build_geographic_family(bases, num_res=1)
        with self.assertRaisesRegex(ValueError, "all be positive"):
            build_geographic_family(bases, num_dev=0)

    def test_cli_requires_local_products_and_explicit_synthetic(self):
        from nightjar.geographic_family import main
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            stderr = StringIO()
            with self.assertRaises(SystemExit) as error, redirect_stderr(stderr):
                import sys
                old_argv = sys.argv
                sys.argv = ["geographic_family", "--raw-dir", str(root / "raw"),
                            "--output-dir", str(root / "out")]
                try:
                    main()
                finally:
                    sys.argv = old_argv
            self.assertEqual(error.exception.code, 2)
            self.assertIn("emsr578_del.zip", stderr.getvalue())
            self.assertIn("no download", stderr.getvalue())
            self.assertFalse((root / "out").exists())

    def test_cli_explicit_synthetic_output_and_nonempty_guard(self):
        from nightjar.geographic_family import main
        import sys
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "family"
            old_argv = sys.argv
            try:
                sys.argv = ["geographic_family", "--synthetic", "--num-dev", "1", "--num-val", "1",
                            "--num-res", "2", "--output-dir", str(output)]
                main()
            finally:
                sys.argv = old_argv
            import json
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(manifest["source_kind"], "synthetic_fixture")
            self.assertEqual(manifest["counts"]["total"], 4)
            stderr = StringIO()
            with self.assertRaises(SystemExit) as error, redirect_stderr(stderr):
                old_argv = sys.argv
                sys.argv = ["geographic_family", "--synthetic", "--output-dir", str(output)]
                try:
                    main()
                finally:
                    sys.argv = old_argv
            self.assertEqual(error.exception.code, 2)
            self.assertIn("Refusing to mix", stderr.getvalue())

    def test_build_geographic_family_manifest(self):
        cems_bases = {
            "EMSR578": self.fixture_31,
            "EMSR590": self.fixture_30,
        }
        cems_bases.update({"EMSR592": build_synthetic_cems_fixture("EMSR592", 30),
                           "EMSR600": build_synthetic_cems_fixture("EMSR600", 30)})
        family = build_geographic_family(cems_bases, num_dev=4, num_val=4, num_res=4, synthetic=True)
        manifest = family["manifest"]
        self.assertEqual(manifest["schema_version"], FAMILY_SCHEMA_VERSION)
        self.assertEqual(manifest["counts"]["total"], 12)
        self.assertEqual(manifest["counts"]["development"], 4)
        self.assertEqual(manifest["counts"]["validation"], 4)
        self.assertEqual(manifest["counts"]["reserved"], 4)
        self.assertEqual(manifest["source_kind"], "synthetic_fixture")
        self.assertEqual(manifest["incidents_by_split"]["reserved"], ["EMSR592", "EMSR600"])
        reserved_ids = {item["activation_id"] for item in manifest["scenarios"]["reserved"]}
        self.assertEqual(reserved_ids, {"EMSR592", "EMSR600"})
        self.assertEqual({item["source_kind"] for item in family["scenarios"]["reserved"]}, {"synthetic_fixture"})
        self.assertTrue(set(manifest["incidents_by_split"]["development"]).isdisjoint(
            manifest["incidents_by_split"]["validation"]))

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
