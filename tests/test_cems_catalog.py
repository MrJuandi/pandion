"""Unit tests for the CEMS multi-incident catalog and scenario builder."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from zipfile import ZipFile

from nightjar.cems_catalog import (
    ACTIVATIONS,
    UNIVERSAL_CORINE_CROSSWALK,
    build_cems_scenario,
    cems_product_url,
    universal_clc_to_scott_burgan,
)


class TestCemsCatalog(unittest.TestCase):
    def test_universal_crosswalk_completeness(self):
        expected_classes = {
            "211", "212", "221", "222", "223", "231", "242", "243",
            "311", "312", "313", "321", "322", "323", "324", "333",
            "411", "998",
        }
        self.assertEqual(set(UNIVERSAL_CORINE_CROSSWALK.keys()), expected_classes)
        for code, entry in UNIVERSAL_CORINE_CROSSWALK.items():
            self.assertIn("sb_code", entry)
            self.assertIn("sb_model", entry)
            self.assertIn("rationale", entry)

    def test_universal_clc_mapping(self):
        self.assertEqual(universal_clc_to_scott_burgan("323-Sclerophyllous"), 145)  # SH5
        self.assertEqual(universal_clc_to_scott_burgan("223-Olive groves"), 0)       # NB
        self.assertEqual(universal_clc_to_scott_burgan("231-Pastures"), 102)        # GR2
        self.assertEqual(universal_clc_to_scott_burgan("322-Heathland"), 146)       # SH6
        self.assertEqual(universal_clc_to_scott_burgan("333-Sparse"), 101)          # GR1
        self.assertEqual(universal_clc_to_scott_burgan("411-Marshes"), 0)           # NB
        self.assertEqual(universal_clc_to_scott_burgan("unknown"), 0)

    def test_activation_registry_structure(self):
        for act_id in ("EMSR578", "EMSR590", "EMSR592", "EMSR600"):
            self.assertIn(act_id, ACTIVATIONS)
            act = ACTIVATIONS[act_id]
            self.assertIn(act["utm_zone"], (29, 30, 31, 32))
            self.assertTrue(act["epsg"].startswith("EPSG:"))
            self.assertIn(act["default_split"], ("development", "validation", "reserved"))
            self.assertIn("del", act["products"])
            url = cems_product_url(act_id, "del")
            self.assertTrue(url.startswith("https://"))
            self.assertTrue(url.endswith(".zip"))

    def test_additional_development_incidents_are_development_only(self):
        from nightjar.cems_catalog import incidents_for_split, raw_package_name
        development = incidents_for_split("development")
        self.assertEqual(development[0], "EMSR578")
        self.assertEqual(set(development[1:]), {"EMSR583", "EMSR625_AOI01", "EMSR625_AOI02",
                                                "EMSR602_AOI02", "EMSR602_AOI04",
                                                "EMSR365", "EMSR375", "EMSR457", "EMSR458", "EMSR522",
                                                "EMSR538", "EMSR545", "EMSR581", "EMSR597", "EMSR599_AOI01",
                                                "EMSR599_AOI02", "EMSR620", "EMSR623", "EMSR632"})
        self.assertEqual(incidents_for_split("validation"), ("EMSR590",))
        self.assertEqual(incidents_for_split("reserved"), ("EMSR592", "EMSR600"))
        self.assertEqual(raw_package_name("EMSR578"), "emsr578_del.zip")
        self.assertEqual(raw_package_name("EMSR625_AOI02"), "emsr625-aoi02_del.zip")
        self.assertTrue(cems_product_url("EMSR457").endswith("EMSR457_AOI01_DEL_PRODUCT_r1_RTP01_v2_vector.zip"))
        self.assertEqual(
            cems_product_url("EMSR625_AOI02"),
            "https://cems-mapping-website.s3.eu-west-1.amazonaws.com/static/activations/EMSR625/"
            "EMSR625_AOI02_DEL_PRODUCT_r1_RTP01_v1_vector.zip",
        )
        self.assertEqual(ACTIVATIONS["EMSR602_AOI02"]["epsg"], "EPSG:25829")
        self.assertIsNone(ACTIVATIONS["EMSR583"]["products"]["del"]["acquired_at_utc"])

    def test_registry_geography_and_crs_match_registered_activations(self):
        self.assertEqual(
            (ACTIVATIONS["EMSR590"]["name"], ACTIVATIONS["EMSR590"]["country"]),
            ("Las Hurdes / Sierra de Francia", "Spain"),
        )
        self.assertEqual(
            (ACTIVATIONS["EMSR592"]["name"], ACTIVATIONS["EMSR592"]["country"]),
            ("Landiras / La Teste-de-Buch", "France"),
        )
        self.assertEqual(
            (ACTIVATIONS["EMSR600"]["name"], ACTIVATIONS["EMSR600"]["country"],
             ACTIVATIONS["EMSR600"]["utm_zone"], ACTIVATIONS["EMSR600"]["epsg"]),
            ("Massarosa", "Italy", 32, "EPSG:25832"),
        )

    def test_correct_hash_does_not_bypass_geographic_bounds(self):
        from hashlib import sha256

        wrong_place = [[[-4.6, 36.6], [-4.5, 36.6], [-4.5, 36.7],
                        [-4.6, 36.7], [-4.6, 36.6]]]
        feature_collection = {"type": "FeatureCollection", "features": [{
            "type": "Feature", "geometry": {"type": "Polygon", "coordinates": wrong_place},
            "properties": {},
        }]}
        empty_collection = {"type": "FeatureCollection", "features": []}
        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = Path(tmpdir) / "mislabeled_emsr590.zip"
            with ZipFile(zip_path, "w") as zf:
                zf.writestr("areaOfInterestA.json", json.dumps(feature_collection))
                zf.writestr("naturalLandUseA.json", json.dumps(empty_collection))
                zf.writestr("observedEventA.json", json.dumps(empty_collection))
            digest = sha256(zip_path.read_bytes()).hexdigest()
            with patch.dict(ACTIVATIONS["EMSR590"]["products"]["del"], {"sha256": digest}):
                with self.assertRaisesRegex(ValueError, "outside registered WGS84 bounds"):
                    build_cems_scenario("EMSR590", zip_path)

    def test_build_cems_scenario_with_mock_zip_zone30(self):
        """Test building a scenario for Zone 30N (EMSR590) with synthetic GeoJSON."""
        aoi_poly = [[[-6.3, 40.43], [-6.1, 40.43], [-6.1, 40.52], [-6.3, 40.52], [-6.3, 40.43]]]
        burn_poly = [[[-6.25, 40.45], [-6.20, 40.45], [-6.20, 40.48], [-6.25, 40.48], [-6.25, 40.45]]]
        landuse_poly = [[[-6.28, 40.44], [-6.15, 40.44], [-6.15, 40.50], [-6.28, 40.50], [-6.28, 40.44]]]

        mock_aoi = {"type": "FeatureCollection", "features": [{"type": "Feature", "geometry": {"type": "Polygon", "coordinates": aoi_poly}, "properties": {}}]}
        mock_landuse = {"type": "FeatureCollection", "features": [{"type": "Feature", "geometry": {"type": "Polygon", "coordinates": landuse_poly}, "properties": {"info": "322-Heathland"}}]}
        mock_hydro = {"type": "FeatureCollection", "features": [{"type": "Feature", "geometry": {"type": "Point", "coordinates": [-6.22, 40.47]}, "properties": {"name": "Piscina Natural", "obj_type": "BH080-Lake"}}]}
        mock_obs = {"type": "FeatureCollection", "features": [{"type": "Feature", "geometry": {"type": "Polygon", "coordinates": burn_poly}, "properties": {"notation": "Burnt area", "area": 250.0}}]}

        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = Path(tmpdir) / "emsr590_mock.zip"
            with ZipFile(zip_path, "w") as zf:
                zf.writestr("EMSR590_AOI01_DEL_PRODUCT_areaOfInterestA_r1_v1.json", json.dumps(mock_aoi))
                zf.writestr("EMSR590_AOI01_DEL_PRODUCT_naturalLandUseA_r1_v1.json", json.dumps(mock_landuse))
                zf.writestr("EMSR590_AOI01_DEL_PRODUCT_hydrographyA_r1_v1.json", json.dumps(mock_hydro))
                zf.writestr("EMSR590_AOI01_DEL_PRODUCT_observedEventA_r1_v1.json", json.dumps(mock_obs))

            with self.assertRaisesRegex(ValueError, "differs from registered hash"):
                build_cems_scenario("EMSR590", zip_path, cell_size_m=500.0, stage="del")
            from hashlib import sha256
            fixture_hash = sha256(zip_path.read_bytes()).hexdigest()
            with patch.dict(ACTIVATIONS["EMSR590"]["products"]["del"], {"sha256": fixture_hash}):
                scenario = build_cems_scenario("EMSR590", zip_path, cell_size_m=500.0, stage="del")
            self.assertEqual(scenario["schema_version"], "nightjar-cems-scenario-0.1")
            self.assertEqual(scenario["activation_id"], "EMSR590")
            self.assertEqual(scenario["crs"]["utm_zone"], 30)
            self.assertEqual(scenario["crs"]["code"], "EPSG:25830")
            self.assertEqual(scenario["default_split"], "validation")
            self.assertEqual(scenario["provenance"]["source_zip_sha256"], fixture_hash)
            self.assertEqual(scenario["provenance"]["source_stage"], "del")
            self.assertGreater(scenario["initial_perimeter"]["burned_cells_count"], 0)
            # Check fuel code 146 (SH6 from 322) was rasterized
            fuel_grid = scenario["fuel_grid"]
            self.assertTrue(any(146 in row for row in fuel_grid))
            self.assertEqual(scenario["position_semantics"]["source_raster_samples"], "cell centers")
            self.assertEqual(scenario["grid"]["origin_semantics"], "southwest cell boundary")
            self.assertIn("unavailable coverage", scenario["provenance"]["coordinate_contract"]["fuel_coverage"])

    def test_asymmetric_projected_layers_use_north_rows_east_columns(self):
        from hashlib import sha256
        from nightjar.geodesy import utm_to_wgs84

        zone, size = 30, 100.0
        x0, y0 = 500000.25, 4400000.25

        def lonlat(x, y):
            return list(utm_to_wgs84(x, y, zone=zone))

        def polygon(col0, row0, col1, row1):
            return [[lonlat(x0 + col0 * size, y0 + row0 * size),
                     lonlat(x0 + col1 * size, y0 + row0 * size),
                     lonlat(x0 + col1 * size, y0 + row1 * size),
                     lonlat(x0 + col0 * size, y0 + row1 * size),
                     lonlat(x0 + col0 * size, y0 + row0 * size)]]

        def feature(coords, properties=None):
            return {"type": "Feature", "geometry": {"type": "Polygon", "coordinates": coords},
                    "properties": properties or {}}

        aoi = feature(polygon(0, 0, 4, 2))
        land = feature(polygon(2, 0, 3, 1), {"info": "231-Pastures"})
        water = feature(polygon(0, 1, 1, 2), {"name": "water"})
        burn = feature(polygon(3, 1, 4, 2))
        payloads = {
            "areaOfInterestA": {"type": "FeatureCollection", "features": [aoi]},
            "naturalLandUseA": {"type": "FeatureCollection", "features": [land]},
            "hydrographyA": {"type": "FeatureCollection", "features": [water]},
            "observedEventA": {"type": "FeatureCollection", "features": [burn]},
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "asymmetric.zip"
            with ZipFile(path, "w") as archive:
                for layer, data in payloads.items():
                    archive.writestr(f"{layer}.json", json.dumps(data))
            digest = sha256(path.read_bytes()).hexdigest()
            with patch.dict(ACTIVATIONS["EMSR590"]["products"]["del"], {"sha256": digest}), patch.dict(
                ACTIVATIONS["EMSR590"], {"aoi_bounds_wgs84": (-4.0, 39.0, -2.0, 41.0)}
            ):
                scenario = build_cems_scenario("EMSR590", path, cell_size_m=size)

        self.assertEqual((scenario["grid"]["rows"], scenario["grid"]["cols"]), (3, 4))
        self.assertEqual(scenario["fuel_grid"], [[0, 0, 102, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        self.assertEqual(scenario["water_mask"], [[False, False, False, False],
                                                     [True, False, False, False],
                                                     [False, False, False, False]])
        self.assertEqual(scenario["fuel_grid"][1][0], 0)
        self.assertEqual(scenario["burn_mask"], [[False, False, False, False],
                                                   [False, False, False, True],
                                                   [False, False, False, False]])
        self.assertAlmostEqual(scenario["grid"]["origin_utm_easting_m"], x0, places=2)
        self.assertAlmostEqual(scenario["grid"]["origin_utm_northing_m"], y0, places=2)

    def test_build_cems_scenario_with_real_emsr578_if_available(self):
        real_zip = Path("data/raw/emsr578/del.zip")
        if not real_zip.exists():
            self.skipTest("data/raw/emsr578/del.zip not present")

        scenario = build_cems_scenario("EMSR578", real_zip, cell_size_m=100.0, stage="del")
        self.assertEqual(scenario["schema_version"], "nightjar-cems-scenario-0.1")
        self.assertEqual(scenario["activation_id"], "EMSR578")
        self.assertEqual(scenario["crs"]["utm_zone"], 31)
        self.assertEqual(scenario["crs"]["code"], "EPSG:25831")
        self.assertEqual(scenario["default_split"], "development")
        # 1500 cells burned at 100m resolution = 1500.0 ha
        self.assertEqual(scenario["initial_perimeter"]["burned_cells_count"], 1500)
        self.assertEqual(scenario["initial_perimeter"]["rasterized_area_ha"], 1500.0)
        self.assertGreater(len(scenario["water_scooping_waypoints"]), 0)


if __name__ == "__main__":
    unittest.main()
