from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from pandion.pilot_terrain import build_terrain, load_terrain, scenario_sha256


class PilotTerrainTests(unittest.TestCase):
    def setUp(self):
        self.scenario = {
            "schema_version": "nightjar-emsr578-pilot-0.1",
            "scenario_id": "fixture",
            "spatial_reference": {
                "crs": "EPSG:25831",
                "bounds_utm31n": {"min_x": 1000.0, "max_x": 1200.0, "min_y": 2000.0, "max_y": 2200.0},
            },
            "grid": {"rows": 2, "cols": 2, "cell_size_m": 100.0},
        }
        self.header = "ncols 2\nnrows 2\nxllcorner 1000\nyllcorner 2000\ncellsize 100\n"

    def _write(self, directory: str, name: str, content: str) -> Path:
        path = Path(directory) / name
        path.write_text(content, encoding="ascii")
        return path

    def test_reverses_north_first_rows_and_builds_provenance(self):
        with TemporaryDirectory() as directory:
            source = self._write(directory, "grid.asc", self.header + "30 40\n10 20\n")
            artifact = build_terrain(
                self.scenario, source, crs="EPSG:25831", source_url="https://example.test/grid",
                provider="IGN", dataset="DTM", native_resolution_m=25,
                vertical_reference="orthometric", resampling="nearest-neighbor",
            )
            self.assertEqual(artifact["schema_version"], "nightjar-pilot-terrain-0.1")
            self.assertEqual(artifact["elevation_m"], [[10.0, 20.0], [30.0, 40.0]])
            self.assertEqual(artifact["valid_mask"], [[True, True], [True, True]])
            self.assertEqual(artifact["seed"], None)
            self.assertEqual(artifact["scenario_sha256"], scenario_sha256(self.scenario))
            self.assertEqual(artifact["source"]["raw_sha256"], sha256(source.read_bytes()).hexdigest())
            self.assertEqual(artifact["source"]["provider"], "IGN")
            self.assertEqual(artifact["coverage"]["status"], "source_footprint_unknown")
            self.assertEqual(artifact["spatial_reference"]["crs"], "EPSG:25831")

    def test_nodata_is_null_and_zero_remains_valid(self):
        with TemporaryDirectory() as directory:
            source = self._write(directory, "nodata.asc", self.header + "NODATA_value -9999\n0 -9999\n5 6\n")
            artifact = build_terrain(self.scenario, source, crs="EPSG:25831")
            self.assertEqual(artifact["elevation_m"], [[5.0, 6.0], [0.0, None]])
            self.assertEqual(artifact["valid_mask"], [[True, True], [True, False]])
            self.assertEqual(artifact["coverage"], {"status": "nodata_encoded", "nodata_value": -9999.0})

    def test_load_validates_hash_and_mask(self):
        with TemporaryDirectory() as directory:
            source = self._write(directory, "grid.asc", self.header + "3 4\n1 2\n")
            artifact = build_terrain(self.scenario, source, crs="EPSG:25831")
            artifact_path = Path(directory) / "artifact.json"
            artifact_path.write_text(json.dumps(artifact), encoding="utf-8")
            self.assertEqual(load_terrain(artifact_path, self.scenario), artifact)
            changed_scenario = json.loads(json.dumps(self.scenario))
            changed_scenario["scenario_id"] = "changed"
            with self.assertRaisesRegex(ValueError, "scenario_sha256"):
                load_terrain(artifact_path, changed_scenario)
            artifact["valid_mask"][0][0] = False
            artifact_path.write_text(json.dumps(artifact), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "disagrees"):
                load_terrain(artifact_path, self.scenario)

    def test_rejects_alignment_shape_and_crs_errors(self):
        with TemporaryDirectory() as directory:
            source = self._write(directory, "grid.asc", self.header + "3 4\n1 2\n")
            with self.assertRaisesRegex(ValueError, "only caller-asserted"):
                build_terrain(self.scenario, source, crs="EPSG:25830")
            shifted = self._write(directory, "shifted.asc", self.header.replace("1000", "1000.1", 1) + "3 4\n1 2\n")
            with self.assertRaisesRegex(ValueError, "xllcorner"):
                build_terrain(self.scenario, shifted, crs="EPSG:25831")
            wrong_shape = self._write(directory, "shape.asc", "ncols 1\nnrows 2\nxllcorner 1000\nyllcorner 2000\ncellsize 100\n3\n1\n")
            with self.assertRaisesRegex(ValueError, "dimensions"):
                build_terrain(self.scenario, wrong_shape, crs="EPSG:25831")
            wrong_scenario = json.loads(json.dumps(self.scenario))
            wrong_scenario["spatial_reference"]["crs"] = "EPSG:4326"
            with self.assertRaisesRegex(ValueError, "scenario CRS"):
                build_terrain(wrong_scenario, source, crs="EPSG:25831")

    def test_rejects_malformed_headers_and_raster_values(self):
        malformed = {
            "duplicate.asc": self.header + "ncols 2\n1 2\n3 4\n",
            "unknown.asc": self.header + "datum 9\n1 2\n3 4\n",
            "missing.asc": "ncols 2\nnrows 2\nxllcorner 1000\nyllcorner 2000\n1 2\n3 4\n",
            "short.asc": self.header + "1 2\n",
            "extra.asc": self.header + "1 2\n3 4\n5 6\n",
            "few-values.asc": self.header + "1 2\n3\n",
            "many-values.asc": self.header + "1 2 3\n3 4\n",
            "nan.asc": self.header + "1 NaN\n3 4\n",
            "infinity.asc": self.header + "1 2\n3 inf\n",
        }
        with TemporaryDirectory() as directory:
            for name, content in malformed.items():
                with self.subTest(name=name):
                    source = self._write(directory, name, content)
                    with self.assertRaises(ValueError):
                        build_terrain(self.scenario, source, crs="EPSG:25831")


if __name__ == "__main__":
    unittest.main()
