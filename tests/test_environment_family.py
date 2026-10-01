"""Tests for strict terrain enrichment of the verified geographic family."""

from copy import deepcopy
import json
from hashlib import sha256
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from nightjar.environment_family import (
    SCENARIO_SCHEMA_VERSION,
    SCHEMA_VERSION,
    _assert_family_grids,
    _read_dem,
    _validate_grid,
    build_environment_family,
)
from nightjar.geographic_family import canonical_hash


def _scenario(activation: str, split: str, index: int = 0) -> dict:
    crs = {"EMSR578": "EPSG:25831", "EMSR600": "EPSG:25832"}.get(activation, "EPSG:25830")
    scenario = {
        "schema_version": "nightjar-geographic-scenario-0.1",
        "scenario_id": f"geo-{activation.lower()}-{index}",
        "activation_id": activation,
        "split": split,
        "seed": 100 + index,
        "source_kind": "copernicus_cems",
        "crs": {"authority": "EPSG", "code": crs,
                "utm_zone": {"EPSG:25831": 31, "EPSG:25830": 30, "EPSG:25832": 32}[crs],
                "horizontal_datum": "ETRS89"},
        "grid": {"rows": 2, "cols": 3, "cell_size_m": 100.0,
                 "origin_utm_easting_m": 1000.0, "origin_utm_northing_m": 2000.0},
        "fire": {"fuel_grid": [[101] * 3 for _ in range(2)], "wind_to_m_s": [1.5, 1.5],
                 "horizon_s": 3600.0,
                 "moisture": {"dead_1h": 0.04, "dead_10h": 0.05, "dead_100h": 0.07,
                              "dead_herbaceous": 0.04, "live_herbaceous": 0.3, "live_woody": 0.6}},
        "provenance": {"source_zip_sha256": "a" * 64,
                       "image_acquired_at_utc": "2022-06-17T10:06:00Z",
                       "assumptions": {"topography": "flat", "wind": "synthetic",
                                       "moisture": "synthetic", "survey_schedule": "synthetic"}},
    }
    scenario["scenario_hash"] = canonical_hash(scenario)
    return scenario


def _write_ascii(path: Path, rows: list[list[float]], *, x: float = 1000, y: float = 2000,
                 cellsize: float = 100, nodata: int | None = None) -> None:
    lines = [f"ncols {len(rows[0])}", f"nrows {len(rows)}", f"xllcorner {x}", f"yllcorner {y}", f"cellsize {cellsize}"]
    if nodata is not None:
        lines.append(f"NODATA_value {nodata}")
    # Input ESRI rows are north-first.
    lines.extend(" ".join(map(str, row)) for row in rows)
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


class TestEnvironmentFamily(unittest.TestCase):
    def test_ascii_crop_orientation_zero_and_metadata(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "emsr578.asc"
            _write_ascii(path, [[4, 5, 6], [0, 2, 3]])
            elevation, source = _read_dem(path, (2, 3, 100.0, 1000.0, 2000.0, "EPSG:25831"), "EMSR578")
        self.assertEqual(elevation, [[0.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        self.assertEqual(source["provider"], "IGN/CNIG")
        self.assertEqual(source["dataset"], "Elevacion25830_25")
        self.assertEqual(source["native_resolution_m"], 25)
        self.assertEqual(source["source_footprint"], "unknown")
        self.assertEqual(source["crs"], "EPSG:25831")
        self.assertIn("bbox=1000.0%2C2000.0%2C1300.0%2C2200.0", source["url"])

    def test_dem_rejects_alignment_nodata_and_missing_file(self):
        grid = (2, 3, 100.0, 1000.0, 2000.0, "EPSG:25831")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            misaligned = root / "misaligned.asc"
            _write_ascii(misaligned, [[1, 2, 3], [4, 5, 6]], x=1000.11)
            with self.assertRaisesRegex(ValueError, "does not align"):
                _read_dem(misaligned, grid, "EMSR578")
            nodata = root / "nodata.asc"
            _write_ascii(nodata, [[1, 2, 3], [4, -9999, 6]], nodata=-9999)
            with self.assertRaisesRegex(ValueError, "missing or nonfinite"):
                _read_dem(nodata, grid, "EMSR578")
            with self.assertRaises(FileNotFoundError):
                _read_dem(root / "absent.asc", grid, "EMSR578")

    def test_enrichment_preserves_splits_and_parent_hashes(self):
        splits = {
            "development": [_scenario("EMSR578", "development", 0), _scenario("EMSR578", "development", 1)],
            "validation": [_scenario("EMSR590", "validation")],
            "reserved": [_scenario("EMSR592", "reserved"), _scenario("EMSR600", "reserved")],
        }
        originals = deepcopy(splits)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            input_dir, dem_dir, out_dir = root / "source", root / "dem", root / "out"
            input_dir.mkdir()
            dem_dir.mkdir()
            manifest = {"schema_version": "nightjar-geographic-family-0.1", "source_kind": "copernicus_cems",
                        "counts": {"development": 2, "validation": 1, "reserved": 2, "total": 5},
                        "cems_identity": "preserved", "scenarios": {}}
            (input_dir / "manifest.json").write_text(json.dumps(manifest))
            for filename in ("emsr578.asc", "emsr590.asc", "emsr592.asc", "emsr600.asc"):
                _write_ascii(dem_dir / filename, [[10, 11, 12], [0, 8, 9]])
            with patch("nightjar.environment_family.load_geographic_split",
                       side_effect=lambda split, *_a, **_kw: deepcopy(splits[split])):
                result = build_environment_family(input_dir, dem_dir, out_dir)
            saved_manifest = json.loads((out_dir / "manifest.json").read_text())
        self.assertEqual(result["manifest"]["schema_version"], SCHEMA_VERSION)
        self.assertEqual(result["manifest"]["cems_identity"], "preserved")
        self.assertEqual({key: len(value) for key, value in result["scenarios"].items()},
                         {key: len(value) for key, value in originals.items()})
        for split, scenarios in originals.items():
            for index, original in enumerate(scenarios):
                actual = result["scenarios"][split][index]
                self.assertEqual(actual["scenario_id"], original["scenario_id"])
                self.assertEqual(actual["scenario_hash"], canonical_hash({k: v for k, v in actual.items() if k != "scenario_hash"}))
                self.assertEqual(actual["provenance"]["environment_parent_scenario_hash"], original["scenario_hash"])
                self.assertEqual(actual["schema_version"], SCENARIO_SCHEMA_VERSION)
                self.assertEqual(actual["fire"]["elevation_m"], [[0, 8, 9], [10, 11, 12]])
                self.assertIn("terrain", actual["environment"])
        self.assertEqual([s["scenario_id"] for s in result["scenarios"]["development"]],
                         [s["scenario_id"] for s in originals["development"]])
        self.assertEqual(result["scenarios"]["validation"][0]["fire"]["wind_to_m_s"],
                         originals["validation"][0]["fire"]["wind_to_m_s"])
        self.assertEqual(saved_manifest["scenarios"]["reserved"][0]["source_zip_sha256"], "a" * 64)

    def test_per_incident_sources_preserve_mixed_provider_and_crs_metadata(self):
        splits = {
            "development": [_scenario("EMSR578", "development")],
            "validation": [_scenario("EMSR590", "validation")],
            "reserved": [_scenario("EMSR592", "reserved"), _scenario("EMSR600", "reserved")],
        }
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            input_dir, dem_dir, out_dir = root / "source", root / "dem", root / "out"
            input_dir.mkdir()
            dem_dir.mkdir()
            (input_dir / "manifest.json").write_text(json.dumps({
                "schema_version": "nightjar-geographic-family-0.1", "source_kind": "copernicus_cems",
                "counts": {"development": 1, "validation": 1, "reserved": 2, "total": 4},
                "scenarios": {},
            }))
            rows = [[1, 2, 3], [4, 5, 6]]
            sources = {}
            for index, (activation, filename) in enumerate((
                ("EMSR578", "emsr578.asc"), ("EMSR590", "emsr590.asc"),
                ("EMSR592", "emsr592.asc"), ("EMSR600", "emsr600.asc"),
            )):
                path = dem_dir / filename
                _write_ascii(path, rows)
                sources[activation] = {
                    "raw_sha256": sha256(path.read_bytes()).hexdigest(),
                    "provider": "IGN/CNIG" if activation == "EMSR578" else "Copernicus DEM",
                    "dataset": "Elevation 25m" if activation == "EMSR578" else f"GLO-30-{activation}",
                    "native_resolution_m": 25 if activation == "EMSR578" else 30,
                    "vertical_reference": "orthometric",
                    "acquisition_epoch": "unknown",
                    "source_footprint": "unknown",
                    "crs": {"EMSR578": "EPSG:25831", "EMSR590": "EPSG:25830",
                            "EMSR592": "EPSG:25830", "EMSR600": "EPSG:25832"}[activation],
                    "url": f"https://example.invalid/{activation}",
                    "resampling": "nearest-neighbor",
                }
            (dem_dir / "sources.json").write_text(json.dumps(sources))
            with patch("nightjar.environment_family.load_geographic_split",
                       side_effect=lambda split, *_a, **_kw: deepcopy(splits[split])):
                result = build_environment_family(input_dir, dem_dir, out_dir)
            saved_manifest = json.loads((out_dir / "manifest.json").read_text())
        terrain = saved_manifest["environment"]
        self.assertEqual(terrain["terrain_provider"], "mixed")
        self.assertEqual(terrain["terrain_dataset"], "mixed")
        self.assertIsNone(terrain["native_resolution_m"])
        self.assertEqual(terrain["native_resolutions_m"], [25, 30])
        for split, scenarios in result["scenarios"].items():
            for scenario in scenarios:
                activation = scenario["activation_id"]
                metadata = scenario["environment"]["terrain"]["source"]
                for key, value in sources[activation].items():
                    self.assertEqual(metadata[key], value)
                self.assertIn("crs_assertion", metadata)
                self.assertEqual(metadata["crs"], scenario["crs"]["code"])

    def test_sources_manifest_hash_and_crs_must_match_ascii_and_scenario(self):
        grid = (2, 3, 100.0, 1000.0, 2000.0, "EPSG:25831")
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "dem.asc"
            _write_ascii(path, [[1, 2, 3], [4, 5, 6]])
            valid = {"raw_sha256": sha256(path.read_bytes()).hexdigest(), "crs": "EPSG:25831"}
            with self.assertRaisesRegex(ValueError, "raw_sha256 does not match"):
                _read_dem(path, grid, "EMSR578", {**valid, "raw_sha256": "0" * 64})
            with self.assertRaisesRegex(ValueError, "source CRS does not match"):
                _read_dem(path, grid, "EMSR578", {**valid, "crs": "EPSG:25830"})

    def test_station_environment_contract_is_scoped_and_causal(self):
        from nightjar.environment_family import _enrich_scenario
        from nightjar.pilot_terrain import scenario_sha256

        dev = _scenario("EMSR578", "development")
        origin = "2022-06-17T10:06:00Z"
        records = [
            {"interval_start_utc": "2022-06-17T10:06:00Z", "interval_end_utc": "2022-06-17T10:36:00Z",
             "value": 1.0, "quality_code": "V", "usable": True},
            {"interval_start_utc": "2022-06-17T10:36:00Z", "interval_end_utc": "2022-06-17T11:06:00Z",
             "value": 1.0, "quality_code": "V", "usable": True},
        ]
        variables = {
            "air_temperature": {"unit": "°C", "records": [{**r, "value": 24.0} for r in records]},
            "relative_humidity": {"unit": "%", "records": [{**r, "value": 45.0} for r in records]},
            "precipitation": {"unit": "mm", "records": [{**r, "value": 0.0} for r in records]},
            "wind_speed_at_2m": {"unit": "m/s", "records": [{**r, "value": 4.0} for r in records]},
            "wind_direction_at_2m": {"unit": "°", "records": [{**r, "value": 180.0} for r in records]},
        }
        audit = {
            "schema_version": "nightjar-pilot-weather-audit-0.1", "scenario_id": dev["scenario_id"],
            "scenario_sha256": scenario_sha256(dev), "station": {"code": "X6", "name": "Baldomar"},
            "window": {"start_utc": origin}, "variables": variables, "sources": {"provider": "Meteocat"},
        }
        terrain = {"source": {"raw_sha256": "b" * 64}, "units": "m"}
        enriched = _enrich_scenario(dev, [[1, 2, 3], [4, 5, 6]], terrain["source"],
                                    audit=audit, pilot_scenario=dev, station_wind_factor=1.25,
                                    moisture_mode="emc_lag")
        self.assertEqual(enriched["environment"]["weather_mode"], "station_intervals")
        self.assertEqual(len(enriched["environment"]["forcing"]), 2)
        self.assertEqual(enriched["environment"]["forcing"][0]["available_at_s"], 1800.0)
        self.assertEqual(enriched["provenance"]["assumptions"]["wind"], "station_2m_proxy_time_varying")
        self.assertEqual(enriched["provenance"]["assumptions"]["moisture"], "modeled_emc_lag")

        other = _scenario("EMSR590", "validation")
        synthetic_env = _enrich_scenario(other, [[1, 2, 3], [4, 5, 6]], terrain["source"],
                                         audit=audit, pilot_scenario=dev, station_wind_factor=1.25,
                                         moisture_mode="emc_lag")
        self.assertEqual(synthetic_env["environment"]["weather_mode"], "constant_synthetic")
        self.assertEqual(synthetic_env["environment"]["forcing"], [])
        self.assertEqual(synthetic_env["fire"]["wind_to_m_s"], other["fire"]["wind_to_m_s"])

    def test_variant_grid_mismatch_prevents_crop_reuse(self):
        dev = [_scenario("EMSR578", "development", 0), _scenario("EMSR578", "development", 1)]
        dev[1]["grid"]["origin_utm_northing_m"] += 0.2
        with self.assertRaisesRegex(ValueError, "variant grid metadata disagree"):
            _assert_family_grids(dev, "EMSR578")

    def test_real_family_grid_metadata_and_40_by_40_ascii_fixture(self):
        # Mirrors geographic-family 0.1 metadata (full EPSG authority/code,
        # 40x40 cells, 100 m, rounded tactical crop origin) without depending
        # on local CEMS downloads or generated run artifacts.
        scenario = _scenario("EMSR578", "development")
        scenario["grid"].update({"rows": 40, "cols": 40, "origin_utm_easting_m": 331136.44,
                                 "origin_utm_northing_m": 4642171.57})
        expected = (40, 40, 100.0, 331136.44, 4642171.57, "EPSG:25831")
        self.assertEqual(_validate_grid(scenario, "EMSR578"), expected)
        self.assertEqual(_assert_family_grids([scenario, deepcopy(scenario)], "EMSR578"), expected)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "emsr578.asc"
            rows = [[float(r * 40 + c) for c in range(40)] for r in range(40)]
            _write_ascii(path, rows, x=331136.4, y=4642171.6)
            elevation, source = _read_dem(path, expected, "EMSR578")
        self.assertEqual((len(elevation), len(elevation[0])), (40, 40))
        self.assertEqual(elevation[0][0], float(39 * 40))
        self.assertEqual(elevation[-1], [float(c) for c in range(40)])
        self.assertIn("crs=EPSG%3A25831", source["url"])

    def test_output_must_be_new_or_empty(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            out = root / "output"
            out.mkdir()
            (out / "keep.txt").write_text("keep")
            with self.assertRaisesRegex(FileExistsError, "Refusing to mix"):
                build_environment_family(root / "source", root / "dem", out)
            self.assertEqual((out / "keep.txt").read_text(), "keep")


if __name__ == "__main__":
    unittest.main()
