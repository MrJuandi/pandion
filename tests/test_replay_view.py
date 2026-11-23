"""Presentation boundaries and terrain alignment use small deterministic fixtures."""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pandion.replay_view import normalize_replay
from pandion.workspace import terrain_crop, static_map, decorate, build_workspace, load_policy
from pandion.geographic_family import build_synthetic_cems_fixture, generate_geographic_scenario


class ReplayViewTests(unittest.TestCase):
    def setUp(self):
        self.replay = {"scenario_id": "rect", "policy": "heuristic", "horizon_s": 120,
                       "grid": {"rows": 2, "cols": 3, "cell_size_m": 100},
                       "crs": {"code": "EPSG:25831"}, "origin_utm": [320000, 4600000],
                       "arrival_time_s": [[0, None, 30], [None, None, None]],
                       "fuel_grid": [[101, 0, 145], [101, 101, 101]],
                       "decisions": [{"time_s": 60, "decision_time_s": 0, "observation": {
                           "fire": {"cells": [[{"fire_detected": True}, None, {"fire_detected": False}]]},
                           "aircraft": {"position_m": [100, 0]}}}],
                       "drops": [{"time_s": 10, "target_local_m": [200, 0], "treated_cells": [[0, 2]]}]}

    def test_rectangular_geographic_coordinates_and_observations(self):
        original = copy.deepcopy(self.replay)
        out = normalize_replay(self.replay)
        self.assertEqual(out["grid"], self.replay["grid"])
        self.assertEqual(out["origin_m"], [320000, 4600000])
        self.assertEqual(out["crs"], "EPSG:25831")
        self.assertEqual(out["decisions"][0]["time_s"], 0)
        self.assertEqual(out["decisions"][0]["completed_at_s"], 60)
        self.assertEqual(out["decisions"][0]["observation"]["fire_reports"], [[0, 0]])
        self.assertEqual(out["drops"][0]["position_m"], [200, 0])
        self.assertEqual(out["drops"][0]["newly_blocked_cells"], [[0, 2]])
        self.assertFalse(out["water_mask"][0][1])  # nonburnable is not evidence of water
        self.assertEqual(self.replay, original)

    def test_aircraft_identity_survives_normalization_and_invalid_indices_fail(self):
        replay = copy.deepcopy(self.replay)
        replay["aircraft_track"] = [{"time_s": 0, "position_m": [0, 0], "aircraft_index": 0},
                                    {"time_s": 0, "position_m": [300, 0], "aircraft_index": 1}]
        replay["decisions"][0]["aircraft_index"] = 1
        replay["drops"][0]["aircraft_index"] = 1
        out = normalize_replay(replay)
        self.assertEqual([p["aircraft_index"] for p in out["aircraft_track"]], [0, 1])
        self.assertEqual(out["decisions"][0]["aircraft_index"], 1)
        self.assertEqual(out["drops"][0]["aircraft_index"], 1)
        replay["aircraft_track"][1]["aircraft_index"] = True
        with self.assertRaisesRegex(ValueError, "nonnegative integer"):
            normalize_replay(replay)

    def test_missing_resolution_and_arrival_are_rejected(self):
        for missing in ("grid", "arrival_time_s"):
            replay = copy.deepcopy(self.replay)
            replay.pop(missing)
            with self.assertRaises(ValueError):
                normalize_replay(replay)

    def test_invalid_raster_dimensions_and_nonfinite_values(self):
        for key, value in (("arrival_time_s", [[0]]), ("elevation_m", [[1, 2, float('nan')], [1, 2, 3]]),
                           ("water_mask", [[0, False, False], [False, False, False]])):
            with self.subTest(key=key), self.assertRaises(ValueError):
                normalize_replay({**self.replay, key: value})

    def test_synthetic_resolution_from_observation(self):
        r = copy.deepcopy(self.replay)
        r.pop("grid")
        r["decisions"][0]["observation"]["cell_size_m"] = 25
        self.assertEqual(normalize_replay(r)["grid"]["cell_size_m"], 25)

    def test_display_relief_does_not_mutate_simulator(self):
        replay = {**self.replay, "elevation_m": [[350]*3 for _ in range(2)],
                  "display_elevation_m": [[400, 420, 430], [500, 520, 530]],
                  "metadata": {"terrain": {"display_only": True, "label": "Capa visual"}}}
        out = normalize_replay(replay)
        self.assertEqual(out["elevation_m"][0], [400, 420, 430])
        self.assertEqual(replay["elevation_m"][0], [350, 350, 350])
        self.assertTrue(out["metadata"]["terrain"]["display_only"])


class WorkspaceTests(unittest.TestCase):
    def setUp(self):
        base = build_synthetic_cems_fixture("EMSR578", rows=6, cols=7)
        self.scenario = generate_geographic_scenario(base, 0, "development", window_rows=2, window_cols=3, horizon_s=120)
        grid = self.scenario["grid"]
        self.terrain = {"grid": {"rows": 4, "cols": 5, "cell_size_m": 100,
                                   "lower_left_origin_m": {"x": grid["origin_utm_easting_m"]-100.04,
                                                           "y": grid["origin_utm_northing_m"]-100.03}},
                        "spatial_reference": {"crs": "EPSG:25831"},
                        "elevation_m": [[10*r+c for c in range(5)] for r in range(4)],
                        "source": {"provider": "fixture"}, "coverage": {"status": "source_footprint_unknown"},
                        "scenario_sha256": "a"*64}

    def test_metric_crop_orientation_and_rounding_tolerance(self):
        self.assertEqual(terrain_crop(self.scenario, self.terrain), [[11, 12, 13], [21, 22, 23]])
        self.terrain["grid"]["lower_left_origin_m"]["x"] -= 1
        with self.assertRaisesRegex(ValueError, "align"):
            terrain_crop(self.scenario, self.terrain)

    def test_cross_crs_and_outside_coverage_rejected(self):
        bad = copy.deepcopy(self.terrain)
        bad["spatial_reference"]["crs"] = "EPSG:25830"
        with self.assertRaisesRegex(ValueError, "CRS"):
            terrain_crop(self.scenario, bad)
        bad = copy.deepcopy(self.terrain)
        bad["grid"]["rows"] = 2
        with self.assertRaisesRegex(ValueError, "outside"):
            terrain_crop(self.scenario, bad)
        other = {**self.scenario, "activation_id": "EMSR590"}
        self.assertIsNone(terrain_crop(other, self.terrain))

    def test_static_map_has_no_result_or_episode(self):
        out = normalize_replay(decorate(static_map(self.scenario), self.scenario))
        self.assertIsNone(out["affected_area_ha"])
        self.assertEqual(out["horizon_s"], 0)
        self.assertEqual(out["decisions"], [])
        self.assertEqual(out["metadata"]["split"], "development")

    def test_workspace_never_runs_validation_or_reserved(self):
        splits = {"development": [self.scenario], "validation": [{**self.scenario, "scenario_id": "val", "split": "validation"}],
                  "reserved": [{**self.scenario, "scenario_id": "res", "split": "reserved"}]}
        with tempfile.TemporaryDirectory() as tmp, \
             patch("pandion.workspace.load_geographic_split", side_effect=lambda split, *a, **k: splits[split]), \
             patch("pandion.workspace.run_geographic_integrated", return_value=static_map(self.scenario)) as simulator:
            root = Path(tmp)
            (root / "manifest.json").write_text('{}')
            report = build_workspace(root, root / "out", episodes=1)
            self.assertEqual(simulator.call_count, 3)
            self.assertTrue(all(call.args[0]["split"] == "development" for call in simulator.call_args_list))
            self.assertEqual(report["episode_splits"], ["development"])
            self.assertTrue((root / "out" / "index.html").exists())
            with self.assertRaisesRegex(ValueError, "new or empty"):
                build_workspace(root, root / "out")

    def test_legacy_training_report_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = Path(tmp) / "report.json"
            report.write_text('{"schema_version":"nightjar-geographic-ppo-train-0.1", "recurrent":true}')
            with self.assertRaisesRegex(ValueError, "does not match"):
                load_policy(Path(tmp)/"missing.pt", report, [self.scenario])

    def test_workspace_requires_paired_artifacts(self):
        for kwargs in ({"terrain_path": Path("terrain")}, {"checkpoint": Path("weights")}):
            with self.assertRaises(ValueError):
                build_workspace(Path("missing"), Path("missing-output"), **kwargs)
