"""Unit tests for physical fire spread simulation and satellite evaluation (pilot_simulation)."""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
import sys
import unittest
from unittest.mock import patch
from zipfile import ZipFile

import numpy as np

from nightjar import pilot_simulation
from nightjar.pilot_terrain import build_terrain


class PilotSimulationUnitTests(unittest.TestCase):
    def setUp(self):
        # 5x5 test scenario with cell size 100m (1 ha/cell)
        # Center cell (2, 2) is ignited; cell (2, 4) is nonburnable (0)
        self.scenario = {
            "schema_version": "nightjar-emsr578-pilot-0.1",
            "scenario_id": "test-pilot-5x5",
            "spatial_reference": {
                "crs": "EPSG:25831",
                "bounds_utm31n": {"min_x": 0.0, "max_x": 500.0, "min_y": 0.0, "max_y": 500.0},
            },
            "grid": {
                "rows": 5,
                "cols": 5,
                "cell_size_m": 100.0,
                "cell_area_ha": 1.0,
            },
            "fuel_model": [
                [145, 145, 145, 145, 145],
                [145, 145, 145, 145, 145],
                [145, 145, 145, 145,   0],  # (2, 4) is nonburnable
                [145, 145, 145, 145, 145],
                [145, 145, 145, 145, 145],
            ],
            "initial_burn_mask": [
                [False, False, False, False, False],
                [False, False, False, False, False],
                [False, False,  True, False, False],  # (2, 2) ignited
                [False, False, False, False, False],
                [False, False, False, False, False],
            ],
        }

    def test_initial_cell_starts_at_zero(self):
        res = pilot_simulation.simulate_pilot_spread(self.scenario, horizon_s=600.0)
        self.assertEqual(res["arrival_time_s"][2][2], 0.0)

    def test_nonburnable_cell_remains_unburned(self):
        res = pilot_simulation.simulate_pilot_spread(self.scenario, horizon_s=3600.0)
        # Cell (2, 4) is nonburnable
        self.assertIsNone(res["arrival_time_s"][2][4])
        self.assertFalse(res["simulated_burn_mask"][2][4])

    def test_wind_pushes_downwind_faster(self):
        # Wind blowing directly North (wind_to: [0, 5.0])
        res = pilot_simulation.simulate_pilot_spread(
            self.scenario,
            horizon_s=3600.0,
            wind_to_m_s=(0.0, 5.0),
        )
        t_north = res["arrival_time_s"][3][2]  # North neighbor (row 3)
        t_south = res["arrival_time_s"][1][2]  # South neighbor (row 1)
        self.assertIsNotNone(t_north)
        self.assertIsNotNone(t_south)
        self.assertLess(t_north, t_south, "Fire must reach downwind (North) neighbor faster than upwind (South)")

    def test_area_increases_monotonically_with_horizon(self):
        res_short = pilot_simulation.simulate_pilot_spread(self.scenario, horizon_s=600.0)
        res_long = pilot_simulation.simulate_pilot_spread(self.scenario, horizon_s=3600.0)
        self.assertLessEqual(res_short["simulated_fire_cells"], res_long["simulated_fire_cells"])
        self.assertLessEqual(res_short["simulated_fire_ha"], res_long["simulated_fire_ha"])

    def test_invalid_parameters_raise_value_error(self):
        with self.assertRaises(ValueError):
            pilot_simulation.simulate_pilot_spread(self.scenario, horizon_s=-10.0)
        with self.assertRaises(ValueError):
            pilot_simulation.simulate_pilot_spread(self.scenario, wind_to_m_s=[1.0])
        with self.assertRaises(ValueError):
            pilot_simulation.simulate_pilot_spread(self.scenario, elevation_m=[[10.0]])
        with self.assertRaisesRegex(ValueError, "finite elevations"):
            pilot_simulation.simulate_pilot_spread(self.scenario, elevation_m=[[float("nan")] * 5 for _ in range(5)])

    def test_northward_uphill_spread_and_report_provenance(self):
        result = pilot_simulation.simulate_pilot_spread(
            self.scenario, horizon_s=3600, wind_to_m_s=(0, 0),
            elevation_m=[[20.0 * r] * 5 for r in range(5)],
        )
        self.assertLess(result["arrival_time_s"][3][2], result["arrival_time_s"][1][2])
        self.assertEqual(result["units"]["area"], "ha")
        self.assertIsNone(result["seed"])
        self.assertEqual(len(result["scenario_sha256"]), 64)

    def test_cli_uses_bound_terrain_and_rejects_missing_cells(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            scen = root / "scenario.json"
            scen.write_text(json.dumps(self.scenario))
            source = root / "terrain.asc"
            source.write_text("ncols 5\nnrows 5\nxllcorner 0\nyllcorner 0\ncellsize 100\n" + "\n".join(" ".join([str(20*r)]*5) for r in reversed(range(5))) + "\n")
            terrain_path = root / "terrain.json"
            terrain = build_terrain(self.scenario, source, crs="EPSG:25831")
            terrain_path.write_text(json.dumps(terrain))
            output = root / "simulation.json"
            args = ["pilot_simulation", "run", "--scenario", str(scen), "--terrain", str(terrain_path), "--output", str(output)]
            with patch.object(sys, "argv", args):
                pilot_simulation.main()
            result = json.loads(output.read_text())
            self.assertEqual(result["terrain"]["artifact_sha256"], sha256(terrain_path.read_bytes()).hexdigest())
            terrain["elevation_m"][0][0] = None
            terrain["valid_mask"][0][0] = False
            terrain_path.write_text(json.dumps(terrain))
            with patch.object(sys, "argv", args), self.assertRaisesRegex(ValueError, "missing elevations"):
                pilot_simulation.main()

    def test_evaluation_separates_growth_and_flags_time_mismatch(self):
        initial = np.array(self.scenario["initial_burn_mask"], dtype=bool)
        observed = initial.copy()
        observed[3][2] = True
        simulation = {"grid": self.scenario["grid"], "simulated_burn_mask": initial.tolist(), "parameters": {"horizon_s": 3600}}
        self.scenario["observation_timestamps"] = {"t0_initial_utc": "2022-06-17T10:06:00Z"}
        with TemporaryDirectory() as directory:
            source = Path(directory) / "mock.zip"
            source.write_bytes(b"synthetic comparison bytes")
            digest = sha256(source.read_bytes()).hexdigest()
            with patch.object(pilot_simulation, "extract_observed_mask", return_value=observed), patch.object(
                pilot_simulation, "PRODUCTS", (("mock", "MOCK", "2022-06-18T10:47:00Z", digest),)
            ):
                result = pilot_simulation.evaluate_simulation_vs_satellite(simulation, source, self.scenario)
            self.assertEqual(result["timing"]["elapsed_observation_s"], 88860)
            self.assertEqual(result["timing"]["status"], "mismatched_elapsed_time")
            self.assertEqual(result["iou"], 0.5)
            self.assertEqual(result["initial_only_baseline"]["iou"], 0.5)
            self.assertEqual(result["growth_only"]["iou"], 0)


class PilotSimulationIntegrationTests(unittest.TestCase):
    def test_real_scenario_simulation_if_exists(self):
        scenario_path = Path("runs/emsr578-pilot-scenario.json")
        if not scenario_path.exists():
            self.skipTest("runs/emsr578-pilot-scenario.json not found on disk")

        scenario = json.loads(scenario_path.read_text())
        sim = pilot_simulation.simulate_pilot_spread(scenario, horizon_s=3600.0)

        self.assertEqual(sim["schema_version"], "nightjar-pilot-simulation-0.1")
        self.assertEqual(sim["scenario_crs"], "EPSG:25831")
        self.assertGreater(sim["simulated_fire_ha"], sim["initial_fire_ha"])
        self.assertGreater(sim["computation_time_s"], 0.0)

        # Check satellite evaluation if monit01.zip exists
        monit01_path = Path("data/raw/emsr578/monit01.zip")
        if monit01_path.exists():
            metrics = pilot_simulation.evaluate_simulation_vs_satellite(sim, monit01_path, scenario)
            self.assertGreater(metrics["iou"], 0.0)
            self.assertGreater(metrics["dice"], 0.0)
            self.assertGreater(metrics["intersection_area_ha"], 0.0)


if __name__ == "__main__":
    unittest.main()
