import copy
import json
from pathlib import Path
import unittest

from nightjar.fire_rothermel import run_fire_rothermel
from nightjar.integrated import run_integrated
from nightjar.integrated_benchmark import compare_policies
from nightjar.aircraft import CAPACITY_L


EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "synthetic_integrated.json"


class IntegratedEpisodeTests(unittest.TestCase):
    def setUp(self):
        self.scenario = json.loads(EXAMPLE.read_text())

    def test_no_intervention_reproduces_fire_solver(self):
        replay = run_integrated(self.scenario, "no_intervention")
        fire = run_fire_rothermel(self.scenario["fire"])
        within_horizon = [[value if value is not None and value <= self.scenario["fire"]["horizon_s"] else None
                           for value in row] for row in fire["arrival_time_s"]]
        self.assertEqual(replay["arrival_time_s"], within_horizon)
        self.assertEqual(replay["affected_area_ha"], fire["affected_area_ha"])
        self.assertEqual(replay["drops"], [])

    def test_water_blocks_only_future_unburned_cells(self):
        baseline = run_integrated(self.scenario, "no_intervention")
        intervention = run_integrated(self.scenario, "heuristic")
        self.assertGreater(len(intervention["drops"]), 0)
        self.assertLess(intervention["affected_area_ha"], baseline["affected_area_ha"])
        self.assertEqual(intervention["arrival_time_s"][5][5], 0)
        self.assertEqual(intervention["aircraft_metrics"]["water_dropped_l"], len(intervention["drops"]) * CAPACITY_L)
        self.assertEqual(intervention["aircraft_metrics"]["rejected_orders"], 0)
        for drop in intervention["drops"]:
            for r, c in drop["newly_blocked_cells"]:
                self.assertIsNone(intervention["arrival_time_s"][r][c])
        self.assertEqual(intervention, run_integrated(copy.deepcopy(self.scenario), "heuristic"))

    def test_delayed_report_prevents_early_drop(self):
        self.scenario["observation_schedule"] = [{
            "source_id": "delayed", "observed_at_s": 0, "available_at_s": 120,
            "coverage_cells": [[r, c] for r in range(11) for c in range(11)],
        }]
        replay = run_integrated(self.scenario, "heuristic")
        self.assertEqual([decision["order"]["type"] for decision in replay["decisions"][:2]], ["HOLD", "HOLD"])
        self.assertTrue(all(cell is None for row in replay["decisions"][0]["observation"]["fire"]["cells"] for cell in row))

    def test_invalid_effect_and_changed_seed_rejected(self):
        self.scenario["water_effect"]["radius_m"] = -1
        with self.assertRaises(ValueError):
            run_integrated(self.scenario, "heuristic")
        self.scenario["water_effect"]["radius_m"] = 55
        self.scenario["aircraft"]["seed"] = 43
        with self.assertRaises(ValueError):
            run_integrated(self.scenario, "heuristic")

    def test_comparison_uses_identical_scenario_and_fixed_random_seeds(self):
        result = compare_policies(self.scenario, (1, 2, 3))
        self.assertEqual([run["policy_seed"] for run in result["random_valid"]["runs"]], [1, 2, 3])
        self.assertEqual(result, compare_policies(copy.deepcopy(self.scenario), (1, 2, 3)))
        self.assertEqual(result["no_intervention"]["drops"], 0)

    def test_documented_synthetic_comparison(self):
        result = compare_policies(self.scenario)
        self.assertEqual(result["no_intervention"]["affected_area_ha"], 2.0)
        self.assertEqual(result["heuristic"]["affected_area_ha"], 0.75)
        self.assertEqual(result["random_valid"]["mean_affected_area_ha"], 1.3)


if __name__ == "__main__":
    unittest.main()
