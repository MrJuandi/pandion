import copy
import json
from pathlib import Path
import unittest

from nightjar.fire_rothermel import run_fire_rothermel, validate_scenario
from nightjar.integrated import run_integrated


ROOT = Path(__file__).resolve().parents[1]


class CuredGrassTests(unittest.TestCase):
    def setUp(self):
        self.scenario = json.loads((ROOT / "examples/cured_grass_fire.json").read_text())

    def test_cured_grass_spreads_with_explicit_properties(self):
        replay = run_fire_rothermel(self.scenario)
        self.assertEqual(replay["replay_schema_version"], "rothermel-fuels-0.1")
        self.assertEqual(replay["arrival_time_s"][2][2], 0)
        self.assertGreater(replay["arrival_time_s"][2][3], 0)
        self.assertEqual(replay, run_fire_rothermel(copy.deepcopy(self.scenario)))

    def test_partial_curing_and_missing_properties_are_rejected(self):
        self.scenario["fuel_properties"]["CURED_GRASS"]["curing_percent"] = 95
        with self.assertRaisesRegex(ValueError, "curing_percent=100"):
            validate_scenario(self.scenario)
        self.scenario["fuel_properties"]["CURED_GRASS"]["curing_percent"] = 100
        del self.scenario["fuel_properties"]["CURED_GRASS"]["sav_cm_inv"]
        with self.assertRaisesRegex(ValueError, "explicit properties"):
            validate_scenario(self.scenario)

    def test_integrated_no_intervention_matches_fire_solver(self):
        integrated = json.loads((ROOT / "examples/synthetic_integrated.json").read_text())
        fire = integrated["fire"]
        fire["schema_version"] = "rothermel-fuels-0.1"
        fire["fuel_model"] = [["CURED_GRASS" for _ in row] for row in fire["fuel_model"]]
        fire["fuel_properties"] = self.scenario["fuel_properties"]
        full = run_fire_rothermel(fire)
        episode = run_integrated(integrated, "no_intervention")
        within_horizon = [[value if value is not None and value <= fire["horizon_s"] else None
                           for value in row] for row in full["arrival_time_s"]]
        self.assertEqual(episode["arrival_time_s"], within_horizon)
        self.assertEqual(episode["affected_area_ha"], full["affected_area_ha"])


if __name__ == "__main__":
    unittest.main()
