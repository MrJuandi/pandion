"""Structural checks for the optional standard-fuel fire path."""

import copy
import json
from pathlib import Path
import unittest

try:
    import pyretechnics  # noqa: F401
except ImportError:
    pyretechnics = None

from nightjar.fire_standard import run_fire_standard


@unittest.skipIf(pyretechnics is None, "install requirements-standard.txt")
class StandardFuelTests(unittest.TestCase):
    def setUp(self):
        self.scenario = json.loads(Path("examples/standard_fuel_fire.json").read_text())

    def test_example_is_deterministic_and_nonburnable_cell_is_blocked(self):
        result = run_fire_standard(self.scenario)
        self.assertEqual(result, run_fire_standard(self.scenario))
        self.assertIsNone(result["arrival_time_s"][2][2])
        self.assertGreater(result["affected_area_ha"], 0)

    def test_fuel_class_changes_arrival(self):
        self.scenario["horizon_s"] = 20000
        other = copy.deepcopy(self.scenario)
        other["fuel_model"] = [[101 if code else 0 for code in row] for row in other["fuel_model"]]
        self.assertNotEqual(run_fire_standard(self.scenario)["arrival_time_s"], run_fire_standard(other)["arrival_time_s"])

    def test_wind_reversal_flips_early_arrivals(self):
        self.scenario["horizon_s"] = 10000
        east = run_fire_standard(self.scenario)["arrival_time_s"]
        west_scenario = copy.deepcopy(self.scenario)
        west_scenario["midflame_wind_to_m_s"] = [-1, 0]
        west = run_fire_standard(west_scenario)["arrival_time_s"]
        self.assertLess(east[2][1], west[2][1])

    def test_rejects_unsupported_code(self):
        self.scenario["fuel_model"][0][0] = 999
        with self.assertRaisesRegex(ValueError, "fuel codes"):
            run_fire_standard(self.scenario)


if __name__ == "__main__":
    unittest.main()
