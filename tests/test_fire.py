import copy
import json
import unittest
from pathlib import Path

from nightjar.fire import run_fire


EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "synthetic_fire.json"


class SyntheticFireTests(unittest.TestCase):
    def setUp(self):
        self.scenario = json.loads(EXAMPLE.read_text())

    def test_repeatable_and_downwind_faster(self):
        result = run_fire(self.scenario)
        self.assertEqual(result, run_fire(copy.deepcopy(self.scenario)))
        times = result["arrival_time_s"]
        self.assertEqual(times[2][2], 0)
        self.assertLess(times[2][3], times[2][1])

    def test_flat_calm_rate_and_diagonal_distance(self):
        self.scenario["wind_to_m_s"] = [0, 0]
        result = run_fire(self.scenario)["arrival_time_s"]
        self.assertEqual(result[2][3], 200)
        self.assertAlmostEqual(result[3][3], 100 * 2**0.5 / 0.5, places=5)
        self.assertEqual(result[2][1], result[2][3])

    def test_upslope_faster_and_wind_reversal(self):
        self.scenario["wind_to_m_s"] = [0, 0]
        self.scenario["elevation_m"][2] = [0, 0, 0, 10, 20]
        sloped = run_fire(self.scenario)["arrival_time_s"]
        self.assertLess(sloped[2][3], sloped[2][1])
        self.scenario["elevation_m"][2] = [0] * 5
        self.scenario["wind_to_m_s"] = [-5, 0]
        reversed_wind = run_fire(self.scenario)["arrival_time_s"]
        self.assertLess(reversed_wind[2][1], reversed_wind[2][3])

    def test_zero_rate_is_barrier(self):
        self.scenario["base_ros_m_s"][2][3] = 0
        result = run_fire(self.scenario)["arrival_time_s"]
        self.assertIsNone(result[2][3])

    def test_diagonal_cannot_cross_blocked_corners(self):
        self.scenario["base_ros_m_s"] = [[0] * 5 for _ in range(5)]
        self.scenario["base_ros_m_s"][2][2] = 0.5
        self.scenario["base_ros_m_s"][3][3] = 0.5
        result = run_fire(self.scenario)["arrival_time_s"]
        self.assertIsNone(result[3][3])

    def test_area_and_resolution_limit(self):
        self.scenario["horizon_s"] = 1
        result = run_fire(self.scenario)
        self.assertEqual(result["affected_area_ha"], 1)
        self.scenario["cell_size_m"] = 50
        result = run_fire(self.scenario)
        self.assertEqual(result["affected_area_ha"], 0.25)

    def test_invalid_input_rejected(self):
        self.scenario["base_ros_m_s"][0][0] = -1
        with self.assertRaises(ValueError):
            run_fire(self.scenario)
        self.scenario["base_ros_m_s"][0][0] = 0.5
        self.scenario["ignitions"] = [[9, 9]]
        with self.assertRaises(ValueError):
            run_fire(self.scenario)


if __name__ == "__main__":
    unittest.main()
