import copy
import json
import unittest
from pathlib import Path

from nightjar.fire_rothermel import run_fire_rothermel
from nightjar.rothermel import FM1_MOISTURE_EXTINCTION, fm1_components, fm1_directional_rate
from nightjar.sensitivity import compare_resolution


EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "fm1_fire.json"


class RothermelFM1Tests(unittest.TestCase):
    def setUp(self):
        self.scenario = json.loads(EXAMPLE.read_text())

    def test_source_figure_16_four_mph_reference(self):
        # Andrews (2018), figure 16: FM1, 6% dead moisture, 4 mi/h
        # midflame wind, no slope is about 66 ft/min at SAV 3500 ft^-1.
        wind_m_s = 4 * 1609.344 / 3600
        rate_ft_min = fm1_directional_rate(0.06, (wind_m_s, 0), (0, 0), (1, 0)) * 196.8503937007874
        self.assertGreater(rate_ft_min, 60)
        self.assertLess(rate_ft_min, 75)

    def test_zero_wind_and_moisture_extinction(self):
        r0 = fm1_components(0.06)["r0_m_s"]
        east = fm1_directional_rate(0.06, (0, 0), (0, 0), (1, 0))
        west = fm1_directional_rate(0.06, (0, 0), (0, 0), (-1, 0))
        self.assertAlmostEqual(r0, east)
        self.assertAlmostEqual(east, west)
        self.assertAlmostEqual(fm1_directional_rate(FM1_MOISTURE_EXTINCTION, (0, 0), (0, 0), (1, 0)), 0)

    def test_directional_wind_slope_and_moisture(self):
        east = fm1_directional_rate(0.06, (2, 0), (0, 0), (1, 0))
        west = fm1_directional_rate(0.06, (2, 0), (0, 0), (-1, 0))
        self.assertGreater(east, west)
        self.assertGreater(fm1_directional_rate(0.06, (0, 0), (0.1, 0), (1, 0)), fm1_directional_rate(0.06, (0, 0), (0.1, 0), (-1, 0)))
        self.assertGreater(east, fm1_directional_rate(0.10, (2, 0), (0, 0), (1, 0)))

    def test_grid_repeatable_barrier_and_no_corner_cut(self):
        replay = run_fire_rothermel(self.scenario)
        self.assertEqual(replay, run_fire_rothermel(copy.deepcopy(self.scenario)))
        self.assertLess(replay["arrival_time_s"][2][3], replay["arrival_time_s"][2][1])
        self.scenario["fuel_model"][2][3] = "NONE"
        self.assertIsNone(run_fire_rothermel(self.scenario)["arrival_time_s"][2][3])
        self.scenario["fuel_model"] = [["NONE"] * 5 for _ in range(5)]
        self.scenario["fuel_model"][2][2] = "FM1"
        self.scenario["fuel_model"][3][3] = "FM1"
        self.assertIsNone(run_fire_rothermel(self.scenario)["arrival_time_s"][3][3])

    def test_invalid_ignition_and_moisture(self):
        self.scenario["dead_fuel_moisture_fraction"][2][2] = 0.12
        with self.assertRaises(ValueError):
            run_fire_rothermel(self.scenario)
        self.scenario["dead_fuel_moisture_fraction"][2][2] = 0.06
        self.scenario["dead_fuel_moisture_fraction"][0][0] = -0.1
        with self.assertRaises(ValueError):
            run_fire_rothermel(self.scenario)

    def test_resolution_comparison_uses_fixed_extent(self):
        result = compare_resolution((100, 50, 25))
        self.assertEqual([r["grid_cells_per_side"] for r in result["results"]], [11, 21, 41])
        self.assertEqual(result["fixed_center_extent_m"], 1000)
        self.assertNotEqual(result["results"][0]["affected_area_ha"], result["results"][-1]["affected_area_ha"])

    def test_resolution_profile_wind_mirror_and_slope_change(self):
        east = compare_resolution((25, 10), wind_to_m_s=(2, 0))
        west = compare_resolution((25, 10), wind_to_m_s=(-2, 0))
        slope = compare_resolution((25, 10), wind_to_m_s=(2, 0), elevation_gradient=(0.1, 0))
        self.assertEqual([r["affected_area_ha"] for r in east["results"]], [r["affected_area_ha"] for r in west["results"]])
        self.assertNotEqual(east["results"][-1]["affected_area_ha"], slope["results"][-1]["affected_area_ha"])


if __name__ == "__main__":
    unittest.main()
