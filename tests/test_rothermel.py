import copy
import json
import math
import unittest
from pathlib import Path

from nightjar.fire_rothermel import run_fire_rothermel
from nightjar.rothermel import (
    FM1_DEPTH_FT,
    FM1_LOAD_LB_FT2,
    FM1_MOISTURE_EXTINCTION,
    FM1_SAV_FT_INV,
    FT_PER_M,
    HEAT_BTU_LB,
    dead_fuel_components,
    fm1_components,
    fm1_directional_rate,
)
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

    def test_generic_si_fm1_matches_legacy_imperial_components(self):
        legacy = fm1_components(0.06, 2.0, 0.1)
        generic = dead_fuel_components(
            0.06,
            2.0,
            0.1,
            load_kg_m2=FM1_LOAD_LB_FT2 * 0.45359237 / 0.3048**2,
            depth_m=FM1_DEPTH_FT * 0.3048,
            sav_m_inv=FM1_SAV_FT_INV / 0.3048,
            moisture_extinction_fraction=FM1_MOISTURE_EXTINCTION,
            heat_kj_kg=HEAT_BTU_LB * 1055.05585262 / 0.45359237 / 1000,
        )
        self.assertEqual(generic.keys(), legacy.keys())
        for field in legacy:
            self.assertAlmostEqual(generic[field], legacy[field], delta=max(1e-12, abs(legacy[field]) * 1e-6))

    def test_generic_fuel_parameter_validation(self):
        baseline = {
            "load_kg_m2": 0.3,
            "depth_m": 0.2,
            "sav_m_inv": 10000.0,
            "moisture_extinction_fraction": 0.2,
            "heat_kj_kg": 18608.0,
        }
        for parameter in ("load_kg_m2", "depth_m", "sav_m_inv", "heat_kj_kg"):
            for value in (0.0, -1.0, float("nan"), float("inf")):
                with self.subTest(parameter=parameter, value=value):
                    kwargs = dict(baseline)
                    kwargs[parameter] = value
                    with self.assertRaises(ValueError):
                        dead_fuel_components(0.06, **kwargs)
        for value in (0.0, -0.1, 1.01, float("nan"), float("inf")):
            with self.subTest(moisture_extinction_fraction=value):
                kwargs = dict(baseline, moisture_extinction_fraction=value)
                with self.assertRaises(ValueError):
                    dead_fuel_components(0.06, **kwargs)
        for moisture, wind, slope in ((float("nan"), 0, 0), (0.1, float("nan"), 0), (0.1, 0, float("nan")), (0.1, -1, 0), (0.1, 0, -1)):
            with self.subTest(moisture=moisture, wind=wind, slope=slope):
                with self.assertRaises(ValueError):
                    dead_fuel_components(moisture, wind, slope, **baseline)

    def test_generic_extinction_and_wind_response(self):
        fuel = {
            "load_kg_m2": 0.3,
            "depth_m": 0.2,
            "sav_m_inv": 10000.0,
            "moisture_extinction_fraction": 0.2,
        }
        self.assertAlmostEqual(dead_fuel_components(0.2, **fuel)["r0_m_s"], 0.0, places=15)
        self.assertGreater(dead_fuel_components(0.1, **fuel)["r0_m_s"], 0.0)
        calm = dead_fuel_components(0.06, **fuel)
        wind = dead_fuel_components(0.06, 12.0, **fuel)
        self.assertEqual(calm["phi_w"], 0.0)
        self.assertGreater(wind["phi_w"], 0.0)
        self.assertEqual(calm["r0_m_s"], wind["r0_m_s"])

    def test_measured_grass_c064_and_f19_inputs_produce_finite_components(self):
        cases = {
            "C064": {"load_kg_m2": 0.283, "depth_m": 0.21, "sav_m_inv": 9770.0, "moisture": 0.063, "wind_m_s": 2.8},
            "F19": {"load_kg_m2": 0.313, "depth_m": 0.51, "sav_m_inv": 12240.0, "moisture": 0.058, "wind_m_s": 4.4},
        }
        for burn, case in cases.items():
            with self.subTest(burn=burn):
                components = dead_fuel_components(
                    case["moisture"],
                    case["wind_m_s"],
                    load_kg_m2=case["load_kg_m2"],
                    depth_m=case["depth_m"],
                    sav_m_inv=case["sav_m_inv"],
                    moisture_extinction_fraction=0.22,
                    heat_kj_kg=18608.0,
                )
                self.assertTrue(all(math.isfinite(value) for value in components.values()))

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
