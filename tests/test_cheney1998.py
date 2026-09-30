"""Checks for the isolated Cheney (1998) equation reconstruction."""

from __future__ import annotations

import math
import unittest

from nightjar.cheney1998 import (
    MODULE_VERSION,
    GrassInputs,
    estimate_dead_moisture_percent,
    grass_rate,
)


class GrassInputsTests(unittest.TestCase):
    def test_accepts_domain_endpoints_and_is_frozen(self) -> None:
        inputs = GrassInputs(0, 2, 50, "undisturbed")
        self.assertEqual(inputs.open_wind_10m_km_h, 0)
        with self.assertRaises((AttributeError, TypeError)):
            inputs.open_wind_10m_km_h = 2  # type: ignore[misc]
        self.assertEqual(GrassInputs(80, 24, 100, "cut_grazed").curing_percent, 100)
        self.assertEqual(MODULE_VERSION, "cheney1998-reconstruction-0.1")

    def test_rejects_nonfinite_boolean_and_out_of_domain_numeric_inputs(self) -> None:
        for field_values in (
            (math.nan, 6, 95, "undisturbed"),
            (math.inf, 6, 95, "undisturbed"),
            (5, -math.inf, 95, "undisturbed"),
            (True, 6, 95, "undisturbed"),
            (5, False, 95, "undisturbed"),
            (5, 6, True, "undisturbed"),
            (-0.01, 6, 95, "undisturbed"),
            (80.01, 6, 95, "undisturbed"),
            (5, 1.99, 95, "undisturbed"),
            (5, 24.01, 95, "undisturbed"),
            (5, 6, 49.99, "undisturbed"),
            (5, 6, 100.01, "undisturbed"),
        ):
            with self.subTest(values=field_values), self.assertRaises(ValueError):
                GrassInputs(*field_values)

    def test_rejects_unknown_pasture_and_eaten_out_low_wind(self) -> None:
        with self.assertRaises(ValueError):
            GrassInputs(6, 6, 95, "natural")
        with self.assertRaises(ValueError):
            GrassInputs(5, 6, 95, "eaten_out")
        with self.assertRaises(ValueError):
            GrassInputs(0, 6, 95, "eaten_out")


class GrassRateTests(unittest.TestCase):
    def assertClose(self, actual: float, expected: float, places: int = 12) -> None:
        self.assertAlmostEqual(actual, expected, places=places)

    def test_independent_high_precision_golden_calculations(self) -> None:
        # Expected values were computed independently at 50-digit precision
        # from the published factors and equations (mpmath 1.3.0), not by
        # calling this implementation. Inputs and outputs below retain the
        # equation's stated units. The final case verifies the moisture floor.
        fixtures = (
            # U, M, C, pasture, base km/h, raw phiM, phiC, rate m/s
            (5, 6, 95, "undisturbed", 1.399, 0.5230909131025008075,
             0.9155294302156374726, 0.1861078529854362299),
            (5, 12, 50, "cut_grazed", 1.099, 0.2736,
             0.01860465116279069767, 0.001553934883720930233),
            (8, 8, 80, "undisturbed", 3.518040975132462664,
             0.4214728147759176057, 0.4600248855498040596,
             0.1894738528098316475),
            (12, 15, 100, "cut_grazed", 4.794611583964124188, 0.205,
             0.999872968220634334, 0.2729918099345181745),
            (20, 24, 60, "eaten_out", 4.059855513999374902, -0.0002,
             0.06177079540863220861, 0.0),
        )
        for wind, moisture, curing, pasture, base, raw_phi_m, phi_c, rate in fixtures:
            with self.subTest(wind=wind, moisture=moisture, pasture=pasture):
                result = grass_rate(GrassInputs(wind, moisture, curing, pasture))
                self.assertClose(result["base_rate_km_h"], base)  # type: ignore[arg-type]
                self.assertClose(result["raw_moisture_factor"], raw_phi_m)  # type: ignore[arg-type]
                self.assertClose(result["curing_factor"], phi_c)  # type: ignore[arg-type]
                self.assertClose(result["rate_m_s"], rate)  # type: ignore[arg-type]

    def test_wind_boundary_uses_low_branch_at_five(self) -> None:
        undisturbed_at_five = grass_rate(GrassInputs(5, 6, 95, "undisturbed"))
        cut_at_five = grass_rate(GrassInputs(5, 6, 95, "cut_grazed"))
        undisturbed_above_five = grass_rate(GrassInputs(5.001, 6, 95, "undisturbed"))
        self.assertClose(undisturbed_at_five["base_rate_km_h"], 1.399)  # type: ignore[arg-type]
        self.assertClose(cut_at_five["base_rate_km_h"], 1.099)  # type: ignore[arg-type]
        self.assertGreater(undisturbed_above_five["base_rate_km_h"], 1.4)

    def test_moisture_boundary_uses_exponential_below_twelve_and_linear_at_twelve(self) -> None:
        below = grass_rate(GrassInputs(8, 11.999, 90, "undisturbed"))
        at_boundary = grass_rate(GrassInputs(8, 12, 90, "undisturbed"))
        self.assertClose(below["raw_moisture_factor"], math.exp(-0.108 * 11.999))  # type: ignore[arg-type]
        self.assertClose(at_boundary["raw_moisture_factor"], 0.684 - 0.0342 * 12)  # type: ignore[arg-type]

    def test_wind_ten_boundary_selects_high_wind_linear_moisture_factor(self) -> None:
        just_below = grass_rate(GrassInputs(9.999, 15, 90, "undisturbed"))
        at_ten = grass_rate(GrassInputs(10, 15, 90, "undisturbed"))
        self.assertClose(just_below["raw_moisture_factor"], 0.684 - 0.0342 * 15)  # type: ignore[arg-type]
        self.assertClose(at_ten["raw_moisture_factor"], 0.547 - 0.0228 * 15)  # type: ignore[arg-type]

    def test_negative_rounded_moisture_factor_is_exposed_and_floored(self) -> None:
        result = grass_rate(GrassInputs(10, 24, 60, "undisturbed"))
        self.assertClose(result["raw_moisture_factor"], -0.0002)  # type: ignore[arg-type]
        self.assertEqual(result["moisture_factor"], 0.0)
        self.assertTrue(result["moisture_floor_applied"])
        self.assertEqual(result["rate_m_s"], 0.0)

    def test_curing_endpoints_are_used_without_normalization(self) -> None:
        low = grass_rate(GrassInputs(12, 8, 50, "undisturbed"))
        high = grass_rate(GrassInputs(12, 8, 100, "undisturbed"))
        self.assertClose(low["curing_factor"], 1.12 / 60.2)  # type: ignore[arg-type]
        self.assertLess(high["curing_factor"], 1.0)

    def test_output_rate_converts_km_per_hour_to_metres_per_second(self) -> None:
        result = grass_rate(GrassInputs(5, 12, 50, "cut_grazed"))
        self.assertClose(
            result["rate_m_s"],
            result["base_rate_km_h"] * result["moisture_factor"]
            * result["curing_factor"] / 3.6,  # type: ignore[operator]
        )

    def test_estimated_moisture_is_separate_and_not_clamped(self) -> None:
        self.assertAlmostEqual(estimate_dead_moisture_percent(20, 50), 12.38)
        self.assertLess(estimate_dead_moisture_percent(60, 0), 0.0)
        for temp, rh in ((math.nan, 50), (20, math.inf), (20, -0.1), (20, 100.1), (True, 50)):
            with self.subTest(temp=temp, rh=rh), self.assertRaises(ValueError):
                estimate_dead_moisture_percent(temp, rh)


if __name__ == "__main__":
    unittest.main()
