"""Tests for the water-parameter sensitivity tool."""

import unittest

from nightjar import geographic_integrated as gi
from nightjar import water_sensitivity as ws


class WaterSensitivityTests(unittest.TestCase):
    def tearDown(self):
        ws.apply_variant(ws.BASE_LIMITS, 1.0)

    def test_variants_change_hold_limits_and_delay(self):
        ws.apply_variant(((1, 500.0), (3, 1600.0)), 0.0)
        self.assertEqual(gi.drops_needed(400.0), 1)
        self.assertEqual(gi.drops_needed(900.0), 3)
        self.assertEqual(gi.drops_needed(2000.0), float("inf"))
        self.assertEqual(gi.WATER_HEATING_KJ_KG + gi.WATER_EVAPORATION_KJ_KG, 0.0)
        ws.apply_variant(ws.BASE_LIMITS, 1.0)
        self.assertEqual(gi.drops_needed(900.0), 1)
        self.assertAlmostEqual(gi.WATER_HEATING_KJ_KG + gi.WATER_EVAPORATION_KJ_KG, 4.186 * 80.0 + 2257.0)

    def test_baseline_variant_matches_simulator_defaults(self):
        name, limits, scale = ws.VARIANTS[0]
        self.assertEqual(name, "baseline")
        self.assertEqual(limits, gi.DROP_HOLD_LIMITS_KW_M)
        self.assertEqual(scale, 1.0)

    def test_summary_reports_growth_prevented_and_gap_to_containment(self):
        rows = []
        for scenario, none, heuristic, containment in (("a", 10.0, 6.0, 7.0), ("b", 30.0, 24.0, 23.0)):
            rows += [(scenario, "baseline", "no_intervention", none),
                     (scenario, "baseline", "heuristic", heuristic),
                     (scenario, "baseline", "containment_heuristic", containment)]
        (row,) = ws.summarize(rows)
        self.assertEqual(row["no_intervention_ha"], 20.0)
        self.assertEqual(row["heuristic_ha"], 15.0)
        self.assertEqual(row["growth_prevented_by_heuristic_pct"], 25.0)
        self.assertEqual(row["heuristic_minus_containment_ha"], 0.0)


if __name__ == "__main__":
    unittest.main()
