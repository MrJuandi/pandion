"""Invariant and classification tests for the isolated numerical benchmark."""

from math import hypot, sqrt
from importlib.util import find_spec
import unittest

from nightjar.propagation_verification import (
    ANALYTICAL_TOLERANCE,
    REFINEMENT_TOLERANCE,
    SCHEMA_VERSION,
    TARGETS_M,
    _passes,
    run_verification,
)


class PropagationVerificationTests(unittest.TestCase):
    def test_strict_acceptance_boundaries_are_classified_without_rounding(self):
        self.assertTrue(_passes(ANALYTICAL_TOLERANCE - 1e-9, ANALYTICAL_TOLERANCE))
        self.assertFalse(_passes(ANALYTICAL_TOLERANCE, ANALYTICAL_TOLERANCE))
        self.assertFalse(_passes(REFINEMENT_TOLERANCE + 1e-9, REFINEMENT_TOLERANCE))
        self.assertFalse(_passes(float("nan"), REFINEMENT_TOLERANCE))

    @unittest.skipIf(find_spec("pyretechnics") is None, "install requirements-standard.txt")
    def test_report_metadata_coordinates_and_classifications_are_consistent(self):
        report = run_verification(include_levelset=True)
        self.assertEqual(report["schema_version"], SCHEMA_VERSION)
        self.assertEqual(report["crs"], "LOCAL_METERS")
        self.assertEqual(report["units"]["arrival"], "s")
        self.assertFalse(report["checks"]["dijkstra_analytical_arrival_lt_5pct"])
        self.assertEqual({tuple(case["target_xy_m"]) for case in report["cases"]}, set(TARGETS_M))
        self.assertEqual(len(report["cases"]), len(report["configuration"]["cell_sizes_m"]) * len(TARGETS_M))
        for case in report["cases"]:
            # Independent eight-neighbor distance formula catches accidental
            # changes to center distances, diagonals, units or target indices.
            x, y = case["target_xy_m"]
            octile = max(x, y) + (sqrt(2) - 1) * min(x, y)
            self.assertAlmostEqual(case["dijkstra_arrival_s"] * case["still_air_rate_m_s"], octile, delta=1e-6)
            self.assertAlmostEqual(case["analytic_point_arrival_s"] * case["still_air_rate_m_s"], hypot(x, y), places=8)
            expected = _passes(case["dijkstra_relative_error"], ANALYTICAL_TOLERANCE)
            self.assertEqual(case["dijkstra_analytical_pass"], expected)
            self.assertGreater(case["target_distance_over_cell_size"], 0)
        for comparison in report["grid_refinement"]:
            self.assertEqual(comparison["pass"], _passes(comparison["arrival_metric_change_fraction"], REFINEMENT_TOLERANCE))
        for pair in report["resolution_pair_acceptance"]:
            self.assertEqual(pair["pair_pass"], pair["analytical_pass_at_both_resolutions"] and pair["refinement_pass"])
            if pair["engine"] == "dijkstra":
                self.assertFalse(pair["pair_pass"])
            elif pair["coarse_cell_size_m"] == 2:
                self.assertTrue(pair["pair_pass"])
            else:
                self.assertFalse(pair["pair_pass"])
        self.assertEqual(set(report["code_sha256"]), {"nightjar/fire_standard.py", "nightjar/propagation_verification.py"})
        self.assertTrue(all(len(digest) == 64 for digest in report["code_sha256"].values()))
        self.assertFalse(report["seed_sensitivity"]["seed_sensitive"])
        self.assertEqual(report["seed_sensitivity"]["arrival_absolute_change_s"], 0.0)
        self.assertEqual(report["overall_pass"], all(value is not False for value in report["checks"].values()))


if __name__ == "__main__":
    unittest.main()
