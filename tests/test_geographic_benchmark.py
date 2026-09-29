"""Unit tests for the geographic benchmark comparison tool."""

import unittest

from nightjar.geographic_benchmark import (
    BENCHMARK_SCHEMA_VERSION,
    run_geographic_benchmark,
)
from nightjar.geographic_family import (
    build_synthetic_cems_fixture,
    generate_geographic_scenario,
)


class TestGeographicBenchmark(unittest.TestCase):
    def setUp(self):
        fixture = build_synthetic_cems_fixture("EMSR578", utm_zone=31, rows=35, cols=35)
        self.scenario = generate_geographic_scenario(
            fixture,
            index=0,
            split="development",
            window_rows=25,
            window_cols=25,
            horizon_s=1200.0,
        )

    def test_benchmark_executes_all_policies(self):
        res = run_geographic_benchmark(self.scenario, random_seeds=(1, 2))
        self.assertEqual(res["schema_version"], BENCHMARK_SCHEMA_VERSION)
        self.assertIn("no_intervention", res["policies"])
        self.assertIn("random_valid_mean", res["policies"])
        self.assertIn("heuristic", res["policies"])
        self.assertIn("containment_heuristic", res["policies"])

        no_int = res["policies"]["no_intervention"]
        heur = res["policies"]["heuristic"]
        self.assertEqual(no_int["area_saved_ha"], 0.0)
        self.assertGreaterEqual(heur["area_saved_ha"], 0.0)
        self.assertLessEqual(heur["final_burned_ha"], no_int["final_burned_ha"])


if __name__ == "__main__":
    unittest.main()
