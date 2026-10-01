"""Unit and integration tests for the geographic integrated decision environment."""

import unittest

from nightjar.geographic_family import (
    build_synthetic_cems_fixture,
    generate_geographic_scenario,
)
from nightjar.geographic_integrated import (
    REPLAY_SCHEMA_VERSION,
    run_geographic_integrated,
)


class TestGeographicIntegrated(unittest.TestCase):
    def setUp(self):
        fixture = build_synthetic_cems_fixture("EMSR578", utm_zone=31, rows=40, cols=40)
        self.scenario = generate_geographic_scenario(
            fixture,
            index=0,
            split="development",
            window_rows=30,
            window_cols=30,
            horizon_s=1800.0,
        )

    def test_no_intervention_spreads_without_aircraft_activity(self):
        res = run_geographic_integrated(self.scenario, policy="no_intervention")
        self.assertEqual(res["schema_version"], REPLAY_SCHEMA_VERSION)
        self.assertEqual(res["policy"], "no_intervention")
        self.assertEqual(res["aircraft_metrics"]["drops_count"], 0)
        self.assertEqual(res["aircraft_metrics"]["total_water_dropped_l"], 0)
        self.assertEqual(res["aircraft_metrics"]["total_flight_distance_km"], 0.0)
        self.assertGreater(res["outcomes"]["final_burned_ha"], res["outcomes"]["initial_burned_ha"])

    def test_heuristic_suppression_reduces_burned_area(self):
        no_int = run_geographic_integrated(self.scenario, policy="no_intervention")
        heur = run_geographic_integrated(self.scenario, policy="heuristic")

        self.assertEqual(heur["schema_version"], REPLAY_SCHEMA_VERSION)
        self.assertGreater(heur["aircraft_metrics"]["drops_count"], 0)
        self.assertGreater(heur["aircraft_metrics"]["total_water_dropped_l"], 0)
        self.assertEqual(heur["aircraft_metrics"]["rejected_orders_count"], 0)
        # Water conservation check
        self.assertEqual(
            heur["aircraft_metrics"]["total_water_dropped_l"],
            heur["aircraft_metrics"]["drops_count"] * 6137,
        )
        # Suppression should not burn more than unmitigated fire
        self.assertLessEqual(heur["outcomes"]["final_burned_ha"], no_int["outcomes"]["final_burned_ha"])

    def test_containment_heuristic_runs_without_rejected_orders(self):
        cont = run_geographic_integrated(self.scenario, policy="containment_heuristic")
        self.assertEqual(cont["schema_version"], REPLAY_SCHEMA_VERSION)
        self.assertEqual(cont["aircraft_metrics"]["rejected_orders_count"], 0)

    def test_deterministic_seed_replay(self):
        run1 = run_geographic_integrated(self.scenario, policy="random_valid", policy_seed=123)
        run2 = run_geographic_integrated(self.scenario, policy="random_valid", policy_seed=123)
        self.assertEqual(run1["outcomes"]["final_burned_ha"], run2["outcomes"]["final_burned_ha"])
        self.assertEqual(run1["aircraft_metrics"]["drops_count"], run2["aircraft_metrics"]["drops_count"])
        self.assertEqual(run1["aircraft_metrics"]["total_water_dropped_l"], run2["aircraft_metrics"]["total_water_dropped_l"])

    def test_drop_replay_utm_uses_selected_local_cell_anchor(self):
        from nightjar.geodesy import utm_to_wgs84, wgs84_to_utm

        row, col = 4, 5
        size = self.scenario["grid"]["cell_size_m"]
        east0 = self.scenario["grid"]["origin_utm_easting_m"]
        north0 = self.scenario["grid"]["origin_utm_northing_m"]
        target = [col * size, row * size]
        self.scenario["aircraft"]["initial_position_utm"] = [east0 + target[0], north0 + target[1]]
        result = run_geographic_integrated(self.scenario, policy="no_intervention", _actions=[{"type": "DROP", "target_m": target}], _stop_after_steps=1)
        drop = result["drops"][0]
        self.assertEqual(drop["target_local_m"], target)
        self.assertEqual(drop["target_utm_m"], [east0 + target[0], north0 + target[1]])
        lon, lat = utm_to_wgs84(*drop["target_utm_m"], zone=self.scenario["crs"]["utm_zone"])
        self.assertAlmostEqual(wgs84_to_utm(lon, lat, zone=self.scenario["crs"]["utm_zone"])[0], east0 + col * size, places=3)
        self.assertAlmostEqual(wgs84_to_utm(lon, lat, zone=self.scenario["crs"]["utm_zone"])[1], north0 + row * size, places=3)

    def test_manual_action_scripting(self):
        actions = [{"type": "HOLD"}, {"type": "HOLD"}]
        res = run_geographic_integrated(self.scenario, policy="heuristic", _actions=actions, _stop_after_steps=2)
        self.assertEqual(len(res["decisions"]), 2)
        self.assertEqual(res["decisions"][0]["order"]["type"], "HOLD")
        self.assertEqual(res["decisions"][1]["order"]["type"], "HOLD")

    def test_replay_records_true_arrivals_and_delivered_observations_separately(self):
        result = run_geographic_integrated(self.scenario, policy="heuristic")
        self.assertEqual(result["grid"]["cell_size_m"], self.scenario["grid"]["cell_size_m"])
        self.assertEqual(len(result["arrival_time_s"]), 30)
        burned = sum(t is not None and t <= result["horizon_s"] for row in result["arrival_time_s"] for t in row)
        self.assertEqual(burned, result["outcomes"]["burned_cells_count"])
        self.assertEqual(result["seed"], self.scenario["seed"])
        self.assertEqual(result["crs"], self.scenario["crs"])
        self.assertEqual(result["water_mask"], self.scenario["fire"]["water_mask"])
        self.assertEqual(result["units"]["elevation"], "m")
        for decision in result["decisions"]:
            obs = decision["observation"]
            self.assertEqual(obs["time_s"], decision["decision_time_s"])
            self.assertLessEqual(decision["decision_time_s"], decision["time_s"])
            self.assertNotIn("arrival_time_s", obs)
            for row in obs["fire"]["cells"]:
                for cell in row:
                    if cell is not None:
                        self.assertLessEqual(cell["observed_at_s"], obs["time_s"])


if __name__ == "__main__":
    unittest.main()
