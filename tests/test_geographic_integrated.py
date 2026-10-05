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

    def test_realtime_observation_is_full_current_state(self):
        result = run_geographic_integrated(self.scenario, policy="heuristic")
        self.assertEqual(result["observation_mode"], "realtime")
        for decision in result["decisions"]:
            obs = decision["observation"]
            t = obs["time_s"]
            self.assertEqual(obs["observation_mode"], "realtime")
            self.assertIsNotNone(obs["fuel_moisture"])
            for r, row in enumerate(obs["fire"]["cells"]):
                for c, cell in enumerate(row):
                    self.assertIsNotNone(cell)
                    self.assertEqual(cell["age_s"], 0.0)
                    arrival = result["arrival_time_s"][r][c]
                    reached = arrival is not None and arrival <= t + 1e-6
                    self.assertEqual(cell["fire_detected"], reached)
                    if not reached:
                        self.assertEqual(cell["fire_state"], "unburned")
                        self.assertIsNone(cell["arrival_time_s"])
                    else:
                        burnout = result["burnout_time_s"][r][c]
                        expected = "burning" if burnout > t + 1e-6 else "burned"
                        self.assertEqual(cell["fire_state"], expected)

    def test_burning_cells_outlast_front_arrival_and_later_burn_out(self):
        result = run_geographic_integrated(self.scenario, policy="no_intervention")
        states = {cell["fire_state"]
                  for decision in result["decisions"]
                  for row in decision["observation"]["fire"]["cells"]
                  for cell in row}
        self.assertTrue({"unburned", "burning", "burned"} <= states)
        for arrival_row, burnout_row in zip(result["arrival_time_s"], result["burnout_time_s"]):
            for arrival, burnout in zip(arrival_row, burnout_row):
                self.assertEqual(arrival is None, burnout is None)
                if arrival is not None and arrival > 0:
                    # Crossing time alone exceeds Anderson residence (seconds).
                    self.assertGreater(burnout - arrival, 1.0)

    def test_initial_perimeter_interior_is_burned_and_edge_is_burning(self):
        result = run_geographic_integrated(self.scenario, policy="no_intervention", _stop_after_steps=1)
        cells = result["pending_observation"]["fire"]["cells"]
        mask = self.scenario["fire"]["burn_mask"]
        fuel = self.scenario["fire"]["fuel_grid"]
        rows, cols = len(mask), len(mask[0])

        def burnable(code):
            return int(code) > 0 and not 91 <= int(code) <= 99

        interior = edge = 0
        for r in range(rows):
            for c in range(cols):
                if not mask[r][c] or not burnable(fuel[r][c]):
                    continue
                on_front = any(0 <= r + dr < rows and 0 <= c + dc < cols and burnable(fuel[r + dr][c + dc])
                               and not mask[r + dr][c + dc]
                               for dr in (-1, 0, 1) for dc in (-1, 0, 1) if dr or dc)
                expected = "burning" if on_front else "burned"
                self.assertEqual(cells[r][c]["fire_state"], expected, (r, c))
                edge += on_front
                interior += not on_front
        self.assertGreater(edge, 0)

    def test_future_fire_cannot_change_realtime_observation(self):
        first = run_geographic_integrated(self.scenario, policy="no_intervention", _stop_after_steps=5)
        later = run_geographic_integrated(self.scenario, policy="no_intervention", _stop_after_steps=20)
        self.assertEqual(first["pending_observation"], later["decisions"][5]["observation"])

    def test_scheduled_mode_keeps_delayed_reports(self):
        result = run_geographic_integrated(self.scenario, policy="no_intervention", observation_mode="scheduled")
        self.assertEqual(result["observation_mode"], "scheduled")
        ages = [cell["age_s"]
                for decision in result["decisions"]
                for row in decision["observation"]["fire"]["cells"]
                for cell in row if cell is not None]
        self.assertTrue(any(age > 0 for age in ages))

    def test_training_prefix_can_skip_audit_observations(self):
        full = run_geographic_integrated(self.scenario, policy="no_intervention", _stop_after_steps=4)
        lean = run_geographic_integrated(self.scenario, policy="no_intervention", _stop_after_steps=4,
                                         _record_observations=False)
        self.assertTrue(all("observation" not in d for d in lean["decisions"]))
        self.assertEqual(full["pending_observation"], lean["pending_observation"])
        self.assertEqual(full["outcomes"], lean["outcomes"])

    def test_unknown_observation_mode_rejected(self):
        with self.assertRaises(ValueError):
            run_geographic_integrated(self.scenario, policy="no_intervention", observation_mode="delayed")


if __name__ == "__main__":
    unittest.main()
