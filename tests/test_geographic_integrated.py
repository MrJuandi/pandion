"""Unit and integration tests for the geographic integrated decision environment."""

from copy import deepcopy
import unittest

from pandion.environment import SCHEMA_VERSION as ENVIRONMENT_SCHEMA_VERSION
from pandion.geographic_family import (
    MOISTURE_ENSEMBLES,
    build_synthetic_cems_fixture,
    generate_geographic_scenario,
)
from pandion.geographic_integrated import (
    REPLAY_SCHEMA_VERSION,
    geographic_episode,
    run_geographic_integrated,
)


def growing_fire_scenario(horizon_s=3600.0):
    """Uniform GR2 under a brisk wind: the front is still advancing at the horizon."""
    fixture = build_synthetic_cems_fixture("EMSR578", utm_zone=31, rows=60, cols=60)
    scenario = generate_geographic_scenario(fixture, index=0, split="development", window_rows=40,
                                            window_cols=40, horizon_s=horizon_s)
    scenario["fire"]["fuel_grid"] = [[102] * 40 for _ in range(40)]
    scenario["fire"]["burn_mask"] = [[abs(r - 20) <= 1 and abs(c - 20) <= 1 for c in range(40)] for r in range(40)]
    return scenario


def with_weather(scenario, moisture, wind_to_m_s):
    """Uniform moisture and wind for the whole episode."""
    out = deepcopy(scenario)
    out["fire"]["moisture"] = dict(zip(("dead_1h", "dead_10h", "dead_100h", "dead_herbaceous", "live_herbaceous",
                                        "live_woody"), moisture))
    out["fire"]["wind_to_m_s"] = list(wind_to_m_s)
    out.pop("environment", None)
    return out


class TestDropEffect(unittest.TestCase):
    """D-74: one treated cell per drop, holding only fronts up to the single-drop intensity limit."""

    def treated_cells_reached(self, replay):
        arrival = replay["arrival_time_s"]
        treated = [cell for drop in replay["drops"] for cell in drop["treated_cells"]]
        return treated, [cell for cell in treated if arrival[cell[0]][cell[1]] is not None]

    def test_each_drop_treats_at_most_the_target_cell(self):
        replay = run_geographic_integrated(growing_fire_scenario(), policy="heuristic")
        self.assertTrue(replay["drops"])
        self.assertTrue(all(drop["treated_cells_count"] <= 1 for drop in replay["drops"]))

    def test_low_intensity_front_is_held_by_a_treated_cell(self):
        # GR2 at 8% dead moisture and 1 m/s midflame wind: head fire about 160 kW/m.
        scenario = with_weather(growing_fire_scenario(), (0.08, 0.09, 0.10, 0.08, 0.6, 0.9), (0.7, 0.7))
        treated, reached = self.treated_cells_reached(run_geographic_integrated(scenario, policy="heuristic"))
        self.assertTrue(treated)
        self.assertEqual(reached, [])

    def test_intense_front_breaches_a_treated_cell(self):
        # GR2 at 3% dead moisture and 4 m/s midflame wind: head fire about 2,500 kW/m.
        scenario = with_weather(growing_fire_scenario(), (0.03, 0.04, 0.05, 0.03, 0.3, 0.6), (2.83, 2.83))
        treated, reached = self.treated_cells_reached(run_geographic_integrated(scenario, policy="heuristic"))
        self.assertTrue(treated)
        self.assertTrue(reached)


class TestStackedDrops(unittest.TestCase):
    """D-76: up to three drops stack on a cell; three hold fronts up to 3,200 kW/m."""

    def stacked(self, wind):
        from collections import Counter
        scenario = with_weather(growing_fire_scenario(), (0.03, 0.04, 0.05, 0.03, 0.3, 0.6), wind)
        replay = run_geographic_integrated(scenario, policy="heuristic")
        arrival = replay["arrival_time_s"]
        counts = Counter(tuple(c) for d in replay["drops"] for c in d["treated_cells"])
        reached = {cell: n for cell, n in counts.items() if arrival[cell[0]][cell[1]] is not None}
        return counts, reached

    def test_three_stacked_drops_hold_a_front_one_drop_cannot(self):
        # GR2, 3 % dead moisture, 4 m/s midflame wind: head fire about 2,500 kW/m.
        counts, reached = self.stacked((2.83, 2.83))
        self.assertLessEqual(max(counts.values()), 3)
        self.assertIn(3, counts.values())
        self.assertNotIn(3, reached.values())
        self.assertIn(1, reached.values())

    def test_a_breaching_front_is_delayed_by_boiling_off_the_water(self):
        # D-78: one drop on the head of a ~4,000 kW/m front delays its arrival there by
        # water mass x (heating + evaporation) / (intensity x cell width).
        from pandion.geographic_integrated import WATER_EVAPORATION_KJ_KG, WATER_HEATING_KJ_KG
        from pandion.geographic_training_env import GeographicTrainingEnv
        from pandion.policies import candidate_cells, choose_order
        scenario = with_weather(growing_fire_scenario(), (0.03, 0.04, 0.05, 0.03, 0.3, 0.6), (4.6, 4.6))
        scenario["aircraft"]["fleet_size"] = 1

        def run(drop):
            env = GeographicTrainingEnv(scenario, drop_bonus_per_cell=0.0, action_mode="spatial")
            obs, done, target = env.reset(), False, None
            while not done:
                if drop and target is None and obs["aircraft"]["phase"] == "idle" and obs["aircraft"]["water_l"] > 0:
                    order = choose_order("heuristic", obs, candidates=candidate_cells(obs))
                    target = (round(order["target_m"][1] / 100), round(order["target_m"][0] / 100))
                else:
                    order = {"type": "HOLD"} if {"type": "HOLD"} in env.action_mask else env.action_mask[0]
                obs, _, done, _ = env.step(order)
            return env.replay, target

        plain, _ = run(False)
        wet, (r, c) = run(True)
        self.assertEqual([d["treated_cells"] for d in wet["drops"]], [[[r, c]]])
        delay = wet["arrival_time_s"][r][c] - plain["arrival_time_s"][r][c]
        energy_kj = 6137 * (WATER_HEATING_KJ_KG + WATER_EVAPORATION_KJ_KG)
        self.assertGreater(delay, 0.0)
        # Implied intensity of the crossing front: the head of this fire, about 4,000 kW/m.
        self.assertAlmostEqual(energy_kj / (delay * 100.0), 4033.0, delta=60.0)

    def test_a_front_above_the_stacked_limit_breaches_three_drops(self):
        # Same fuel at 6.5 m/s: head fire about 4,000 kW/m, beyond what three drops hold.
        counts, reached = self.stacked((4.6, 4.6))
        self.assertIn(3, reached.values())


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
        # Very low moisture (D1L1) so the fire spreads whichever ensemble member was drawn.
        self.scenario["fire"]["moisture"] = dict(MOISTURE_ENSEMBLES[0])

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

    def test_hold_aware_heuristic_runs_without_rejected_orders_and_is_deterministic(self):
        one = run_geographic_integrated(self.scenario, policy="hold_aware_heuristic")
        two = run_geographic_integrated(self.scenario, policy="hold_aware_heuristic")
        self.assertEqual(one["policy"], "hold_aware_heuristic")
        self.assertEqual(one["aircraft_metrics"]["rejected_orders_count"], 0)
        self.assertEqual(one["arrival_time_s"], two["arrival_time_s"])

    def test_deterministic_seed_replay(self):
        run1 = run_geographic_integrated(self.scenario, policy="random_valid", policy_seed=123)
        run2 = run_geographic_integrated(self.scenario, policy="random_valid", policy_seed=123)
        self.assertEqual(run1["outcomes"]["final_burned_ha"], run2["outcomes"]["final_burned_ha"])
        self.assertEqual(run1["aircraft_metrics"]["drops_count"], run2["aircraft_metrics"]["drops_count"])
        self.assertEqual(run1["aircraft_metrics"]["total_water_dropped_l"], run2["aircraft_metrics"]["total_water_dropped_l"])

    def test_drop_replay_utm_uses_selected_local_cell_anchor(self):
        from pandion.geodesy import utm_to_wgs84, wgs84_to_utm

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

    def test_refill_beyond_horizon_is_not_completed_or_applied(self):
        scenario = deepcopy(self.scenario)
        scenario["fire"]["horizon_s"] = 100.0
        scenario["aircraft"]["initial_position_utm"] = [
            scenario["grid"]["origin_utm_easting_m"] + 6000.0,
            scenario["grid"]["origin_utm_northing_m"],
        ]
        initial = [6000.0, 0.0]
        result = run_geographic_integrated(scenario, policy="no_intervention", _actions=[
            {"type": "DROP", "target_m": initial}, {"type": "RETURN_TO_REFILL"}])
        self.assertEqual(result["decisions"][0]["order"]["type"], "DROP")
        decision = result["decisions"][1]
        self.assertGreater(decision["time_s"], result["horizon_s"])
        self.assertEqual(result["aircraft_metrics"]["refills_count"], 0)
        self.assertEqual(decision["observation"]["aircraft"]["water_l"], 0)
        self.assertGreater(decision["completed_at_s"], result["horizon_s"])
        arrival = decision["refill_transit_s"]
        self.assertLess(arrival, result["horizon_s"])
        arrival_point = next(point for point in result["aircraft_track"] if point["time_s"] == arrival)
        self.assertEqual(arrival_point["position_m"], result["aircraft_track"][-1]["position_m"])
        self.assertEqual(result["aircraft_track"][-1]["time_s"], result["horizon_s"])
        from math import dist
        outbound_m = dist(result["aircraft_track"][0]["position_m"], arrival_point["position_m"])
        self.assertAlmostEqual(result["aircraft_metrics"]["total_flight_distance_km"], outbound_m / 1000, places=3)

    def test_refill_transit_track_ends_at_horizon_without_future_waypoints(self):
        scenario = deepcopy(self.scenario)
        scenario["fire"]["horizon_s"] = 50.0
        start = [12000.0, 0.0]
        scenario["aircraft"]["initial_position_utm"] = [
            scenario["grid"]["origin_utm_easting_m"] + start[0],
            scenario["grid"]["origin_utm_northing_m"] + start[1],
        ]
        result = run_geographic_integrated(
            scenario, policy="no_intervention", _actions=[{"type": "RETURN_TO_REFILL"}])
        self.assertGreater(result["decisions"][0]["refill_transit_s"], result["horizon_s"])
        self.assertTrue(all(point["time_s"] <= result["horizon_s"] for point in result["aircraft_track"]))
        terminal = result["aircraft_track"][-1]
        self.assertEqual(terminal["time_s"], result["horizon_s"])
        self.assertNotEqual(terminal["position_m"], start)

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

    def test_cells_keep_burning_while_the_front_advances_from_them(self):
        """D-67: a 'burned' cell almost never borders a cell the fire reaches later.

        The remaining cases are neighbours reached later from another direction
        (not yet scheduled when the observation was taken); before D-67 the slow
        flanks made this common.
        """
        result = run_geographic_integrated(self.scenario, policy="no_intervention")
        arrival = result["arrival_time_s"]
        rows, cols = len(arrival), len(arrival[0])
        burned = later = 0
        for decision in result["decisions"]:
            t = decision["observation"]["time_s"]
            cells = decision["observation"]["fire"]["cells"]
            for r in range(rows):
                for c in range(cols):
                    if cells[r][c]["fire_state"] != "burned":
                        continue
                    burned += 1
                    later += any(0 <= r + dr < rows and 0 <= c + dc < cols and arrival[r + dr][c + dc] is not None
                                 and arrival[r + dr][c + dc] > t + 1e-6
                                 for dr in (-1, 0, 1) for dc in (-1, 0, 1) if dr or dc)
        self.assertGreater(burned, 0)
        self.assertLess(later / burned, 0.01)

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

    def _fleet(self, size):
        import copy
        scenario = copy.deepcopy(self.scenario)
        scenario["aircraft"]["fleet_size"] = size
        return scenario

    def test_two_aircraft_both_act_and_observe_the_fleet(self):
        result = run_geographic_integrated(self._fleet(2), policy="heuristic")
        egos = {d["aircraft_index"] for d in result["decisions"]}
        self.assertEqual(egos, {0, 1})
        drops = [d for d in result["decisions"] if d["order"]["type"] == "DROP"]
        self.assertEqual({d["aircraft_index"] for d in drops}, {0, 1})
        for decision in result["decisions"]:
            obs = decision["observation"]
            self.assertEqual(obs["aircraft_index"], decision["aircraft_index"])
            ego = obs["fleet"][obs["aircraft_index"]]
            self.assertEqual(ego["phase"], "idle")
            self.assertEqual(ego["position_m"], obs["aircraft"]["position_m"])
            self.assertEqual(len(obs["fleet"]), 2)
        times = [d["decision_time_s"] for d in result["decisions"]]
        self.assertEqual(times, sorted(times))
        self.assertEqual({e["aircraft_index"] for e in result["aircraft_track"]}, {0, 1})

    def test_other_aircraft_position_is_interpolated_during_transit(self):
        result = run_geographic_integrated(self._fleet(2), policy="heuristic")
        for decision in result["decisions"]:
            for other in decision["observation"]["fleet"]:
                if other["phase"] == "transit_to_drop" and other["busy_remaining_s"] > 1.0:
                    self.assertNotEqual(other["position_m"], [round(v, 2) for v in other["target_m"]])
                    return
        self.fail("no aircraft observed in transit")

    def test_drop_effects_respect_time_order_and_never_precede_decisions(self):
        result = run_geographic_integrated(self._fleet(2), policy="heuristic")
        drop_times = [d["time_s"] for d in result["drops"]]
        self.assertEqual(drop_times, sorted(drop_times))
        for decision in result["decisions"]:
            if decision["order"]["type"] == "DROP":
                self.assertGreaterEqual(decision["time_s"], decision["decision_time_s"])

    def test_more_aircraft_do_not_change_unmitigated_fire(self):
        one = run_geographic_integrated(self.scenario, policy="no_intervention")
        two = run_geographic_integrated(self._fleet(2), policy="no_intervention")
        self.assertEqual(one["arrival_time_s"], two["arrival_time_s"])

    def test_audit_observation_is_what_the_controller_saw(self):
        result = run_geographic_integrated(self.scenario, policy="heuristic")
        seen = []
        for decision in result["decisions"]:
            self.assertEqual(decision["observation"]["known_treated_cells"], seen)
            seen = seen + [cell for drop in result["drops"] if drop["time_s"] == decision["time_s"]
                           and decision["order"]["type"] == "DROP" for cell in drop["treated_cells"]]

    def test_no_policy_drops_overlap_another_pending_drop(self):
        import itertools
        footprint = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))
        for policy in ("heuristic", "containment_heuristic", "random_valid"):
            result = run_geographic_integrated(self._fleet(2), policy=policy, policy_seed=3)
            drops = [d for d in result["decisions"] if d["order"]["type"] == "DROP"]
            self.assertTrue({d["aircraft_index"] for d in drops} == {0, 1}, policy)
            size = self.scenario["grid"]["cell_size_m"]
            spans = []
            for d in drops:
                r = round(d["order"]["target_m"][1] / size)
                c = round(d["order"]["target_m"][0] / size)
                cells = {(r + dr, c + dc) for dr, dc in footprint}
                spans.append((d["decision_time_s"], d["time_s"], d["aircraft_index"], cells))
            for (s1, e1, a1, f1), (s2, e2, a2, f2) in itertools.combinations(spans, 2):
                if a1 != a2 and s1 < e2 and s2 < e1:  # both drops pending at the same time
                    self.assertFalse(f1 & f2, policy)

    def test_invalid_fleet_size_rejected(self):
        with self.assertRaises(ValueError):
            run_geographic_integrated(self._fleet(0), policy="no_intervention")

    def test_unknown_observation_mode_rejected(self):
        with self.assertRaises(ValueError):
            run_geographic_integrated(self.scenario, policy="no_intervention", observation_mode="delayed")


    def test_future_forcing_cannot_change_observed_fire_state(self):
        """Burning/burned must not reveal whether a future wind lets the front cross an edge."""
        base = growing_fire_scenario()
        moisture = base["fire"]["moisture"]

        def with_late_wind(wind):
            scenario = deepcopy(base)
            scenario["environment"] = {"schema_version": ENVIRONMENT_SCHEMA_VERSION,
                                       "weather_mode": "synthetic_intervals", "forcing": [
                {"start_s": 0.0, "end_s": 3300.0, "available_at_s": 0.0,
                 "midflame_wind_to_m_s": [3.0, 0.0], "moisture": moisture},
                {"start_s": 3300.0, "end_s": 3600.0, "available_at_s": 3300.0,
                 "midflame_wind_to_m_s": wind, "moisture": moisture}]}
            return scenario

        steady, reversed_late = with_late_wind([3.0, 0.0]), with_late_wind([-8.0, 0.0])
        for steps in (50, 52, 54):  # decisions at 3000-3240 s, before the late interval starts
            first = run_geographic_integrated(steady, "no_intervention", _stop_after_steps=steps)
            second = run_geographic_integrated(reversed_late, "no_intervention", _stop_after_steps=steps)
            self.assertLess(first["pending_observation"]["time_s"], 3300.0)
            self.assertEqual(first["pending_observation"]["fire"], second["pending_observation"]["fire"])

    def test_truncated_episode_reaching_the_horizon_is_scored_at_the_horizon(self):
        """The step driver (training, RL evaluation) and the batch driver (baselines) agree."""
        scenario = growing_fire_scenario()
        scenario["fire"]["wind_to_m_s"] = [4.0, 0.0]
        scenario["fire"]["burn_mask"] = [[abs(r - 20) <= 1 and abs(c - 5) <= 1 for c in range(40)] for r in range(40)]

        def hold_to_the_end(truncated):
            episode = geographic_episode(scenario, "no_intervention", truncated=truncated, record_observations=False)
            next(episode)
            try:
                while True:
                    episode.send({"type": "HOLD"})
            except StopIteration as finished:
                return finished.value

        full, stepped = hold_to_the_end(False), hold_to_the_end(True)
        self.assertIsNone(stepped["pending_observation"])
        self.assertEqual(stepped["outcomes"], full["outcomes"])
        self.assertEqual(stepped["arrival_time_s"], full["arrival_time_s"])


if __name__ == "__main__":
    unittest.main()
