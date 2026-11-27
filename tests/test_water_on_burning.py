"""Opt-in direct attack on a geographic burning front (O-20)."""

from copy import deepcopy
from hashlib import sha256
import json
import unittest

from pandion.action_features import ObservedFireBehavior
from pandion.geographic_integrated import geographic_episode, run_geographic_integrated
from pandion.policies import MAX_STACKED_DROPS, candidate_cells, choose_order


HOLD = {"type": "HOLD"}
REFILL = {"type": "RETURN_TO_REFILL"}
DROP = {"type": "DROP", "target_m": [0.0, 0.0]}


def corridor(wind=1.0):
    """Two GR2 cells with one initial ignition at the upwind end."""
    return {
        "grid": {"rows": 1, "cols": 2, "cell_size_m": 100},
        "fire": {
            "fuel_grid": [[102, 102]],
            "burn_mask": [[True, False]],
            "horizon_s": 900.0,
            "wind_to_m_s": [wind, 0.0],
        },
        "aircraft": {"initial_position_utm": [80.0, 0.0], "refill_service_time_s": 1.0},
    }


def burning_corridor(wind=1.0):
    scenario = corridor(wind)
    scenario["water_model"] = {"burning_cells": True}
    scenario["fire"]["moisture"] = {
        "dead_1h": 0.03, "dead_10h": 0.04, "dead_100h": 0.05,
        "dead_herbaceous": 0.03, "live_herbaceous": 0.30, "live_woody": 0.60,
    }
    return scenario


class TestBurningWater(unittest.TestCase):
    def test_flag_off_matches_legacy_replay(self):
        scenario = corridor()
        implicit = run_geographic_integrated(scenario, "heuristic")
        scenario["water_model"] = {"burning_cells": False, "candidates": "burning_only"}
        explicit = run_geographic_integrated(scenario, "heuristic")
        self.assertEqual(implicit, explicit)
        # Captured from the pre-prototype simulator: area, all drops, and the first observation.
        legacy = (implicit["affected_area_ha"], implicit["drops"], implicit["decisions"][0]["observation"])
        self.assertEqual(sha256(json.dumps(legacy, sort_keys=True).encode()).hexdigest(),
                         "79e3a3fe44be2c729345ef9d69ff9cb0b093edfeeabd0cf0536f7861ecb32f79")

    def test_one_drop_cancels_a_queued_low_intensity_arrival(self):
        scenario = burning_corridor()
        dry = run_geographic_integrated(scenario, "no_intervention")
        wet = run_geographic_integrated(scenario, "no_intervention", _actions=[DROP])
        self.assertIsNotNone(dry["arrival_time_s"][0][1])
        self.assertEqual(wet["arrival_time_s"], [[0.0, None]])
        after_drop = wet["decisions"][1]["observation"]
        self.assertEqual(after_drop["known_burning_drops"], [[0, 0]])
        self.assertEqual(after_drop["known_treated_cells"], [])
        cell = after_drop["fire"]["cells"][0][0]
        self.assertEqual(cell["fire_state"], "burning")
        self.assertEqual(cell["visible_state"], "burned")
        self.assertIsNone(cell["thermal_byte"])
        after_burnout = wet["decisions"][-1]["observation"]["fire"]["cells"][0][0]
        self.assertEqual(after_burnout["fire_state"], "burned")

    def test_medium_intensity_needs_three_drops_and_stack_is_capped(self):
        # The GR2 eastward head at 4 m/s is about 2,465 kW/m.
        scenario = burning_corridor(4.0)
        dry = run_geographic_integrated(scenario, "no_intervention")
        for count in (1, 2, 3, 4):
            with self.subTest(count=count):
                wet = run_geographic_integrated(scenario, "no_intervention",
                                                _actions=[DROP, REFILL] * (count - 1) + [DROP])
                expected = dry["arrival_time_s"][0][1] if count < 3 else None
                self.assertEqual(wet["arrival_time_s"][0][1], expected)
                self.assertEqual([drop["burning_drop"] for drop in wet["drops"]],
                                 [i < MAX_STACKED_DROPS for i in range(count)])
                self.assertEqual(wet["drops"][-1]["treated_cells_count"], int(count <= MAX_STACKED_DROPS))
                after_drop = wet["decisions"][2 * count - 1]["observation"]
                self.assertEqual(after_drop["known_burning_drops"], [[0, 0]] * min(count, MAX_STACKED_DROPS))

    def test_high_intensity_arrival_is_neither_held_nor_delayed(self):
        # At 6.5 m/s the head is about 4,033 kW/m.
        scenario = burning_corridor(6.5)
        dry = run_geographic_integrated(scenario, "no_intervention")
        wet = run_geographic_integrated(scenario, "no_intervention", _actions=[DROP, REFILL, DROP, REFILL, DROP])
        self.assertIsNotNone(dry["arrival_time_s"][0][1])
        self.assertEqual(wet["arrival_time_s"], dry["arrival_time_s"])
        self.assertEqual(wet["drops"][-1]["treated_cells_count"], 1)
        after_drop = wet["decisions"][5]["observation"]["fire"]["cells"][0][0]
        self.assertEqual(after_drop["visible_state"], "burning")

    def test_only_the_held_outgoing_direction_stops(self):
        scenario = burning_corridor(4.0)
        scenario["grid"]["cols"] = 3
        scenario["fire"]["fuel_grid"] = [[102, 102, 102]]
        scenario["fire"]["burn_mask"] = [[False, True, False]]
        scenario["fire"]["horizon_s"] = 7200.0
        obs, _, _ = next(geographic_episode(scenario, "heuristic"))
        behavior = ObservedFireBehavior()
        self.assertEqual(behavior.needed_from(obs, 0, 1, 0, -1), 1)
        self.assertEqual(behavior.needed_from(obs, 0, 1, 0, 1), 3)
        obs["water_on_burning_candidates"] = "burning_only"
        obs["known_burning_drops"] = [[0, 1]]
        self.assertEqual(choose_order("hold_aware_heuristic", obs, behavior=behavior),
                         {"type": "DROP", "target_m": [100.0, 0.0]})
        dry = run_geographic_integrated(scenario, "no_intervention")
        wet = run_geographic_integrated(scenario, "no_intervention", _actions=[{"type": "DROP", "target_m": [100, 0]}])
        self.assertIsNotNone(dry["arrival_time_s"][0][0])
        self.assertIsNone(wet["arrival_time_s"][0][0])
        self.assertEqual(wet["arrival_time_s"][0][2], dry["arrival_time_s"][0][2])
        cell = wet["decisions"][1]["observation"]["fire"]["cells"][0][1]
        self.assertEqual(cell["visible_state"], "burning")

    def test_drop_on_burned_out_cell_is_recorded_without_treatment(self):
        scenario = burning_corridor(6.5)
        wet = run_geographic_integrated(scenario, "no_intervention", _actions=[HOLD, HOLD, DROP])
        after_drop = wet["decisions"][3]["observation"]
        self.assertEqual(after_drop["fire"]["cells"][0][0]["fire_state"], "burned")
        self.assertEqual(after_drop["known_burning_drops"], [])
        self.assertFalse(wet["drops"][0]["burning_drop"])
        self.assertEqual(wet["drops"][0]["treated_cells_count"], 0)


class TestBurningCandidates(unittest.TestCase):
    def test_opt_in_and_stack_limit(self):
        obs, _, _ = next(geographic_episode(corridor(), "heuristic"))
        self.assertEqual(candidate_cells(obs), [[0, 1]])
        obs, _, _ = next(geographic_episode(burning_corridor(), "heuristic"))
        self.assertIs(obs["water_on_burning"], True)
        self.assertEqual(candidate_cells(obs), [[0, 0], [0, 1]])
        obs["known_burning_drops"] = [[0, 0]] * (MAX_STACKED_DROPS - 1)
        self.assertEqual(candidate_cells(obs), [[0, 0], [0, 1]])
        obs["known_burning_drops"].append([0, 0])
        self.assertEqual(candidate_cells(obs), [[0, 1]])

    def test_burning_only_and_all_rules_can_target_an_isolated_ignition(self):
        scenario = burning_corridor()
        scenario["water_model"]["candidates"] = "burning_only"
        obs, _, _ = next(geographic_episode(scenario, "heuristic"))
        self.assertEqual(candidate_cells(obs), [[0, 0]])
        for policy in ("heuristic", "hold_aware_heuristic", "blind_hold_aware_heuristic"):
            with self.subTest(policy=policy):
                self.assertEqual(choose_order(policy, obs), DROP)
        obs["known_burning_drops"] = [[0, 0]]
        self.assertEqual(choose_order("hold_aware_heuristic", obs), HOLD)
        self.assertEqual(choose_order("blind_hold_aware_heuristic", obs), DROP)

    def test_direct_targets_respect_observed_state_fuel_and_fleet_claims(self):
        obs, _, _ = next(geographic_episode(burning_corridor(), "heuristic"))
        cell = obs["fire"]["cells"][0][0]
        cell["visible_state"] = "burned"
        self.assertEqual(candidate_cells(obs), [[0, 1]])
        del cell["visible_state"]
        self.assertEqual(candidate_cells(obs), [[0, 0], [0, 1]])
        water = deepcopy(obs)
        water["fuel_model"][0][1] = "98"
        water["water_on_burning_candidates"] = "burning_only"
        self.assertEqual(candidate_cells(water), [])
        obs["fleet"] = [{"index": 1, "phase": "transit_to_drop", "target_m": [0, 0]}]
        self.assertEqual(candidate_cells(obs), [])


if __name__ == "__main__":
    unittest.main()
