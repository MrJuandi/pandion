import random
import unittest

from nightjar.policies import candidate_cells, choose_order


def observation(*, cells=None, phase="idle", water=100.0, position=(0.0, 0.0), wind=(1.0, 0.0)):
    if cells is None:
        cells = [
            [dict(fire_detected=False), dict(fire_detected=True), dict(fire_detected=False)],
            [dict(fire_detected=False), dict(fire_detected=False), dict(fire_detected=False)],
            [dict(fire_detected=None), dict(fire_detected=False), None],
        ]
    return {
        "fire": {"cells": cells},
        "fuel_model": [["FM1"] * 3 for _ in range(3)],
        "aircraft": {"phase": phase, "water_l": water, "position_m": list(position)},
        "midflame_wind_to_m_s": list(wind),
        "cell_size_m": 10.0,
    }


class PolicyTests(unittest.TestCase):
    def test_candidates_require_observed_unburned_fuel_and_orthogonal_adjacency(self):
        obs = observation()
        obs["fuel_model"][1][1] = "NONE"
        obs["fire"]["cells"][0][0] = None
        self.assertEqual(candidate_cells(obs), [[0, 2]])
        # Center is burning's diagonal and must not be treated as adjacent.
        self.assertNotIn([0, 0], candidate_cells(obs))

    def test_no_intervention_and_empty_unknown_observation_hold(self):
        self.assertEqual(choose_order("no_intervention", observation()), {"type": "HOLD"})
        obs = observation(cells=[[None] * 3 for _ in range(3)])
        self.assertEqual(candidate_cells(obs), [])
        self.assertEqual(choose_order("heuristic", obs), {"type": "HOLD"})

    def test_busy_aircraft_holds_and_empty_idle_aircraft_refills(self):
        self.assertEqual(choose_order("heuristic", observation(phase="to_drop")), {"type": "HOLD"})
        self.assertEqual(choose_order("random_valid", observation(water=0), rng=random.Random(1)),
                         {"type": "RETURN_TO_REFILL"})

    def test_random_policy_is_seeded_and_targets_a_valid_candidate(self):
        obs = observation()
        first = choose_order("random_valid", obs, rng=random.Random(17))
        second = choose_order("random_valid", obs, rng=random.Random(17))
        self.assertEqual(first, second)
        self.assertIn(first["target_m"], [[20.0, 0.0], [10.0, 10.0]])

    def test_heuristic_chooses_downwind_then_proximity_then_coordinate(self):
        obs = observation(position=(20.0, 0.0), wind=(1.0, 0.0))
        # The candidate east of the observed fire is downwind; south is crosswind.
        self.assertEqual(choose_order("heuristic", obs)["target_m"], [20.0, 0.0])
        obs["midflame_wind_to_m_s"] = [0.0, -1.0]
        self.assertEqual(choose_order("heuristic", obs)["target_m"], [20.0, 0.0])
        obs["midflame_wind_to_m_s"] = [0.0, 1.0]
        self.assertEqual(choose_order("heuristic", obs)["target_m"], [10.0, 10.0])
        obs["fire"]["cells"][0][1] = {"fire_detected": False}
        obs["fire"]["cells"][1][1] = {"fire_detected": True}
        self.assertEqual(choose_order("heuristic", obs)["target_m"], [10.0, 20.0])

    def test_policy_output_is_independent_of_unobserved_future_state(self):
        first = observation()
        second = observation()
        # Latent/future fields are deliberately not part of the policy interface.
        first["latent_arrival_time_s"] = [[0, 50, 100]]
        second["latent_arrival_time_s"] = [[999, 0, 0]]
        self.assertEqual(candidate_cells(first), candidate_cells(second))
        self.assertEqual(choose_order("heuristic", first), choose_order("heuristic", second))

    def test_known_treated_cell_is_not_selected_again(self):
        obs = observation()
        obs["known_treated_cells"] = [[0, 2]]
        self.assertEqual(candidate_cells(obs), [[0, 0], [1, 1]])
        self.assertEqual(choose_order("heuristic", obs)["target_m"], [10.0, 10.0])

    def test_missing_random_generator_and_unknown_policy_are_rejected(self):
        with self.assertRaises(ValueError):
            choose_order("random_valid", observation())
        with self.assertRaises(ValueError):
            choose_order("oracle", observation())


if __name__ == "__main__":
    unittest.main()
