"""Geographic drop caps and the effects of water above three drops."""

from copy import deepcopy
import json
import unittest

from pandion.geographic_integrated import geographic_episode, run_geographic_integrated
from pandion.policies import candidate_cells
from tests.test_water_on_burning import REFILL, burning_corridor


def stack(scenario, cell, count):
    drop = {"type": "DROP", "target_m": [100.0 * cell, 0.0]}
    actions = [drop, REFILL] * (count - 1) + [drop] if count else []
    return run_geographic_integrated(scenario, "no_intervention", _actions=actions)


class TestDropCap(unittest.TestCase):
    def test_omitted_cap_matches_explicit_legacy_cap(self):
        for burning_cells, cell in ((False, 1), (True, 0)):
            with self.subTest(burning_cells=burning_cells):
                implicit = burning_corridor(6.5)
                if not burning_cells:
                    del implicit["water_model"]
                explicit = deepcopy(implicit)
                explicit.setdefault("water_model", {})["max_stacked_drops"] = 3
                replay = stack(implicit, cell, 5)
                self.assertEqual(json.dumps(replay), json.dumps(stack(explicit, cell, 5)))
                self.assertEqual([drop["treated_cells_count"] for drop in replay["drops"]], [1, 1, 1, 0, 0])

    def test_finite_cap_applies_to_both_stacks(self):
        scenario = burning_corridor(6.5)
        scenario["water_model"]["max_stacked_drops"] = 2
        for cell in (0, 1):
            with self.subTest(cell=cell):
                replay = stack(scenario, cell, 3)
                self.assertEqual([drop["treated_cells_count"] for drop in replay["drops"]], [1, 1, 0])
                self.assertEqual(replay["decisions"][0]["observation"]["max_stacked_drops"], 2)

    def test_extra_pre_wetting_drops_add_breaching_delay(self):
        # The eastward GR2 head at 6.5 m/s exceeds the three-drop hold limit.
        scenario = burning_corridor(6.5)
        scenario["water_model"] = {"max_stacked_drops": None}
        dry_arrival = stack(scenario, 1, 0)["arrival_time_s"][0][1]
        stacks = {n: stack(scenario, 1, n) for n in (3, 4, 5)}
        three_arrival = stacks[3]["arrival_time_s"][0][1]
        delay_per_drop = (three_arrival - dry_arrival) / 3
        self.assertGreater(delay_per_drop, 0)
        for n in (4, 5):
            with self.subTest(drops=n):
                replay = stacks[n]
                self.assertEqual([drop["treated_cells"] for drop in replay["drops"]], [[[0, 1]]] * n)
                obs = replay["decisions"][2 * n - 1]["observation"]
                self.assertIsNone(obs["max_stacked_drops"])
                self.assertEqual(obs["known_treated_cells"], [[0, 1]] * n)
                self.assertAlmostEqual(replay["arrival_time_s"][0][1] - three_arrival,
                                       (n - 3) * delay_per_drop, places=5)

    def test_fourth_direct_drop_lands_without_changing_the_fire(self):
        # At 4 m/s the third drop holds the edge. At 6.5 m/s the edge breaches.
        for wind in (4.0, 6.5):
            with self.subTest(wind=wind):
                scenario = burning_corridor(wind)
                scenario["water_model"]["max_stacked_drops"] = None
                three = stack(scenario, 0, 3)
                four = stack(scenario, 0, 4)
                dry = stack(scenario, 0, 0)
                expected = None if wind == 4.0 else dry["arrival_time_s"][0][1]
                self.assertEqual(four["arrival_time_s"][0][1], expected)
                self.assertEqual(four["arrival_time_s"], three["arrival_time_s"])
                self.assertEqual(four["burnout_time_s"], three["burnout_time_s"])
                self.assertTrue(four["drops"][-1]["burning_drop"])
                self.assertEqual(four["drops"][-1]["treated_cells"], [[0, 0]])
                obs = four["decisions"][7]["observation"]
                self.assertIsNone(obs["max_stacked_drops"])
                self.assertEqual(obs["known_burning_drops"], [[0, 0]] * 4)

    def test_invalid_caps_are_rejected(self):
        for cap in (True, False, 0, -1, 3.0, "3"):
            with self.subTest(cap=cap):
                scenario = burning_corridor()
                scenario["water_model"]["max_stacked_drops"] = cap
                with self.assertRaisesRegex(ValueError, r"water_model\.max_stacked_drops"):
                    run_geographic_integrated(scenario, "heuristic")

    def test_unlimited_stacking_requires_positive_finite_refill_time(self):
        scenario = burning_corridor(6.5)
        scenario["aircraft"].update(initial_position_utm=[100, 0], water_scooping_utm=[100, 0],
                                    refill_service_time_s=0)
        del scenario["water_model"]
        replay = run_geographic_integrated(scenario, "heuristic", _stop_after_steps=12)
        self.assertEqual(len(replay["drops"]), 3)
        self.assertGreater(replay["pending_observation"]["time_s"], 0)

        scenario["water_model"] = {"max_stacked_drops": None}
        for refill_time in (0, -1, float("inf"), float("nan"), None, "invalid"):
            with self.subTest(refill_time=refill_time):
                scenario["aircraft"]["refill_service_time_s"] = refill_time
                with self.assertRaisesRegex(ValueError, r"aircraft\.refill_service_time_s"):
                    run_geographic_integrated(scenario, "heuristic", _stop_after_steps=12)

    def test_candidates_keep_both_kinds_of_stacked_cells_when_unlimited(self):
        scenario = burning_corridor(6.5)
        scenario["water_model"]["max_stacked_drops"] = None
        obs, _, _ = next(geographic_episode(scenario, "heuristic"))
        obs["known_burning_drops"] = [[0, 0]] * 4
        obs["known_treated_cells"] = [[0, 1]] * 4
        self.assertEqual(candidate_cells(obs), [[0, 0], [0, 1]])
        obs["max_stacked_drops"] = 3
        self.assertEqual(candidate_cells(obs), [])


if __name__ == "__main__":
    unittest.main()
