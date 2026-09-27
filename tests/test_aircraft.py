import json
import unittest
from pathlib import Path

from nightjar.aircraft import CAPACITY_L, run_episode


EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


class AircraftEpisodeTests(unittest.TestCase):
    def setUp(self):
        self.scenario = json.loads((EXAMPLES / "synthetic_aircraft.json").read_text())
        self.orders = json.loads((EXAMPLES / "synthetic_orders.json").read_text())

    def test_drop_return_refill_and_deterministic_replay(self):
        replay = run_episode(self.scenario, self.orders)
        self.assertEqual(replay, run_episode(self.scenario, self.orders))
        self.assertEqual([event["kind"] for event in replay["events"]], ["drop", "refill_started", "refill_complete"])
        self.assertEqual(replay["events"][0]["time_s"], 60)
        self.assertAlmostEqual(replay["events"][1]["time_s"], 60 + 3000 / 55, places=5)
        self.assertAlmostEqual(replay["events"][2]["time_s"], 240 + 3000 / 55, places=5)
        self.assertEqual(replay["metrics"]["water_dropped_l"], CAPACITY_L)
        self.assertEqual(replay["final_state"]["water_l"], CAPACITY_L)
        self.assertEqual(replay["final_state"]["position_m"], [0, 0])

    def test_busy_aircraft_rejects_new_mission_without_teleporting(self):
        self.orders[0]["target_m"] = [12000, 0]
        self.orders[1] = {"type": "RETURN_TO_REFILL"}
        replay = run_episode(self.scenario, self.orders)
        self.assertEqual(replay["decisions"][1]["reason"], "aircraft_busy")
        self.assertEqual(replay["decisions"][1]["observation"]["aircraft"]["position_m"], [3000, 0])
        self.assertEqual(replay["events"][0]["kind"], "order_rejected")

    def test_no_water_cannot_drop(self):
        self.scenario["aircraft"]["initial_water_l"] = 0
        replay = run_episode(self.scenario, self.orders[:1])
        self.assertEqual(replay["decisions"][0]["reason"], "no_water")
        self.assertEqual(replay["metrics"]["water_dropped_l"], 0)

    def test_rejects_wrong_provenance_and_overfill(self):
        self.scenario["provenance"] = "observed"
        with self.assertRaises(ValueError):
            run_episode(self.scenario, [])
        self.scenario["provenance"] = "synthetic"
        self.scenario["aircraft"]["initial_water_l"] = CAPACITY_L + 1
        with self.assertRaises(ValueError):
            run_episode(self.scenario, [])


if __name__ == "__main__":
    unittest.main()
