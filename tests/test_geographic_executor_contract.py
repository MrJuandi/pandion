"""Executor-side contract of the geographic simulator: tank use and observation coverage."""

import unittest

from pandion.geographic_integrated import geographic_episode, run_geographic_integrated


def corridor(cols=3):
    return {
        "grid": {"rows": 1, "cols": cols, "cell_size_m": 100},
        "fire": {"fuel_grid": [[102] * cols], "burn_mask": [[True] + [False] * (cols - 1)],
                 "horizon_s": 900.0, "wind_to_m_s": [1.0, 0.0]},
        "aircraft": {"initial_position_utm": [80.0, 0.0], "refill_service_time_s": 1.0},
    }


class ExecutorContractTests(unittest.TestCase):
    def test_second_drop_without_refill_is_rejected(self):
        drop = {"type": "DROP", "target_m": [100.0, 0.0]}
        with self.assertRaisesRegex(ValueError, "DROP ordered"):
            run_geographic_integrated(corridor(), "no_intervention", _actions=[drop, drop])
        # A refill in between makes the same sequence valid.
        replay = run_geographic_integrated(corridor(), "no_intervention",
                                           _actions=[drop, {"type": "RETURN_TO_REFILL"}, drop])
        self.assertEqual(len(replay["drops"]), 2)

    def test_scheduled_capture_keeps_its_coverage(self):
        scenario = corridor()
        scenario["observation_mode"] = "scheduled"
        scenario["observation_schedule"] = [
            {"observed_at_s": 0.0, "available_at_s": 0.0, "coverage_cells": [[0, 0]]}]
        episode = geographic_episode(scenario, "no_intervention")
        observation, _, _ = next(episode)
        cells = observation["fire"]["cells"][0]
        self.assertTrue(cells[0]["fire_detected"])
        self.assertIsNone(cells[1])  # never covered: unknown, not unburned
        self.assertIsNone(cells[2])


if __name__ == "__main__":
    unittest.main()
