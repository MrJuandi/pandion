import unittest

from pandion.observation import build_reports, observe


class ObservationTests(unittest.TestCase):
    def test_latency_coverage_and_age(self):
        arrivals = [[0, 400], [200, None]]
        schedule = [{"source_id": "test", "observed_at_s": 250, "available_at_s": 300,
                     "coverage_cells": [[0, 0], [0, 1]]}]
        reports = build_reports(arrivals, schedule)
        self.assertEqual(observe(reports, 299, 2, 2)["cells"], [[None, None], [None, None]])
        view = observe(reports, 350, 2, 2)
        self.assertTrue(view["cells"][0][0]["fire_detected"])
        self.assertFalse(view["cells"][0][1]["fire_detected"])
        self.assertIsNone(view["cells"][1][0])
        self.assertEqual(view["cells"][0][0]["age_s"], 100)

    def test_hidden_state_and_future_fire_do_not_leak(self):
        schedule = [{"source_id": "test", "observed_at_s": 100, "available_at_s": 120,
                     "coverage_cells": [[0, 0]]}]
        first = observe(build_reports([[50, None], [None, None]], schedule), 200, 2, 2)
        hidden_changed = observe(build_reports([[50, 0], [0, 0]], schedule), 200, 2, 2)
        future_changed = observe(build_reports([[50, 0], [0, 0]], schedule), 110, 2, 2)
        self.assertEqual(first, hidden_changed)
        self.assertIsNone(future_changed["cells"][0][0])

    def test_latest_delivered_report_wins(self):
        arrivals = [[150]]
        schedule = [
            {"source_id": "old", "observed_at_s": 100, "available_at_s": 110, "coverage_cells": [[0, 0]]},
            {"source_id": "new", "observed_at_s": 200, "available_at_s": 250, "coverage_cells": [[0, 0]]},
        ]
        reports = build_reports(arrivals, schedule)
        self.assertFalse(observe(reports, 249, 1, 1)["cells"][0][0]["fire_detected"])
        self.assertTrue(observe(reports, 250, 1, 1)["cells"][0][0]["fire_detected"])


if __name__ == "__main__":
    unittest.main()
