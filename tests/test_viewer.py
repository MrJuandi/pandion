import json
import tempfile
import unittest
from pathlib import Path

from nightjar.viewer import build_viewer_html, write_viewer_html


class Viewer3DTests(unittest.TestCase):
    def setUp(self):
        self.sample_replay_1 = {
            "scenario_id": "discrimination-reserved-00",
            "policy": "ppo",
            "horizon_s": 1200,
            "cell_size_m": 50,
            "affected_area_ha": 7.25,
            "aircraft_metrics": {
                "flight_s": 92.0,
                "water_dropped_l": 42959.0,
                "rejected_orders": 0,
            },
            "arrival_time_s": [[None] * 11 for _ in range(11)],
            "water_barrier_cells": [],
            "drops": [
                {
                    "time_s": 10.0,
                    "position_m": [500.0, 450.0],
                    "newly_blocked_cells": [[9, 10]],
                }
            ],
            "decisions": [
                {
                    "time_s": 0,
                    "order": {"type": "DROP", "target_m": [500.0, 450.0]},
                    "candidate_cells": [[9, 10]],
                    "observation": {
                        "aircraft": {
                            "position_m": [500.0, 500.0],
                            "water_l": 6137.0,
                            "phase": "idle",
                        },
                        "water_source_m": [500.0, 250.0],
                        "fuel_model": [["FM1"] * 11 for _ in range(11)],
                        "fire_reports": [[5, 5]],
                    },
                }
            ],
        }

        self.sample_replay_2 = {
            "scenario_id": "discrimination-reserved-00",
            "policy": "heuristic",
            "horizon_s": 1200,
            "cell_size_m": 50,
            "affected_area_ha": 6.50,
            "aircraft_metrics": {
                "flight_s": 46.0,
                "water_dropped_l": 42959.0,
                "rejected_orders": 0,
            },
            "arrival_time_s": [[None] * 11 for _ in range(11)],
            "water_barrier_cells": [],
            "drops": [],
            "decisions": [],
        }

    def test_build_viewer_html_rejects_empty(self):
        with self.assertRaises(ValueError):
            build_viewer_html([])
        with tempfile.TemporaryDirectory() as tmpdir, self.assertRaises(ValueError):
            write_viewer_html([], Path(tmpdir) / "empty.html")

    def test_build_viewer_html_single_replay(self):
        html = build_viewer_html([self.sample_replay_1])
        self.assertIn("<!DOCTYPE html>", html)
        self.assertIn("discrimination-reserved-00", html)
        self.assertIn("Nightjar Replay Viewer", html)
        self.assertIn("time-slider", html)
        self.assertIn("canvas", html)

    def test_build_viewer_html_multi_replay(self):
        html = build_viewer_html([self.sample_replay_1, self.sample_replay_2], title="Comparison")
        self.assertIn("Comparison", html)
        self.assertIn("ppo", html)
        self.assertIn("heuristic", html)
        self.assertIn("Side-by-side", html)

    def test_write_viewer_html_creates_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = Path(tmpdir) / "sub" / "viewer.html"
            write_viewer_html([self.sample_replay_1], out_file)
            self.assertTrue(out_file.exists())
            content = out_file.read_text(encoding="utf-8")
            self.assertIn("<!DOCTYPE html>", content)
            self.assertIn("discrimination-reserved-00", content)

    def test_offline_webgl_bundle_includes_license_and_fallback(self):
        page = build_viewer_html([self.sample_replay_1])
        self.assertIn("Copyright © 2010-2021 three.js authors", page)
        self.assertIn("NightjarScene3D", page)
        self.assertIn("OrbitControls", page)
        self.assertIn("canvas-fallback", page)
        self.assertNotIn('<script src=', page)
        self.assertIn('id="compare-policy-select"', page)

    def test_embedded_labels_cannot_inject_html_or_script(self):
        replay = {**self.sample_replay_1, "policy": '</script><script>alert(1)</script> __JS__'}
        html = build_viewer_html([replay], title='<img src=x onerror=alert(1)>')
        self.assertNotIn('<img src=x', html)
        self.assertNotIn('</script><script>alert(1)', html)
        self.assertIn('\\u003c/script\\u003e', html)
        self.assertIn(' __JS__"', html)


if __name__ == "__main__":
    unittest.main()
