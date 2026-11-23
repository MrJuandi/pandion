import json
import tempfile
import unittest
from pathlib import Path

from pandion.viewer import build_viewer_html, write_viewer_html


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
        self.assertIn("Pandion Replay Viewer", html)
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
        self.assertIn("PandionScene3D", page)
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


class DisplayImageryTests(unittest.TestCase):
    """Sentinel-2 base images reach the viewer only for exactly matching grids (display only)."""

    def setUp(self):
        import numpy as np
        from hashlib import sha256
        self.np = np
        self.tmp = tempfile.TemporaryDirectory(); self.dir = Path(self.tmp.name)
        # 2 x 3 cells at 2 px per cell; row 0 = south, as stored by imagery_acquire.
        self.image = np.zeros((4, 6, 3), dtype=np.uint8); self.image[0] = 200; self.image[-1] = 50
        np.save(self.dir / "act.npy", self.image)
        grid = {"rows": 2, "cols": 3, "cell_size_m": 100.0, "origin_utm_easting_m": 1000.0, "origin_utm_northing_m": 2000.0}
        self.source = {"grid": grid, "crs": "EPSG:25831", "px_per_cell": 2, "pixel_size_m": 50.0, "array": "act.npy",
                       "array_sha256": sha256((self.dir / "act.npy").read_bytes()).hexdigest(),
                       "primary_datetime_utc": "2022-04-30T10:59:32Z", "scenes_used": [{}, {}],
                       "attribution": "Contains modified Copernicus Sentinel data 2022", "collection": "sentinel-2-l2a"}
        self.write_manifest()
        self.replay = {"scenario_id": "s", "policy": "p", "horizon_s": 0, "grid": {"rows": 2, "cols": 3, "cell_size_m": 100.0},
                       "arrival_time_s": [[None] * 3 for _ in range(2)], "origin_utm": [1000.0, 2000.0],
                       "crs": {"code": "EPSG:25831"}}

    def tearDown(self):
        self.tmp.cleanup()

    def write_manifest(self):
        (self.dir / "sources.json").write_text(json.dumps({"schema_version": "nightjar-imagery-sources-0.1",
                                                           "activations": {"ACT": self.source}}))

    def attach(self, replay):
        from pandion.replay_view import normalize_replay
        from pandion.viewer import attach_display_imagery
        normalized = [normalize_replay(replay)]
        return normalized[0], attach_display_imagery(normalized, [self.dir])

    def test_matching_grid_embeds_north_up_png_once(self):
        import base64, struct, zlib
        replay, imagery = self.attach(self.replay)
        self.assertEqual(list(imagery), [replay["display_imagery_id"]])
        payload = imagery[replay["display_imagery_id"]]
        self.assertEqual(payload["attribution"], "Contains modified Copernicus Sentinel data 2022")
        png = base64.b64decode(payload["png_data_uri"].split(",", 1)[1])
        self.assertEqual(png[:8], b"\x89PNG\r\n\x1a\n")
        width, height = struct.unpack(">II", png[16:24])
        self.assertEqual((width, height), (6, 4))
        raw = zlib.decompress(png[png.index(b"IDAT") + 4:png.index(b"IEND") - 8])
        stride = 1 + 6 * 3
        self.assertEqual(raw[1], 50)                  # first PNG row = northern edge (last stored row)
        self.assertEqual(raw[3 * stride + 1], 200)    # last PNG row = southern edge

    def test_other_origin_or_crs_gets_no_image(self):
        for change in ({"origin_utm": [1000.0, 2100.0]}, {"crs": {"code": "EPSG:25830"}}):
            replay, imagery = self.attach({**self.replay, **change})
            self.assertEqual(imagery, {})
            self.assertNotIn("display_imagery_id", replay)

    def test_tampered_array_is_rejected(self):
        self.source["array_sha256"] = "0" * 64; self.write_manifest()
        with self.assertRaises(ValueError):
            self.attach(self.replay)

    def test_page_declares_imagery_and_attribution_slot(self):
        page = build_viewer_html([self.replay], imagery_dirs=[self.dir])
        self.assertIn("const IMAGERY = {", page)
        self.assertIn("data:image/png;base64,", page)
        self.assertIn('id="map-attribution"', page)
        self.assertIn("const IMAGERY = {}", build_viewer_html([self.replay]))


if __name__ == "__main__":
    unittest.main()
