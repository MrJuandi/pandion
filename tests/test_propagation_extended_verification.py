"""Mathematical, input API, and protocol tests; no propagation simulations run."""
from datetime import datetime, timezone
import json
from math import radians, sin, sqrt
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import platform
from importlib import metadata

from nightjar.propagation_extended_verification import (
    CASES,
    CELLS_M,
    HETERO_TARGETS,
    PROTOCOL,
    TARGETS,
    TOL,
    _config,
    _domain_failure_records,
    _failure_records,
    _levelset,
    _make_levelset_cubes,
    _native_head_rate,
    _point_reference,
    _passes,
    _scenario,
    _validate_freeze,
    _behavior,
    ellipse_arrival,
    freeze,
)


class ExtendedPropagationInvariantTests(unittest.TestCase):
    def test_circle_origin_axial_heading_and_backing_arrivals(self):
        self.assertEqual(ellipse_arrival(0.0, 0.0, 2.0, 2.0, 0.0, 0.0), 0.0)
        self.assertAlmostEqual(ellipse_arrival(10.0, 0.0, 2.0, 2.0, 0.0, 0.0), 5.0)
        self.assertAlmostEqual(ellipse_arrival(0.0, 100.0, 2.0, 2.0, 0.0, 0.0), 5.0)
        self.assertAlmostEqual(ellipse_arrival(20.0, 0.0, 5.0, 4.0, 0.6, 3.0), 2.5)
        self.assertAlmostEqual(ellipse_arrival(-20.0, 0.0, 5.0, 4.0, 0.6, 3.0), 10.0)
        # Construct a known point on the translated ellipse at t=2.
        p, perpendicular = 6.0 + 5.0 * sqrt(2), 4.0 * sqrt(2)
        self.assertAlmostEqual(ellipse_arrival(p, perpendicular**2, 5.0, 4.0, 0.6, 3.0), 2.0)
        with self.assertRaises(ValueError):
            ellipse_arrival(1.0, 0.0, 0.0, 1.0, 0.0, 0.0)

    def test_declared_point_angles_and_tilted_plane_geometry(self):
        self.assertEqual(CELLS_M, (4.0, 2.0, 1.0))
        self.assertEqual([(case["upwind_deg"], case["aspect_deg"]) for case in CASES], [
            (0.0, 0.0), (270.0, 0.0), (225.0, 0.0), (0.0, 270.0), (180.0, 270.0)
        ])
        gx = -0.5 * sin(radians(270.0))
        self.assertAlmostEqual(gx, 0.5)

    def test_native_wind_units_and_head_rate_match_burn_cell_api(self):
        # burn_cell_as_head_fire accepts the same open U10 in km/h, performs
        # native U10->U20 conversion and midflame conversion, and returns m/min.
        for case in CASES:
            direct = _behavior(101, case)
            native_m_min = _native_head_rate(101, case)
            self.assertAlmostEqual(direct["head_m_min"], native_m_min, delta=abs(native_m_min) * 1e-5)
            self.assertAlmostEqual(direct["midflame_m_s"], direct["midflame_m_min"] / 60.0)
        east = next(case for case in CASES if case["id"] == "wind-east")
        scenario, _ = _scenario(1.0, east, horizon_s=100.0)
        expected = _behavior(101, east)["midflame_m_min"] / 60.0
        self.assertAlmostEqual(scenario["midflame_wind_to_m_s"][0], expected, places=7)
        self.assertAlmostEqual(scenario["midflame_wind_to_m_s"][1], 0.0, places=7)

    def test_surface_ellipse_reference_matches_independent_radial_law(self):
        from pyretechnics import surface_fire
        for case in CASES:
            behavior = _behavior(101, case)["behavior"]
            for xy in TARGETS:
                arrival, reference = _point_reference(case, xy)
                gx, gy = reference["slope_gradient"]
                q = (xy[0], xy[1], gx * xy[0] + gy * xy[1])
                distance = sqrt(sum(v * v for v in q))
                direction = tuple(v / distance for v in q)
                radial = surface_fire.calc_surface_fire_behavior_in_direction(behavior, direction)["spread_rate"]
                expected_seconds = distance / radial * 60.0
                self.assertAlmostEqual(arrival, expected_seconds, delta=expected_seconds * 1e-5)

    def test_second_layer_fuel_has_aligned_native_still_air_rate(self):
        calm = CASES[0]
        reference = _behavior(104, calm)["head_m_min"]
        native = _native_head_rate(104, calm)
        self.assertAlmostEqual(reference, native, delta=native * 1e-5)

    def test_heterogeneous_input_cubes_are_float32_and_ignite_full_column(self):
        case = {"id": "hetero-east", "wind_u10_kmh": 0.0, "upwind_deg": 0.0,
                "slope": 0.0, "aspect_deg": 0.0}
        scenario, _ = _scenario(4.0, case, heterogeneous=True, horizon_s=500.0)
        self.assertEqual(scenario["ignitions"], [[row, 30] for row in range(61)])
        cubes = _make_levelset_cubes(scenario, case)
        import numpy as np
        fuel_data = np.asarray(cubes["fuel_model"].data)
        self.assertEqual(str(fuel_data.dtype), "float32")
        self.assertEqual(fuel_data.shape, (1, 61, 61))
        self.assertEqual(float(fuel_data[0, 30, 39]), 101.0)
        self.assertEqual(float(fuel_data[0, 30, 40]), 104.0)
        self.assertEqual(HETERO_TARGETS, ((20.0, 0.0), (60.0, 0.0), (80.0, 0.0)))

    def test_failure_helpers_keep_every_engine_target_record(self):
        error = RuntimeError("engine failed")
        records = _failure_records("case", 2.0, TARGETS, ("dijkstra", "levelset"), error)
        self.assertEqual(len(records), len(TARGETS) * 2)
        self.assertEqual({(tuple(row["target_xy_m"]), row["engine"]) for row in records},
                         {(xy, engine) for xy in TARGETS for engine in ("dijkstra", "levelset")})
        self.assertTrue(all(row["pass"] is False and "RuntimeError" in row["error"] for row in records))
        domain = _domain_failure_records("case", HETERO_TARGETS, ("dijkstra", "levelset"), error,
                                         {xy: 1.0 for xy in HETERO_TARGETS})
        self.assertEqual(len(domain), len(HETERO_TARGETS) * 2)
        self.assertTrue(all(row["analytic_arrival_s"] == 1.0 and row["pass"] is False for row in domain))

    def test_strict_gate_rejects_equal_boundary_and_nonfinite(self):
        self.assertTrue(_passes(TOL - 1e-10))
        self.assertFalse(_passes(TOL))
        self.assertFalse(_passes(float("nan")))
        self.assertFalse(_passes(None))

    @unittest.skipUnless(PROTOCOL.exists(), "protocol document is created by coordinator")
    def test_fresh_freeze_validates_metadata_hashes_and_runtime_before_tamper(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "frozen.json"
            frozen = freeze(path)
            loaded = _validate_freeze(path)
            self.assertEqual(loaded["configuration"], _config())
            self.assertEqual(frozen["seed"], 0)
            self.assertEqual(frozen["schema_version"], "nightjar-propagation-extended-0.1")
            self.assertEqual(frozen["crs"], "LOCAL_METERS")
            self.assertEqual(frozen["provenance"], "synthetic numerical verification")
            self.assertEqual(frozen["synthetic_time_origin_utc"], "2000-01-01T00:00:00Z")
            timestamp = datetime.fromisoformat(frozen["created_at_utc"])
            self.assertIsNotNone(timestamp.tzinfo)
            self.assertEqual(timestamp.utcoffset(), timezone.utc.utcoffset(timestamp))
            self.assertEqual(frozen["units"]["wind_native"], "m/min at 20ft")
            self.assertEqual(frozen["runtime"]["engine_versions"]["pyretechnics"], "2026.8.10")
            self.assertEqual(frozen["runtime"]["python"], platform.python_version())
            self.assertEqual(frozen["runtime"]["platform"], platform.platform())
            self.assertEqual(frozen["runtime"]["engine_versions"], {
                name: metadata.version(name) for name in ("pyretechnics", "numpy", "scipy")
            })
            self.assertTrue({
                "nightjar/fire_standard.py", "nightjar/propagation_extended_verification.py",
                "docs/PROPAGATION_EXTENDED_PROTOCOL.md",
                "pyretechnics/burn_cells.py", "pyretechnics/surface_fire.py",
                "pyretechnics/eulerian_level_set.py", "pyretechnics/conversion.py",
                "pyretechnics/fuel_models.py",
                "pyretechnics/space_time_cube.py",
            }.issubset(frozen["code_sha256"]))
            from pyretechnics import surface_fire
            self.assertIn("pyretechnics/" + Path(surface_fire.__file__).name, frozen["code_sha256"])
            self.assertTrue(all(len(value) == 64 for value in frozen["code_sha256"].values()))
            self.assertEqual(frozen["configuration"]["horizon_safety_multiplier"], 1.25)
            with self.assertRaises(FileExistsError):
                freeze(path)
            changed = json.loads(path.read_text())
            changed["configuration"]["domain_half_extent_m"] = 121
            path.write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError, "configuration"):
                _validate_freeze(path)


if __name__ == "__main__":
    unittest.main()
