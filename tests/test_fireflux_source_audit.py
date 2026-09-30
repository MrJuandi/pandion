from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from nightjar import fireflux_source_audit as audit


class FilteredFireFluxAuditTests(unittest.TestCase):
    def fixture(self, root, bad_shape=False):
        import numpy as np

        arrays = {"TimeMean": np.array([-1., 0., 1.]), "ROS": np.array([999.]),
                  "ArrivalTime": np.array([999.]), "TimeW10ObstoDelete": np.array([None], dtype=object)}
        for key, columns in audit.OBSERVATION_COLUMNS.items():
            count = columns + 1 if bad_shape else columns
            arrays[key] = np.ones((3, count))
            arrays[key + "Std"] = np.ones((3, count)) * .5
        np.savez(root / "MainTower_Anel_25m.npz", **arrays)
        (root / "costes2022.pdf").write_bytes(b"synthetic method source")
        return {name: {**source, "sha256": sha256((root / name).read_bytes()).hexdigest()}
                for name, source in audit.SOURCE_FILES.items()}

    def test_only_observation_arrays_enter_report_without_unpickling(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(audit, "SOURCE_FILES", self.fixture(root)), \
                 patch.object(audit.subprocess, "run", side_effect=AssertionError("unexpected download")):
                report = audit.audit(root)
        self.assertEqual(set(report["observation_statistics"]), set(audit.OBSERVATION_COLUMNS))
        self.assertIn("ROS", report["excluded_arrays"])
        self.assertIn("ArrivalTime", report["excluded_arrays"])
        self.assertIn("TimeW10ObstoDelete", report["excluded_arrays"])
        self.assertIsNone(report["time_origin_utc"])
        self.assertIsNone(report["physical_validation_pass"])
        self.assertEqual(report["time_axis"]["count"], 3)

    def test_changed_source_and_invalid_shape_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pins = self.fixture(root, bad_shape=True)
            with patch.object(audit, "SOURCE_FILES", pins):
                with self.assertRaisesRegex(ValueError, "unexpected observation shape"):
                    audit.audit(root)
                (root / "costes2022.pdf").write_bytes(b"changed source")
                with self.assertRaisesRegex(ValueError, "source hash changed"):
                    audit.audit(root)

    def test_cached_fetch_does_not_invent_an_acquisition_timestamp(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(audit, "SOURCE_FILES", self.fixture(root)), \
                 patch.object(audit.subprocess, "run", side_effect=AssertionError("unexpected download")):
                audit.fetch_sources(root)
            manifest = json.loads((root / "filtered-source-manifest.json").read_text())
            self.assertEqual(manifest["downloaded_filenames_in_this_call"], [])
            self.assertNotIn("acquired_at_utc", manifest)

    def test_pinned_filtered_observations_when_available(self):
        root = Path(__file__).resolve().parents[1] / "data/raw/fireflux"
        if not all((root / name).exists() for name in audit.SOURCE_FILES):
            self.skipTest("optional published filtered FireFlux sources absent")
        report = audit.audit(root)
        self.assertEqual(report["time_axis"]["count"], 219)
        self.assertEqual(report["observation_statistics"]["ObsHWSInterp"]["shape"], [219, 4])


if __name__ == "__main__":
    unittest.main()
