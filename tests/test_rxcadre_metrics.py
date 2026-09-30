"""Synthetic contract tests; these fixtures are not RxCADRE observations."""
import copy
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from nightjar.rxcadre_metrics import score


def fixture():
    metadata = {"crs": "LOCAL_METERS", "time_origin_utc": "2012-11-01T00:00:00Z", "seed": 0,
                "units": {"rate": "m/s", "arrival": "s", "position": "m"}}
    records = [{"burn_id": f"synthetic-{i}", "sample_id": "a", "split": split, "source": "synthetic fixture",
                "window_start_s": 0, "window_end_s": 200, "direction_deg": 90,
                "rate_definition": "matched segment", "rate_m_s": 0.5, "rate_interval_m_s": [0.4, 0.6],
                "arrival_s": 100, "front_position_m": 50}
               for i, split in enumerate(("development", "evaluation", "evaluation"))]
    obs = {**metadata, "schema_version": "rxcadre-observations-0.1", "records": records}
    pred = {**metadata, "schema_version": "rxcadre-predictions-0.1", "records": copy.deepcopy(records),
            "frozen_configuration_sha256": "a" * 64,
            "observations_sha256": sha256(json.dumps(obs, sort_keys=True, separators=(",", ":")).encode()).hexdigest()}
    return obs, pred


class RxCADREMetricsTests(unittest.TestCase):
    def test_bad_burn_is_not_hidden_by_other_burns(self):
        obs, pred = fixture()
        pred["records"][2]["rate_m_s"] = 1.0
        result = score(obs, pred)
        self.assertTrue(result["minimum_split_coverage"])
        self.assertFalse(result["all_evaluation_rate_criteria_pass"])
        self.assertEqual(result["by_burn"]["synthetic-2"]["normalized_rate_mae"], 1)
        self.assertAlmostEqual(result["by_burn"]["synthetic-2"]["rate_interval_distance_mean_m_s"], 0.4)

    def test_dropping_or_duplicating_a_sample_rejected(self):
        for change in (lambda r: r.pop(), lambda r: r.append(r[0])):
            obs, pred = fixture()
            change(pred["records"])
            with self.assertRaises(ValueError):
                score(obs, pred)

    def test_mismatched_physical_window_rejected(self):
        obs, pred = fixture()
        pred["records"][0]["window_end_s"] = 3600
        with self.assertRaisesRegex(ValueError, "window_end_s"):
            score(obs, pred)

    def test_nonfinite_prediction_and_changed_contract_rejected(self):
        obs, pred = fixture()
        pred["records"][0]["rate_m_s"] = float("nan")
        with self.assertRaises(ValueError):
            score(obs, pred)
        obs, pred = fixture()
        obs["records"][0]["rate_m_s"] = 0.4
        with self.assertRaisesRegex(ValueError, "exact observation"):
            score(obs, pred)

    def test_cli_records_artifact_hashes_without_claiming_validation(self):
        obs, pred = fixture()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name, value in (("obs", obs), ("pred", pred)):
                (root / f"{name}.json").write_text(json.dumps(value))
            run = subprocess.run([sys.executable, "-m", "nightjar.rxcadre_metrics", "--observations", str(root / "obs.json"),
                                  "--predictions", str(root / "pred.json"), "--output", str(root / "report.json")],
                                 capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            result = json.loads((root / "report.json").read_text())
            self.assertEqual(result["input_file_sha256"]["observations"], sha256((root / "obs.json").read_bytes()).hexdigest())
            self.assertEqual(result["physical_validation_status"], "not established by metrics alone")


if __name__ == "__main__":
    unittest.main()
