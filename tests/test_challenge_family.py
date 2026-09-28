from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from nightjar.challenge_family import build_scenarios, canonical_hash, write_family
from nightjar.integrated import run_integrated


class ChallengeFamilyTests(unittest.TestCase):
    def test_build_is_repeatable_valid_and_does_not_mutate_source(self):
        source = json.loads(Path("examples/synthetic_integrated.json").read_text())
        before = deepcopy(source)
        dev, reserved = build_scenarios(source)
        again = build_scenarios(source)
        self.assertEqual(source, before)
        self.assertEqual([[canonical_hash(s) for s in group] for group in (dev, reserved)],
                         [[canonical_hash(s) for s in group] for group in again])
        self.assertEqual((len(dev), len(reserved)), (4, 4))
        for scenario in dev + reserved:
            self.assertEqual(scenario["fire"]["horizon_s"], 1200)
            self.assertEqual(scenario["aircraft"]["horizon_s"], 1200)
            self.assertEqual(scenario["water_effect"]["radius_m"], 35)
            self.assertEqual(len(scenario["observation_schedule"]), 21)
            self.assertEqual(scenario["observation_schedule"][0]["available_at_s"], 0)
            self.assertEqual(scenario["observation_schedule"][1]["available_at_s"] - 60,
                             scenario["observation_schedule"][2]["available_at_s"] - 120)

    def test_development_has_heuristic_wins_and_losses_against_random_mean(self):
        dev, _ = build_scenarios()
        differences = []
        for scenario in dev:
            heuristic = run_integrated(scenario, "heuristic")["affected_area_ha"]
            random_mean = sum(run_integrated(scenario, "random_valid", policy_seed=seed)["affected_area_ha"]
                              for seed in range(10)) / 10
            differences.append(heuristic - random_mean)
        self.assertTrue(any(value < 0 for value in differences), differences)
        self.assertTrue(any(value > 0 for value in differences), differences)

    def test_writer_only_evaluates_development_and_hashes_replays(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = write_family(root)
            report = json.loads((root / "comparison_report.json").read_text())
            self.assertEqual(manifest["replay_count"], 48)
            self.assertEqual(len(report["runs"]), 48)
            reserved = {row["scenario_id"] for row in manifest["scenarios"] if row["split"] == "reserved"}
            self.assertFalse(reserved & {row["scenario_id"] for row in report["runs"]})
            self.assertFalse(any((root / "replays" / sid).exists() for sid in reserved))
            for row in report["runs"]:
                replay = json.loads((root / row["replay_file"]).read_text())
                scenario_spec = next(x for x in manifest["scenarios"] if x["scenario_id"] == row["scenario_id"])
                self.assertEqual(replay["scenario_sha256"], scenario_spec["sha256"])
                self.assertEqual(row["scenario_sha256"], scenario_spec["sha256"])


if __name__ == "__main__":
    unittest.main()
