"""Tests for the leave-one-incident-out fold report (D-71)."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from nightjar.fold_report import bootstrap_ci, format_table, load_reports, summarise


def report(activation, seed, policy, heuristic, containment=None, **config):
    rows = [{"scenario_id": f"{activation.lower()}-{i:02d}", "policy_ha": p, "heuristic_ha": h,
             "containment_heuristic_ha": (containment or heuristic)[i], "no_intervention_ha": 500.0,
             "random_valid_mean_ha": 300.0}
            for i, (p, h) in enumerate(zip(policy, heuristic))]
    base = {"schema_version": "nightjar-geographic-ppo-selection-0.1", "seed": seed,
            "holdout_activation": activation, "recurrent": True, "ppo": {"epochs": 4}, "input_mode": "image",
            "action_mode": "parametric", "include_ros": True, "include_thermal": True, "fleet_size": 2,
            "imagery_dir": "img", "candidates_count": 1, "dataset": {"scenarios_dir": "family"},
            "holdout": {"rows": rows}, "_path": f"{activation}-{seed}.json"}
    base.update(config)
    return base


class FoldReportTests(unittest.TestCase):
    def test_paired_difference_counts_and_relative_change(self):
        summary = summarise([report("A", 1, [90.0, 100.0, 120.0], [100.0, 100.0, 100.0])], resamples=200)
        fold = summary["folds"][0]["versus"]["heuristic"]
        self.assertAlmostEqual(fold["mean_difference_ha"], 10.0 / 3, places=3)
        self.assertEqual((fold["policy_better"], fold["equal"], fold["policy_worse"]), (1, 1, 1))
        self.assertAlmostEqual(fold["relative_difference"], (10.0 / 3) / 100.0, places=3)
        self.assertLessEqual(fold["ci95_ha"][0], fold["mean_difference_ha"])
        self.assertGreaterEqual(fold["ci95_ha"][1], fold["mean_difference_ha"])

    def test_seeds_are_averaged_per_scenario_and_their_means_kept(self):
        reports = [report("A", 1, [80.0, 100.0], [100.0, 100.0]), report("A", 2, [100.0, 120.0], [100.0, 100.0])]
        fold = summarise(reports, resamples=200)["folds"][0]
        self.assertEqual(fold["seeds"], [1, 2])
        self.assertEqual(fold["policy_mean_ha"], 100.0)
        self.assertEqual(fold["policy_seed_means_ha"], [90.0, 110.0])
        # Scenario 0 averages to 90 (better), scenario 1 to 110 (worse).
        self.assertEqual(fold["versus"]["heuristic"]["policy_better"], 1)

    def test_pooled_interval_resamples_within_folds(self):
        reports = [report("A", 1, [90.0] * 4, [100.0] * 4), report("B", 1, [130.0] * 2, [100.0] * 2)]
        pooled = summarise(reports, resamples=200)["pooled"]
        self.assertEqual(pooled["scenarios"], 6)
        # Constant differences inside each fold leave nothing to resample.
        self.assertEqual(pooled["versus"]["heuristic"]["ci95_ha"], [round(20.0 / 6, 4)] * 2)
        self.assertEqual(pooled["folds_policy_better_than_heuristic"], 1)

    def test_bootstrap_is_seeded(self):
        strata = [np.array([1.0, 5.0, 9.0, 2.0])]
        self.assertEqual(bootstrap_ci(strata, resamples=500, seed=3), bootstrap_ci(strata, resamples=500, seed=3))

    def test_reports_with_a_different_configuration_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "fleet_size"):
            summarise([report("A", 1, [1.0], [1.0]), report("B", 1, [1.0], [1.0], fleet_size=1)])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            summarise([report("A", 1, [1.0], [1.0]), report("A", 1, [1.0], [1.0])])

    def test_seeds_must_share_scenarios_and_baselines(self):
        with self.assertRaisesRegex(ValueError, "heuristic differs"):
            summarise([report("A", 1, [1.0], [1.0]), report("A", 2, [1.0], [2.0])])

    def test_content_versions_and_aoi_group_duplicates_are_rejected(self):
        first = report("EMSR602_AOI02", 1, [1.0], [1.0])
        second = report("EMSR602_AOI04", 1, [1.0], [1.0])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            summarise([first, second], resamples=100)

    def test_complete_provenance_accepts_relocation_and_fixed_selection(self):
        def entry(name):
            return {"scenario_id": f"{name.lower()}-00", "scenario_hash": name * 64,
                    "content_sha256": name * 64, "incident_group": name}
        a, b, v = entry("A"), entry("B"), entry("V")
        reports = []
        for group, train, held in (("A", b, a), ("B", a, b)):
            r = report(group, 42, [90.0], [100.0], slot_features=True, skip_forced=True,
                       updates=200, episodes_per_update=8,
                       training_settings={"learning_rate": 0.0003, "augment": True, "drop_bonus_per_cell": 0},
                       execution={"source_sha256": "a" * 64, "python": "3.12", "packages": {"torch": "2"}},
                       artifact={"checkpoint_sha256": "b" * 64})
            r["dataset"] = {"scenarios_dir": group, "imagery_content_sha256": "c" * 64,
                            "splits": {"development": [train], "validation": [v], "holdout": [held]}}
            r["imagery_dir"] = group
            r["holdout"]["rows"][0]["scenario_hash"] = held["scenario_hash"]
            reports.append(r)
        summary = summarise(reports, strict=True, resamples=100)
        self.assertEqual(summary["provenance_status"], "complete_metadata")
        self.assertEqual(summary["warnings"], [])
        reports[1]["dataset"]["splits"]["validation"] = [a]
        reports[1]["dataset"]["splits"]["development"] = [v]
        with self.assertRaisesRegex(ValueError, "selection_sha256"):
            summarise(reports, strict=True, resamples=100)
        first = report("A", 1, [1.0], [1.0], execution={"source_sha256": "a"})
        second = report("B", 1, [1.0], [1.0], execution={"source_sha256": "b"})
        with self.assertRaisesRegex(ValueError, "source_sha256"):
            summarise([first, second], resamples=100)

    def test_legacy_limits_strict_mode_and_independence_leakage(self):
        first = report("A", 1, [1.0], [1.0])
        summary = summarise([first], resamples=100)
        self.assertEqual(summary["provenance_status"], "legacy_incomplete")
        self.assertTrue(summary["warnings"])
        with self.assertRaisesRegex(ValueError, "strict comparison"):
            summarise([first], strict=True, resamples=100)
        first["dataset"]["splits"] = {"development": [{"incident_group": "A"}], "holdout": [{"incident_group": "A"}]}
        with self.assertRaisesRegex(ValueError, "not independent"):
            summarise([first], resamples=100)

    def test_same_id_changed_hash_and_invalid_areas_are_rejected(self):
        first, second = report("A", 1, [1.0], [1.0]), report("A", 2, [1.0], [1.0])
        first["holdout"]["rows"][0]["scenario_hash"] = "a"
        second["holdout"]["rows"][0]["scenario_hash"] = "b"
        with self.assertRaisesRegex(ValueError, "content differs"):
            summarise([first, second], resamples=100)
        first["holdout"]["rows"][0]["policy_ha"] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            summarise([first], resamples=100)

    def test_hold_aware_baseline_is_compared_when_every_report_has_it(self):
        reports = [report("A", 1, [90.0, 100.0], [100.0, 100.0]), report("B", 1, [90.0, 100.0], [100.0, 100.0])]
        for r in reports:
            for row in r["holdout"]["rows"]:
                row["hold_aware_heuristic_ha"] = 95.0
        summary = summarise(reports, resamples=100)
        self.assertIn("hold_aware_heuristic", summary["pooled"]["versus"])
        self.assertIn("hold_aware_heuristic", summary["folds"][0]["versus"])
        self.assertEqual(summary["pooled"]["baselines_mean_ha"]["hold_aware_heuristic"], 95.0)
        self.assertIn("| Hold-aware ha |", format_table(summary).splitlines()[0])

    def test_hold_aware_baseline_is_left_out_when_one_report_lacks_it(self):
        first, second = report("A", 1, [90.0, 100.0], [100.0, 100.0]), report("B", 1, [90.0, 100.0], [100.0, 100.0])
        for row in first["holdout"]["rows"]:
            row["hold_aware_heuristic_ha"] = 95.0
        summary = summarise([first, second], resamples=100)
        self.assertNotIn("hold_aware_heuristic", summary["pooled"]["versus"])
        self.assertNotIn("hold_aware_heuristic", summary["pooled"]["baselines_mean_ha"])
        self.assertNotIn("Hold-aware", format_table(summary))

    def test_load_skips_files_without_holdout_and_table_has_pooled_row(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            for name, payload in {"a.json": report("A", 1, [90.0], [100.0]), "b.json": {"schema_version": "x"}}.items():
                payload = {k: v for k, v in payload.items() if k != "_path"}
                (directory / name).write_text(json.dumps(payload))
            reports = load_reports([directory])
        self.assertEqual(len(reports), 1)
        table = format_table(summarise(reports, resamples=100))
        self.assertIn("| A | 1 | 90.0 | 100.0 |", table)
        self.assertIn("**Pooled (1 folds)**", table)


if __name__ == "__main__":
    unittest.main()
