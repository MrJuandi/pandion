from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from nightjar.discrimination_family import (
    DEVELOPMENT_INDICES,
    VALIDATION_INDICES,
    RESERVED_INDICES,
    build_scenarios,
    canonical_hash,
    write_family,
)
from nightjar.integrated import run_integrated


class DiscriminationFamilyTests(unittest.TestCase):
    def test_build_is_repeatable_valid_and_does_not_mutate_source(self):
        source = json.loads(Path("examples/synthetic_integrated.json").read_text())
        before = deepcopy(source)
        dev, val, reserved = build_scenarios(source)
        again_dev, again_val, again_res = build_scenarios(source)
        self.assertEqual(source, before)
        self.assertEqual(
            [[canonical_hash(s) for s in group] for group in (dev, val, reserved)],
            [[canonical_hash(s) for s in group] for group in (again_dev, again_val, again_res)],
        )
        self.assertEqual((len(dev), len(val), len(reserved)), (8, 8, 8))

        all_scenarios = dev + val + reserved
        # Check all seeds disjoint
        seeds = [s["seed"] for s in all_scenarios]
        self.assertEqual(len(seeds), len(set(seeds)))

        for scenario in all_scenarios:
            self.assertEqual(scenario["fire"]["horizon_s"], 1200)
            self.assertEqual(scenario["aircraft"]["horizon_s"], 1200)
            self.assertEqual(scenario["water_effect"]["radius_m"], 35)
            # Check 2 ignitions
            ignitions = scenario["fire"]["ignitions"]
            self.assertEqual(len(ignitions), 2)
            dist = abs(ignitions[0][0] - ignitions[1][0]) + abs(ignitions[0][1] - ignitions[1][1])
            self.assertGreaterEqual(dist, 4)
            # Observation schedule
            self.assertEqual(len(scenario["observation_schedule"]), 21)
            self.assertEqual(scenario["observation_schedule"][0]["available_at_s"], 0)

    def test_unmitigated_spread_is_non_trivial(self):
        dev, _, _ = build_scenarios()
        for scenario in dev:
            noint = run_integrated(scenario, "no_intervention")["affected_area_ha"]
            # All scenarios must burn at least 4.0 ha unmitigated (no trivial 0.5 ha floor cases)
            self.assertGreaterEqual(noint, 4.0)

    def test_development_has_distinct_decisions_between_policies(self):
        dev, _, _ = build_scenarios()
        heur_areas = [run_integrated(s, "heuristic")["affected_area_ha"] for s in dev]
        cont_areas = [run_integrated(s, "containment_heuristic")["affected_area_ha"] for s in dev]
        # Containment and heuristic must not produce identical outcomes across all cases
        self.assertNotEqual(heur_areas, cont_areas)

    def test_writer_only_evaluates_development_and_hashes_replays(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = write_family(root)
            report = json.loads((root / "comparison_report.json").read_text())
            # 8 dev scenarios * (1 noint + 1 heur + 1 cont + 10 rand) = 104 runs
            self.assertEqual(manifest["replay_count"], 104)
            self.assertEqual(len(report["runs"]), 104)
            unopened = {
                row["scenario_id"]
                for row in manifest["scenarios"]
                if row["split"] in ("validation", "reserved")
            }
            self.assertEqual(len(unopened), 16)
            self.assertFalse(unopened & {row["scenario_id"] for row in report["runs"]})
            self.assertFalse(any((root / "replays" / sid).exists() for sid in unopened))
            for row in report["runs"]:
                replay = json.loads((root / row["replay_file"]).read_text())
                scenario_spec = next(x for x in manifest["scenarios"] if x["scenario_id"] == row["scenario_id"])
                self.assertEqual(replay["scenario_sha256"], scenario_spec["sha256"])
                self.assertEqual(row["scenario_sha256"], scenario_spec["sha256"])


if __name__ == "__main__":
    unittest.main()
