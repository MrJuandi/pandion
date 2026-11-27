from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from pandion.discrimination_family import (
    build_reserved_v2_scenarios,
    build_scenarios,
    canonical_hash,
    generate_domain_randomized_scenario,
    write_family,
)
from pandion.integrated import run_integrated


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
        self.assertEqual((len(dev), len(val), len(reserved)), (8, 32, 8))

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
            self.assertEqual(len(unopened), 40)
            self.assertFalse(unopened & {row["scenario_id"] for row in report["runs"]})
            self.assertFalse(any((root / "replays" / sid).exists() for sid in unopened))
            for row in report["runs"]:
                replay = json.loads((root / row["replay_file"]).read_text())
                scenario_spec = next(x for x in manifest["scenarios"] if x["scenario_id"] == row["scenario_id"])
                self.assertEqual(replay["scenario_sha256"], scenario_spec["sha256"])
                self.assertEqual(row["scenario_sha256"], scenario_spec["sha256"])

    def test_generate_domain_randomized_scenario(self):
        s1 = generate_domain_randomized_scenario(seed=42)
        s2 = generate_domain_randomized_scenario(seed=42)
        s3 = generate_domain_randomized_scenario(seed=43)

        self.assertEqual(canonical_hash(s1), canonical_hash(s2))
        self.assertNotEqual(canonical_hash(s1), canonical_hash(s3))
        self.assertEqual(s1["scenario_id"], "discrimination-random-42")
        self.assertEqual(s1["fire"]["horizon_s"], 1200)
        self.assertEqual(s1["aircraft"]["horizon_s"], 1200)

        # Check multi-foci separation
        ign = s1["fire"]["ignitions"]
        self.assertEqual(len(ign), 2)
        dist = abs(ign[0][0] - ign[1][0]) + abs(ign[0][1] - ign[1][1])
        self.assertGreaterEqual(dist, 4)

        # Check immediate neighborhood of ignitions is burnable
        fuel = s1["fire"]["fuel_model"]
        for r, c in ign:
            self.assertNotEqual(fuel[r][c], "NONE")
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                if 0 <= r + dr < 11 and 0 <= c + dc < 11:
                    self.assertNotEqual(fuel[r + dr][c + dc], "NONE")

    def test_build_reserved_v2_scenarios(self):
        res_v2 = build_reserved_v2_scenarios()
        again = build_reserved_v2_scenarios()
        self.assertEqual(len(res_v2), 32)
        self.assertEqual(
            [canonical_hash(s) for s in res_v2],
            [canonical_hash(s) for s in again],
        )
        seeds_v2 = {s["seed"] for s in res_v2}
        self.assertEqual(len(seeds_v2), 32)
        self.assertTrue(all(s["scenario_id"].startswith("discrimination-reserved-v2-") for s in res_v2))
        self.assertTrue(all(s["generation"]["generator_version"] == "discrimination-family-0.2" for s in res_v2))

        # Check disjoint seeds with dev, val, res_v1
        dev, val, res_v1 = build_scenarios()
        existing_seeds = {s["seed"] for s in dev + val + res_v1}
        self.assertFalse(existing_seeds & seeds_v2)

        for scenario in res_v2:
            self.assertEqual(scenario["fire"]["horizon_s"], 1200)
            self.assertEqual(scenario["aircraft"]["horizon_s"], 1200)
            self.assertEqual(scenario["water_effect"]["radius_m"], 35)
            ignitions = scenario["fire"]["ignitions"]
            self.assertEqual(len(ignitions), 2)
            dist = abs(ignitions[0][0] - ignitions[1][0]) + abs(ignitions[0][1] - ignitions[1][1])
            self.assertGreaterEqual(dist, 4)


if __name__ == "__main__":
    unittest.main()
