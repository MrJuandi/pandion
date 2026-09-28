from copy import deepcopy
import json
import unittest

from nightjar.scenario_family import TIME_ORIGIN_UTC, VARIANT_SPECS, build_scenarios, canonical_hash
from nightjar.integrated import run_integrated


class ScenarioFamilyTests(unittest.TestCase):
    def test_family_is_fixed_valid_balanced_and_metadata_agrees(self):
        scenarios = build_scenarios()
        self.assertEqual(len(scenarios), 8)
        self.assertEqual(len({s["scenario_id"] for s in scenarios}), 8)
        for scenario in scenarios:
            for key in ("scenario_id", "seed", "provenance", "crs"):
                self.assertEqual(scenario[key], scenario["fire"][key])
                self.assertEqual(scenario[key], scenario["aircraft"][key])
            for document in (scenario, scenario["fire"], scenario["aircraft"]):
                self.assertEqual(document["time_origin_utc"], TIME_ORIGIN_UTC)
                self.assertEqual(document["units"]["time"], "s")
            self.assertEqual(scenario["fire"]["rows"], 11)
            self.assertEqual(scenario["fire"]["cols"], 11)
        for split in ("development", "reserved"):
            specs = [item for item in VARIANT_SPECS if item["split"] == split]
            self.assertEqual(len({item["water_radius_m"] for item in specs}), 2)
            self.assertGreaterEqual(len({tuple(item["wind"]) for item in specs}), 2)
        for left, right in zip(scenarios[::2], scenarios[1::2]):
            self.assertEqual(left["seed"], right["seed"])
            for component in ("fire", "aircraft"):
                a, b = deepcopy(left[component]), deepcopy(right[component])
                a.pop("scenario_id")
                b.pop("scenario_id")
                self.assertEqual(a, b)
            self.assertEqual({left["water_effect"]["radius_m"], right["water_effect"]["radius_m"]}, {35, 55})

    def test_build_is_deterministic_and_does_not_mutate_source(self):
        with open("examples/synthetic_integrated.json") as source_file:
            source = json.load(source_file)
        before = deepcopy(source)
        first = build_scenarios(source)
        second = build_scenarios(source)
        self.assertEqual(source, before)
        self.assertEqual([canonical_hash(s) for s in first], [canonical_hash(s) for s in second])
        self.assertEqual(source["fire"]["ignitions"], [[5, 5]])

    def test_existing_policies_replay_from_observation_path_without_rejection(self):
        scenario = build_scenarios()[0]
        for policy in ("no_intervention", "heuristic", "random_valid"):
            replay = run_integrated(scenario, policy, policy_seed=0)
            self.assertEqual(replay["scenario_id"], scenario["scenario_id"])
            self.assertEqual(replay["aircraft_metrics"]["rejected_orders"], 0)
            self.assertTrue(all("observation" in decision and "order" in decision for decision in replay["decisions"]))


if __name__ == "__main__":
    unittest.main()
