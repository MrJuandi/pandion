"""Observation-only direct attack inputs for the geographic frontier policy."""

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from pandion.action_features import BURNING_FEATURE_NAMES, TARGET_FEATURE_NAMES, target_feature_array
from pandion.geographic_family import build_geographic_family, canonical_hash
from pandion.geographic_integrated import geographic_episode, run_geographic_integrated
from pandion.geographic_ppo import HAS_TORCH, build_policy_model, main, with_water_model
from pandion.geographic_training_env import GeographicTrainingEnv
from pandion.parametric_actions import MAX_BURNING_FRONTIER, MAX_FRONTIER
from pandion.policy_artifact import load_policy_artifact
from pandion.policy_inputs import GeographicInputTensorizer, fuel_family
from tests.test_water_on_burning import DROP, burning_corridor, corridor


def image_inputs(scenario):
    grid = scenario["grid"]
    image = np.zeros((grid["rows"], grid["cols"], 3), dtype=np.uint8)
    return GeographicInputTensorizer(scenario, image, include_thermal=True, measured_intensity=True,
                                     focus_slots=True, action_features=True, frontier=True)


class DirectAttackInputs(unittest.TestCase):
    def test_flag_off_preserves_frontier_inputs(self):
        baseline = corridor()
        disabled = with_water_model([baseline], {"burning_cells": False})[0]
        baseline_obs, _, _ = next(geographic_episode(baseline, "heuristic"))
        disabled_obs, _, _ = next(geographic_episode(disabled, "heuristic"))
        before = image_inputs(baseline).extract_arrays(baseline_obs)
        after = image_inputs(disabled).extract_arrays(disabled_obs)
        for original, explicit in zip(before, after):
            np.testing.assert_array_equal(original, explicit)
        # RGB/thermal, cell channels, vector and target features from the existing corridor fixture.
        self.assertEqual([sha256(array.tobytes()).hexdigest() for array in after], [
            "a9c580b28996c213b9ca81737086ac48e49f5454fd479db88ecbcddde2ae4213",
            "b9082a6d78fb165e730535100ec23141057e04bfa67208f44cc85ee2e4593ff7",
            "00bc50c1e12f51bdc510eed7168507216b6abf4b0055207f418266bb02fa7d48",
            "987aa54c83fc55d94cce3df6248a0c1c1523628383097fd94c676eacdb631d45",
        ])
        baseline_mapper = GeographicTrainingEnv(baseline, action_mode="frontier").action_mapper()
        disabled_mapper = GeographicTrainingEnv(disabled, action_mode="frontier").action_mapper()
        mask = disabled_mapper.compute_mask(disabled_obs)
        self.assertEqual(mask, baseline_mapper.compute_mask(baseline_obs))
        self.assertEqual(mask, [True, False, True] + [False] * (MAX_FRONTIER - 1))

    def test_direct_target_features_describe_the_protected_neighbour(self):
        scenario = burning_corridor(4.0)
        scenario["water_model"]["max_stacked_drops"] = None
        scenario["grid"]["cols"] = 3
        scenario["fire"]["fuel_grid"] = [[102, 102, 141]]
        scenario["fire"]["burn_mask"] = [[False, True, False]]
        scenario["elevation_m"] = [[0.0, 20.0, 50.0]]
        obs, _, _ = next(geographic_episode(scenario, "heuristic"))
        obs["aircraft"]["position_m"] = [20.0, 0.0]
        obs["water_source_m"] = [0.0, 0.0]
        obs["known_burning_drops"] = [[0, 1]]
        obs["known_treated_cells"] = [[0, 2]]
        self.assertIsNone(obs["max_stacked_drops"])
        mapper = GeographicTrainingEnv(scenario, action_mode="frontier").action_mapper()
        self.assertEqual(mapper.num_actions, 2 + MAX_BURNING_FRONTIER)
        self.assertEqual(mapper.target_details(obs), {2: (0, 0, 0, -1, 0), 3: (0, 1, 0, 1, 0), 4: (0, 2, 0, 1, 0)})
        self.assertEqual(mapper.to_order(3, obs), {"type": "DROP", "target_m": [100.0, 0.0]})
        self.assertEqual(sum(mapper.compute_mask(obs)[2:]), 3)

        tensorizer = image_inputs(scenario)
        _, cells, _, actions = tensorizer.extract_arrays(obs)
        legacy = deepcopy(obs)
        del legacy["water_on_burning"]
        baseline = deepcopy(scenario)
        del baseline["water_model"]
        _, old_cells, _, old_actions = image_inputs(baseline).extract_arrays(legacy)
        self.assertEqual(cells.shape[0], old_cells.shape[0] + 1)
        self.assertEqual(actions.shape, (MAX_BURNING_FRONTIER, old_actions.shape[1] + 2))
        self.assertEqual(actions.shape[1] - 1, tensorizer.num_action_features)
        direct_channel = tensorizer.BASE_CELL_CHANNELS
        np.testing.assert_allclose(cells[direct_channel], [[0.0, 1 / 3, 0.0]])
        np.testing.assert_allclose(cells[direct_channel - 2], [[0.0, 0.0, 1 / 3]])
        np.testing.assert_allclose(actions[:3, -2:], [[0.0, 0.0], [1.0, 1 / 3], [0.0, 0.0]])
        obs["known_burning_drops"] *= 5
        _, cells, _, actions = tensorizer.extract_arrays(obs)
        self.assertEqual(cells[direct_channel, 0, 1], 1.0)
        self.assertEqual(actions[1, -1], 1.0)
        self.assertTrue(mapper.compute_mask(obs)[3])

        probs = np.array([[[.1, 1., .4]], [[.2, 1., .8]], [[.3, 1., .9]]])
        minutes = np.array([1., 10., 60.])
        with patch("pandion.action_features.risk_before_and_after",
                   return_value=(np.array([.8, .7, .7]), np.array([.5, .4, .4]))) as predict:
            features = target_feature_array(obs, mapper, fuel_family, "hold.npz",
                                            arrival=(probs, minutes, 10.0))
        row = features[1]
        self.assertEqual(row[0], 1)  # gathered trunk features still come from the drop location
        names = TARGET_FEATURE_NAMES + ("before", "after", "gain", "has_risk", "by_drop", "by_ten") + BURNING_FEATURE_NAMES
        target = dict(zip(names, row[1:]))
        expected = {"valid": 1, "stacked": 0, "reached_neighbours": 1 / 8,
                    "aircraft_distance": 180 / np.hypot(100, 300), "water_distance": 200 / 20000,
                    "downwind": 1, "offset": 0, "slope_to": .3, "wind_to": .4,
                    "fuel_3": 1, "before": .7, "after": .4, "gain": .3, "has_risk": 1,
                    "by_drop": .4 * 80 / 10 / 60, "by_ten": .8}
        for name, value in expected.items():
            with self.subTest(name=name):
                self.assertAlmostEqual(target[name], value)
        source_heat = obs["fire"]["cells"][0][1]["thermal_byte"] / 255
        self.assertAlmostEqual(target["heat_max"], source_heat)
        self.assertAlmostEqual(target["heat_mean"], source_heat)
        hold_inputs = predict.call_args.args[0]
        np.testing.assert_array_equal(hold_inputs[1], hold_inputs[2])
        self.assertAlmostEqual(hold_inputs[1, 7], 1 / 3)  # pre-wetting observed on protected fuel

    def test_knockdown_renders_as_burned(self):
        scenario = burning_corridor()
        replay = run_geographic_integrated(scenario, "no_intervention", _actions=[DROP])
        burning, burned = (record["observation"] for record in replay["decisions"][:2])
        cell = burned["fire"]["cells"][0][0]
        self.assertEqual(cell["fire_state"], "burning")
        self.assertEqual(cell["visible_state"], "burned")
        tensorizer = image_inputs(scenario)
        before, _, _, _ = tensorizer.extract_arrays(burning)
        after, _, _, _ = tensorizer.extract_arrays(burned)
        self.assertGreaterEqual(before[3, 0, 0], 76)
        self.assertLessEqual(after[3, 0, 0], 51)
        self.assertLess(after[0, 0, 0], before[0, 0, 0])

    def test_water_override_copies_rehashes_and_preserves_unset_keys(self):
        scenario = {"scenario_id": "wet", "scenario_hash": "original",
                    "water_model": {"candidates": "all"}, "provenance": {"source": "fixture"}}
        overrides = {"burning_cells": True, "max_stacked_drops": None}
        modified = with_water_model([scenario], overrides)[0]
        self.assertEqual(modified["water_model"], {"candidates": "all", **overrides})
        self.assertEqual(modified["provenance"]["water_model_override"],
                         {"water_model": overrides, "original_scenario_hash": "original"})
        self.assertEqual(modified.pop("scenario_hash"), canonical_hash(modified))
        self.assertEqual(scenario["water_model"], {"candidates": "all"})
        self.assertEqual(scenario["provenance"], {"source": "fixture"})
        scenarios = [scenario]
        self.assertIs(with_water_model(scenarios, {}), scenarios)

    def test_frontier_counts_decisions_that_exceed_the_ceiling(self):
        cols = MAX_BURNING_FRONTIER + 1
        scenario = burning_corridor()
        scenario["grid"]["cols"] = cols
        scenario["fire"].update(fuel_grid=[[102] * cols], burn_mask=[[c % 2 == 0 for c in range(cols)]],
                                horizon_s=10.0)
        env = GeographicTrainingEnv(scenario, action_mode="frontier")
        observation = env.reset()
        mask = env.action_mapper().compute_mask(observation)
        self.assertEqual(sum(mask[2:]), MAX_BURNING_FRONTIER)
        self.assertEqual(sum(order["type"] == "DROP" for order in env.action_mask), cols)
        _, _, done, _ = env.step_discrete(0)
        self.assertTrue(done)
        self.assertEqual(env.frontier_truncated_decisions, 1)


@unittest.skipUnless(HAS_TORCH, "torch not installed")
class DirectAttackTraining(unittest.TestCase):
    def test_smoke_checkpoint_and_evaluation_restore_the_water_model(self):
        family = build_geographic_family(num_dev=1, num_val=1, num_res=2, synthetic=True)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "manifest.json").write_text(json.dumps(family["manifest"]))
            for split in ("development", "validation"):
                (root / split).mkdir()
                for scenario in family["scenarios"][split]:
                    (root / split / f"{scenario['scenario_id']}.json").write_text(json.dumps(scenario))
            checkpoint = root / "weights.pt"
            report_path = root / "train.json"
            base_args = ["geographic_ppo", "--synthetic", "--scenarios-dir", str(root), "--checkpoint", str(checkpoint)]
            with (patch("sys.argv", base_args + ["--smoke", "--input-mode", "cells", "--action-mode", "frontier",
                         "--measured-intensity", "--action-features", "--water-on-burning", "--drop-cap", "none",
                         "--rollout-envs", "2", "--rollout-workers", "0", "--reward-reference", "heuristic",
                         "--output", str(report_path)]),
                  patch("pandion.geographic_training_env.run_geographic_integrated",
                        wraps=run_geographic_integrated) as reference):
                main()
            development = family["scenarios"]["development"]
            legacy = build_policy_model(development[0], input_mode="cells", action_mode="frontier",
                                        action_features=True, measured_intensity=True)
            model, info = load_policy_artifact(checkpoint, report_path, development)
            self.assertEqual(model.action_feature_dim, legacy.action_feature_dim + 2)
            self.assertEqual(model.stem.in_channels, legacy.stem.in_channels + 1)
            self.assertEqual(model.action_dim, 2 + MAX_BURNING_FRONTIER)
            overrides = {"burning_cells": True, "max_stacked_drops": None}
            reference_scenario = reference.call_args.args[0]
            self.assertEqual(reference_scenario, with_water_model(development, overrides)[0])
            self.assertEqual(info["inference"]["water_model"], overrides)
            self.assertEqual(info["inference"]["frontier_ceiling"], MAX_BURNING_FRONTIER)
            training = json.loads(report_path.read_text())
            self.assertEqual(training["training_frontier_truncated_decisions"], 0)
            self.assertEqual(training["evaluation_metrics"]["frontier_truncated_decisions"], 0)

            eval_path = root / "eval.json"
            with patch("sys.argv", base_args + ["--eval", "--split", "validation", "--training-report",
                       str(report_path), "--output", str(eval_path)]):
                main()
            evaluation = json.loads(eval_path.read_text())
            self.assertEqual(evaluation["water_model"], overrides)
            self.assertEqual(evaluation["artifact"]["inference"]["frontier_ceiling"], MAX_BURNING_FRONTIER)
            validation = with_water_model(family["scenarios"]["validation"], overrides)
            self.assertEqual(evaluation["dataset"]["splits"]["validation"][0]["scenario_hash"],
                             validation[0]["scenario_hash"])
