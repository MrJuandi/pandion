"""Batched collection preserves episode isolation, inputs and PPO targets."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from nightjar.geographic_family import build_synthetic_cems_fixture, generate_geographic_scenario
from nightjar.geographic_ppo import HAS_TORCH, build_policy_model, rollout_geographic, with_fleet
from nightjar.geographic_rollouts import RolloutBackend, rollout_geographic_batch
from tests.test_policy_inputs import PX, write_imagery

if HAS_TORCH:
    import torch


@unittest.skipUnless(HAS_TORCH, "PyTorch not installed")
class BatchedRolloutTests(unittest.TestCase):
    def setUp(self):
        fixture = build_synthetic_cems_fixture("EMSR578", rows=40, cols=40)
        self.scenarios = with_fleet([
            generate_geographic_scenario(fixture, index=i, split="development",
                                        window_rows=12, window_cols=12, horizon_s=horizon)
            for i, horizon in enumerate((240.0, 600.0, 360.0))], 2)
        torch.manual_seed(21)

    def assert_episodes_equal(self, expected, actual):
        self.assertEqual(len(expected), len(actual))
        for left, right in zip(expected, actual):
            self.assertEqual(len(left), len(right))
            for a, b in zip(left, right):
                self.assertEqual(a["reward"], b["reward"])
                for x, y in zip(a["inputs"], b["inputs"]):
                    torch.testing.assert_close(x, y, rtol=0, atol=0)
                torch.testing.assert_close(a["mask"], b["mask"], rtol=0, atol=0)
                torch.testing.assert_close(a["action"], b["action"], rtol=0, atol=0)
                for key in ("log_prob", "return", "advantage"):
                    torch.testing.assert_close(a[key], b[key], rtol=1e-4, atol=2e-5)
                self.assertAlmostEqual(a["value"], b["value"], places=5)

    def test_serial_and_batched_episode_inputs_targets_and_metrics(self):
        # Short/long episodes finish out of order; three scenarios over two
        # slots also requires a fresh recurrent state for a second group.
        for mode, action_mode in (("legacy", "tactical"), ("cells", "parametric")):
            for recurrent in (False, True):
                with self.subTest(mode=mode, recurrent=recurrent):
                    torch.manual_seed(21)
                    options = {"input_mode": mode}
                    model = build_policy_model(self.scenarios[0], recurrent=recurrent,
                                               action_mode=action_mode, **options)
                    before = deepcopy(model.state_dict())
                    serial = [rollout_geographic(model, sc, deterministic=True,
                                                gae_lambda=.95, normalize_advantages=False, **options)
                              for sc in self.scenarios]
                    records, metrics = rollout_geographic_batch(
                        model, self.scenarios, rollout_envs=2, deterministic=True,
                        gae_lambda=.95, normalize_advantages=False, **options)
                    self.assertEqual(metrics, [result[1] for result in serial])
                    self.assert_episodes_equal([result[0] for result in serial], records)
                    for name, value in model.state_dict().items():
                        self.assertTrue(torch.equal(before[name], value))

    def test_image_thermal_parametric_inputs_preserved(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            image = np.random.default_rng(5).integers(0, 256, (12 * PX, 12 * PX, 3), dtype=np.uint8)
            write_imagery(directory, self.scenarios[0], image)
            options = {"input_mode": "image", "imagery_dir": directory, "include_thermal": True}
            model = build_policy_model(self.scenarios[0], recurrent=True,
                                       action_mode="parametric", **options)
            serial = [rollout_geographic(model, sc, deterministic=True, **options)
                      for sc in self.scenarios[:2]]
            records, metrics = rollout_geographic_batch(model, self.scenarios[:2],
                                                        deterministic=True, **options)
            self.assertEqual(metrics, [result[1] for result in serial])
            self.assert_episodes_equal([result[0] for result in serial], records)

    def test_mixed_window_sizes_batch_in_separate_passes(self):
        # A window clipped by a small incident grid has other input shapes.
        fixture = build_synthetic_cems_fixture("EMSR578", rows=40, cols=40)
        small = with_fleet([generate_geographic_scenario(fixture, index=3, split="development", window_rows=10,
                                                         window_cols=10, horizon_s=360.0)], 2)[0]
        scenarios = [self.scenarios[0], small, self.scenarios[2]]
        options = {"input_mode": "cells"}
        for recurrent in (False, True):
            with self.subTest(recurrent=recurrent):
                torch.manual_seed(21)
                model = build_policy_model(scenarios[0], recurrent=recurrent, action_mode="parametric", **options)
                serial = [rollout_geographic(model, sc, deterministic=True, **options) for sc in scenarios]
                records, metrics = rollout_geographic_batch(model, scenarios, rollout_envs=3,
                                                            deterministic=True, **options)
                self.assertEqual(metrics, [result[1] for result in serial])
                self.assert_episodes_equal([result[0] for result in serial], records)

    def test_batched_drop_distances_keep_masks_at_model_action_dim(self):
        # Regression: batched collection once built masks without the model's drop distances.
        model = build_policy_model(self.scenarios[0], recurrent=True, action_mode="parametric",
                                   input_mode="cells", drop_distances=(0, 1))
        records, metrics = rollout_geographic_batch(model, self.scenarios, rollout_envs=2,
                                                    deterministic=True, input_mode="cells")
        self.assertTrue(all(m["rejected_orders"] == 0 for m in metrics))
        masks = [record["mask"] for episode in records for record in episode]
        self.assertTrue(masks)
        self.assertTrue(all(len(mask) == model.action_dim for mask in masks))

    def test_skip_forced_matches_serial_with_nondefault_bonus(self):
        for mode, action_mode in (("legacy", "tactical"), ("cells", "parametric")):
            with self.subTest(mode=mode):
                torch.manual_seed(31)
                model = build_policy_model(self.scenarios[0], recurrent=True,
                                           action_mode=action_mode, input_mode=mode)
                options = {"input_mode": mode, "skip_forced": True,
                           "drop_bonus_per_cell": 0.61, "deterministic": True,
                           "gae_lambda": .93, "normalize_advantages": False}
                serial = [rollout_geographic(model, sc, **options) for sc in self.scenarios]
                records, metrics = rollout_geographic_batch(model, self.scenarios,
                                                            rollout_envs=2, **options)
                self.assertEqual(metrics, [result[1] for result in serial])
                self.assert_episodes_equal([result[0] for result in serial], records)

    def test_augmented_image_and_slot_feature_inputs_match(self):
        from nightjar.augmentation import augment_scenario

        augmented = [augment_scenario(sc, 4) for sc in self.scenarios[:2]]
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            image = np.random.default_rng(12).integers(0, 256, (12 * PX, 12 * PX, 3), dtype=np.uint8)
            write_imagery(directory, augmented[0], image)
            image_options = {"input_mode": "image", "imagery_dir": directory,
                             "include_thermal": True}
            image_model = build_policy_model(augmented[0], recurrent=True,
                                             action_mode="parametric", **image_options)
            serial = [rollout_geographic(image_model, sc, deterministic=True,
                                          skip_forced=True, **image_options)
                      for sc in augmented]
            records, metrics = rollout_geographic_batch(image_model, augmented,
                                                        rollout_envs=2, deterministic=True,
                                                        skip_forced=True,
                                                        **image_options)
            self.assertEqual(metrics, [result[1] for result in serial])
            self.assert_episodes_equal([result[0] for result in serial], records)

        slot_scenario = augment_scenario(self.scenarios[0], 4)
        options = {"input_mode": "cells"}
        model = build_policy_model(slot_scenario, recurrent=True, action_mode="parametric",
                                   slot_features=True, **options)
        serial, serial_metrics = rollout_geographic(model, slot_scenario,
                                                    deterministic=True, skip_forced=True, **options)
        records, metrics = rollout_geographic_batch(model, [slot_scenario],
                                                    deterministic=True, skip_forced=True, **options)
        self.assertEqual(metrics, [serial_metrics])
        self.assert_episodes_equal([serial], records)

    def test_spawned_workers_match_local_and_are_reused_and_closed(self):
        model = build_policy_model(self.scenarios[0], input_mode="cells", action_mode="parametric")
        expected, expected_metrics = rollout_geographic_batch(
            model, self.scenarios, input_mode="cells", rollout_envs=2,
            deterministic=True, skip_forced=True)
        with RolloutBackend(action_mode="parametric", input_options={"input_mode": "cells"},
                            workers=2, skip_forced=True) as backend:
            pids = [p.pid for p in backend.processes]
            for _ in range(2):
                actual, metrics = rollout_geographic_batch(
                    model, self.scenarios, input_mode="cells", rollout_envs=2,
                    deterministic=True, skip_forced=True, backend=backend)
                self.assertEqual(metrics, expected_metrics)
                self.assert_episodes_equal(expected, actual)
                self.assertEqual([p.pid for p in backend.processes], pids)
        self.assertTrue(all(not p.is_alive() for p in backend.processes))
        with self.assertRaisesRegex(RuntimeError, "closed"):
            backend.reset([])

    def test_all_forced_episode_has_no_records_or_policy_calls(self):
        from nightjar.model import ActionMapper

        original_mask = ActionMapper.compute_mask
        def one_valid_action(mapper, observation):
            mask = original_mask(mapper, observation)
            first = next(i for i, valid in enumerate(mask) if valid)
            return [i == first for i in range(len(mask))]

        class NoCallModel:
            action_mode = "tactical"
            recurrent = True
            calls = 0
            def get_action(self, *args, **kwargs):
                self.calls += 1
                raise AssertionError("all-forced episodes must not call the model")

        model = NoCallModel()
        with patch.object(ActionMapper, "compute_mask", one_valid_action):
            _, expected_metrics = rollout_geographic(
                model, self.scenarios[0], skip_forced=True, normalize_advantages=False)
            records, metrics = rollout_geographic_batch(
                model, [self.scenarios[0]], skip_forced=True,
                rollout_envs=1, normalize_advantages=False)
        self.assertEqual(records, [[]])
        self.assertEqual(metrics, [expected_metrics])
        self.assertEqual(metrics[0]["steps_count"], 0)
        self.assertEqual(model.calls, 0)

    def test_worker_errors_propagate_and_cleanup(self):
        model = build_policy_model(self.scenarios[0], input_mode="cells", action_mode="parametric")
        damaged = deepcopy(self.scenarios[0])
        damaged["fire"]["elevation_m"] = [[float("nan")] * 12 for _ in range(12)]
        with RolloutBackend(action_mode="parametric", input_options={"input_mode": "cells"}, workers=1) as backend:
            with self.assertRaisesRegex(RuntimeError, "finite"):
                rollout_geographic_batch(model, [damaged], backend=backend, input_mode="cells")
        self.assertFalse(backend.processes[0].is_alive())

    def test_stochastic_batches_repeat_and_actions_are_valid(self):
        model = build_policy_model(self.scenarios[0], input_mode="cells", action_mode="parametric")
        runs = []
        for _ in range(2):
            torch.manual_seed(57)
            records, metrics = rollout_geographic_batch(model, self.scenarios, input_mode="cells", rollout_envs=2)
            self.assertTrue(all(m["rejected_orders"] == 0 for m in metrics))
            for ep in records:
                for record in ep:
                    self.assertTrue(record["mask"][record["action"]])
                    self.assertTrue(torch.isfinite(record["return"]))
            runs.append((records, metrics))
        self.assertEqual(runs[0][1], runs[1][1])
        self.assert_episodes_equal(runs[0][0], runs[1][0])

    def test_training_batches_keep_episode_count_and_pack_once_per_update(self):
        from nightjar import geographic_ppo as gp
        collected = []
        real = rollout_geographic_batch
        def collect(*args, **kwargs):
            result = real(*args, **kwargs)
            collected.append(len(result[0]))
            return result
        with patch("nightjar.geographic_rollouts.rollout_geographic_batch", side_effect=collect), \
             patch("nightjar.geographic_ppo._pack_episode", wraps=gp._pack_episode) as pack:
            model, losses = gp.train_geographic_ppo(
                self.scenarios, updates=2, episodes_per_update=3, rollout_envs=2,
                input_mode="cells", action_mode="parametric", ppo_epochs=3,
                gae_lambda=.95, reward_scaling=True)
        self.assertEqual(collected, [3, 3])
        self.assertEqual(pack.call_count, 6)
        self.assertEqual(len(losses), 2)
        self.assertTrue(all(torch.isfinite(torch.tensor(x)) for x in losses))
        self.assertTrue(all(torch.isfinite(t).all() for t in model.state_dict().values()))

    def test_invalid_collection_options_fail_before_training(self):
        from nightjar.geographic_ppo import train_geographic_ppo
        for options in ({"rollout_envs": 0}, {"rollout_workers": -1}):
            with self.assertRaises(ValueError):
                train_geographic_ppo(self.scenarios, **options)

    def test_cli_records_and_passes_collection_settings(self):
        import json
        from nightjar import geographic_ppo as gp
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "report.json"
            with patch("sys.argv", ["geographic_ppo", "--train", "--rollout-envs", "8",
                                    "--rollout-workers", "2", "--episodes-per-update", "8",
                                    "--output", str(output)]), \
                 patch("nightjar.geographic_ppo.load_geographic_split", return_value=self.scenarios), \
                 patch("nightjar.geographic_ppo.train_geographic_ppo", return_value=(object(), [0.1])) as train, \
                 patch("nightjar.geographic_ppo.rollout_geographic", return_value=([], {})):
                gp.main()
            self.assertEqual(train.call_args.kwargs["rollout_envs"], 8)
            self.assertEqual(train.call_args.kwargs["rollout_workers"], 2)
            self.assertEqual(train.call_args.kwargs["episodes_per_update"], 8)
            self.assertEqual(json.loads(output.read_text())["collection"],
                             {"rollout_envs": 8, "rollout_workers": 2})


if __name__ == "__main__":
    unittest.main()
