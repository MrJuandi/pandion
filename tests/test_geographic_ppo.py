"""Unit tests for geographic PPO training and evaluation."""

import json
import math
import random
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from nightjar.geographic_family import (
    build_synthetic_cems_fixture,
    generate_geographic_scenario,
)
from nightjar.geographic_ppo import (
    HAS_TORCH,
    load_geographic_split,
    ppo_update_geographic,
    RewardScaler,
    returns_and_advantages,
    rollout_geographic,
    train_geographic_ppo,
)
from nightjar.model import SpatialActorCritic, TACTICAL_ACTION_NAMES

if HAS_TORCH:
    import torch


class TestGeographicLoader(unittest.TestCase):
    def _write_development_family(self, root, num_dev=1):
        from nightjar.geographic_family import build_geographic_family
        family = build_geographic_family(num_dev=num_dev, num_val=1, num_res=2, synthetic=True)
        (root / "manifest.json").write_text(json.dumps(family["manifest"]))
        (root / "development").mkdir()
        for scenario in family["scenarios"]["development"]:
            (root / "development" / f"{scenario['scenario_id']}.json").write_text(json.dumps(scenario))
        return family

    def test_duplicate_scenario_files_do_not_reweight_training(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            family = self._write_development_family(root)
            scenario = family["scenarios"]["development"][0]
            (root / "development" / "copy.json").write_text(json.dumps(scenario))
            with self.assertRaisesRegex(ValueError, "Duplicate scenario ID in split directory"):
                load_geographic_split("development", root, synthetic=True)

    def test_complete_membership_is_required(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            family = self._write_development_family(root, num_dev=2)
            scenario = family["scenarios"]["development"][1]
            path = root / "development" / f"{scenario['scenario_id']}.json"
            path.unlink()
            with self.assertRaisesRegex(ValueError, "membership does not match manifest"):
                load_geographic_split("development", root, synthetic=True)
            scenario["scenario_id"] = "unlisted-scenario"
            path.write_text(json.dumps(scenario))
            with self.assertRaisesRegex(ValueError, "absent from manifest"):
                load_geographic_split("development", root, synthetic=True)

    def test_source_product_metadata_must_match_registered_stage(self):
        from copy import deepcopy
        from nightjar.cems_catalog import ACTIVATIONS
        from nightjar.geographic_family import canonical_hash
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            family = self._write_development_family(root)
            manifest = family["manifest"]
            scenario = family["scenarios"]["development"][0]
            path = root / "development" / f"{scenario['scenario_id']}.json"
            product = ACTIVATIONS["EMSR578"]["products"]["del"]
            manifest["source_kind"] = scenario["source_kind"] = "copernicus_cems"
            scenario["provenance"].update({
                "source_kind": "copernicus_cems",
                "source_stage": "del", "source_zip_sha256": product["sha256"],
                "source_product_code": product["product_code"],
                "source_product_filename": product["vector_filename"],
                "image_acquired_at_utc": product["acquired_at_utc"],
            })
            entry = manifest["scenarios"]["development"][0]
            entry.update(source_kind="copernicus_cems", source_zip_sha256=product["sha256"])

            def persist(value):
                value["scenario_hash"] = canonical_hash({k: v for k, v in value.items() if k != "scenario_hash"})
                entry["hash"] = value["scenario_hash"]
                path.write_text(json.dumps(value))
                (root / "manifest.json").write_text(json.dumps(manifest))

            persist(scenario)
            self.assertEqual(len(load_geographic_split("development", root)), 1)
            for field, value in (("source_stage", "monit01"), ("source_stage", None),
                                 ("source_product_code", "MONIT01"),
                                 ("source_product_filename", "wrong.zip"),
                                 ("image_acquired_at_utc", "2022-06-18T10:47:00Z")):
                with self.subTest(field=field, value=value):
                    damaged = deepcopy(scenario)
                    damaged["provenance"][field] = value
                    persist(damaged)
                    with self.assertRaisesRegex(ValueError, "unregistered source provenance"):
                        load_geographic_split("development", root)

    def test_smoke_report_identifies_only_its_training_pool(self):
        from nightjar.geographic_ppo import main
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            family = self._write_development_family(root, num_dev=3)
            output = root / "smoke.json"
            with patch("sys.argv", ["geographic_ppo", "--smoke", "--synthetic",
                                    "--scenarios-dir", str(root), "--output", str(output)]), \
                 patch("nightjar.geographic_ppo._require_torch"), \
                 patch("nightjar.geographic_ppo.torch"), \
                 patch("nightjar.geographic_ppo.train_geographic_ppo", return_value=(object(), [0.0])) as train, \
                 patch("nightjar.geographic_ppo.rollout_geographic", return_value=([], {})):
                main()
            expected = [s["scenario_id"] for s in family["scenarios"]["development"][:2]]
            self.assertEqual([s["scenario_id"] for s in train.call_args.args[0]], expected)
            smoke_report = json.loads(output.read_text())
            reported = smoke_report["dataset"]["splits"]["development"]
            self.assertEqual([s["scenario_id"] for s in reported], expected)
            self.assertEqual(smoke_report["seed"], 42)

    def test_selection_candidate_seeds_derive_from_cli_seed(self):
        if not HAS_TORCH:
            self.skipTest("PyTorch not installed")
        from nightjar.geographic_ppo import main
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            family = self._write_development_family(root)
            loaded = {
                "development": family["scenarios"]["development"],
                "validation": family["scenarios"]["validation"],
            }
            output = root / "selection.json"
            observed_torch_seeds = []

            def fake_train(*args, **kwargs):
                observed_torch_seeds.append(torch.initial_seed())
                return object(), [0.0]

            with patch("sys.argv", [
                "geographic_ppo", "--select", "--synthetic", "--scenarios-dir", str(root),
                "--output", str(output), "--candidates", "2", "--seed", "123",
            ]), patch("nightjar.geographic_ppo.load_geographic_split", side_effect=lambda split, *_args, **_kwargs: loaded[split]), \
                 patch("nightjar.geographic_ppo.train_geographic_ppo", side_effect=fake_train), \
                 patch("nightjar.geographic_ppo.evaluate_geographic_policy", return_value=[{"final_burned_ha": 1.0, "drops_count": 0}]):
                main()

            report = json.loads(output.read_text())
            self.assertEqual(report["seed"], 123)
            self.assertEqual([candidate["seed"] for candidate in report["candidates"]], [123, 124])
            self.assertEqual(observed_torch_seeds, [123, 124])

    def test_dataset_errors_are_visible(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with self.assertRaisesRegex(FileNotFoundError, str(root / "development")):
                load_geographic_split("development", root)
            (root / "development").mkdir()
            with self.assertRaisesRegex(ValueError, "no scenario JSON"):
                load_geographic_split("development", root)
            (root / "development" / "bad.json").write_text("{")
            (root / "manifest.json").write_text(json.dumps({"scenarios": {"development": []}}))
            with self.assertRaisesRegex(ValueError, "Invalid scenario JSON"):
                load_geographic_split("development", root)

    def test_explicit_synthetic_family_load_and_integrity(self):
        from nightjar.geographic_family import build_geographic_family
        family = build_geographic_family(num_dev=1, num_val=1, num_res=2, synthetic=True)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "manifest.json").write_text(json.dumps(family["manifest"]))
            for split, scenarios in family["scenarios"].items():
                (root / split).mkdir()
                for scenario in scenarios:
                    (root / split / f"{scenario['scenario_id']}.json").write_text(json.dumps(scenario))
            with self.assertRaisesRegex(ValueError, "Unexpected or unregistered source provenance"):
                load_geographic_split("development", root)
            loaded = load_geographic_split("development", root, synthetic=True)
            self.assertEqual(loaded[0]["source_kind"], "synthetic_fixture")
            path = next((root / "development").glob("*.json"))
            original = json.loads(path.read_text())
            damaged = dict(original)
            damaged["scenario_hash"] = "0" * 64
            path.write_text(json.dumps(damaged))
            with self.assertRaisesRegex(ValueError, "embedded or manifest hash mismatch"):
                load_geographic_split("development", root, synthetic=True)

            damaged = dict(original)
            damaged["split"] = "validation"
            path.write_text(json.dumps(damaged))
            with self.assertRaisesRegex(ValueError, "split/activation"):
                load_geographic_split("development", root, synthetic=True)
            damaged["split"] = "development"
            damaged["activation_id"] = "EMSR590"
            path.write_text(json.dumps(damaged))
            with self.assertRaisesRegex(ValueError, "split/activation"):
                load_geographic_split("development", root, synthetic=True)

            damaged = json.loads(json.dumps(original))
            damaged["source_kind"] = "copernicus_cems"
            damaged["provenance"]["source_zip_sha256"] = "f" * 64
            from nightjar.geographic_family import canonical_hash
            damaged["scenario_hash"] = canonical_hash({k: v for k, v in damaged.items() if k != "scenario_hash"})
            manifest = json.loads((root / "manifest.json").read_text())
            entry = manifest["scenarios"]["development"][0]
            entry["source_kind"] = "copernicus_cems"
            entry["source_zip_sha256"] = "f" * 64
            entry["hash"] = damaged["scenario_hash"]
            (root / "manifest.json").write_text(json.dumps(manifest))
            path.write_text(json.dumps(damaged))
            with self.assertRaisesRegex(ValueError, "unregistered source provenance"):
                load_geographic_split("development", root, synthetic=False)


class TestGeographicPPO(unittest.TestCase):
    def setUp(self):
        if not HAS_TORCH:
            self.skipTest("PyTorch not installed")
        random.seed(42)
        torch.manual_seed(42)
        fixture = build_synthetic_cems_fixture("EMSR578", utm_zone=31, rows=25, cols=25)
        self.scenario = generate_geographic_scenario(
            fixture,
            index=0,
            split="development",
            window_rows=15,
            window_cols=15,
            horizon_s=300.0,
        )

    def test_load_geographic_split(self):
        from nightjar.geographic_family import build_geographic_family
        family = build_geographic_family(num_dev=1, num_val=1, num_res=2, synthetic=True)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "manifest.json").write_text(json.dumps(family["manifest"]))
            for split, scenarios in family["scenarios"].items():
                (root / split).mkdir()
                for scenario in scenarios:
                    (root / split / f"{scenario['scenario_id']}.json").write_text(json.dumps(scenario))
            scenarios = load_geographic_split("development", root, synthetic=True)
            self.assertEqual(len(scenarios), 1)
            self.assertIn("grid", scenarios[0])
            self.assertIn("fire", scenarios[0])

    def test_returns_and_advantages(self):
        rewards = [-1.0, -0.5, 2.0]
        values = [-0.8, -0.4, 1.5]
        ret, adv = returns_and_advantages(rewards, values, gamma=0.9)
        self.assertEqual(len(ret), 3)
        self.assertEqual(len(adv), 3)
        self.assertTrue(torch.isfinite(ret).all())
        self.assertTrue(torch.isfinite(adv).all())

    def test_rollout_and_ppo_update(self):
        model = SpatialActorCritic(
            grid_rows=15,
            grid_cols=15,
            action_dim=len(TACTICAL_ACTION_NAMES),
            recurrent=True,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        records, metrics = rollout_geographic(model, self.scenario, deterministic=False)
        self.assertGreater(len(records), 0)
        self.assertEqual(metrics["rejected_orders"], 0)
        self.assertIn("final_burned_ha", metrics)

        loss = ppo_update_geographic(model, optimizer, records)
        self.assertTrue(torch.isfinite(torch.tensor(loss)))

    def test_ppo_update_accumulation_equals_batched_loss(self):
        import copy
        for recurrent in (True, False):
            torch.manual_seed(0)
            model = SpatialActorCritic(grid_rows=15, grid_cols=15, action_dim=len(TACTICAL_ACTION_NAMES),
                                       recurrent=recurrent)
            episodes = [rollout_geographic(model, self.scenario)[0] for _ in range(2)]
            accumulated = copy.deepcopy(model)
            ppo_update_geographic(accumulated, torch.optim.SGD(accumulated.parameters(), lr=0.0), episodes)
            batched = copy.deepcopy(model)
            parts = []
            for ep in episodes:
                inputs = tuple(torch.stack(x) for x in zip(*(r["inputs"] for r in ep)))
                lp, ent, val = batched.evaluate_actions(*inputs, torch.stack([r["mask"] for r in ep]),
                                                        torch.stack([r["action"] for r in ep]))
                parts.append((lp, ent, val, ep))
            lp = torch.cat([p[0] for p in parts]); ent = torch.cat([p[1] for p in parts])
            val = torch.cat([p[2] for p in parts]); flat = [r for p in parts for r in p[3]]
            old = torch.stack([r["log_prob"] for r in flat]); ret = torch.stack([r["return"] for r in flat])
            adv = torch.stack([r["advantage"] for r in flat])
            ratio = (lp - old).exp()
            loss = (-torch.minimum(ratio * adv, ratio.clamp(0.8, 1.2) * adv).mean()
                    + 0.5 * ((val - ret) ** 2).mean() - 0.01 * ent.mean())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(batched.parameters(), max_norm=0.5)
            for a, b in zip(accumulated.parameters(), batched.parameters()):
                torch.testing.assert_close(a.grad, b.grad, atol=1e-6, rtol=1e-5)

    def test_gae_limits_match_monte_carlo_and_td(self):
        rewards, values, gamma = [1.0, -2.0, 0.5], [0.3, -0.1, 0.2], 0.9
        mc_ret, mc_adv = returns_and_advantages(rewards, values, gamma, normalize=False)
        ret1, adv1 = returns_and_advantages(rewards, values, gamma, gae_lambda=1.0, normalize=False)
        torch.testing.assert_close(adv1, mc_adv)
        torch.testing.assert_close(ret1, mc_ret)
        _, adv0 = returns_and_advantages(rewards, values, gamma, gae_lambda=0.0, normalize=False)
        expected = [rewards[0] + gamma * values[1] - values[0], rewards[1] + gamma * values[2] - values[1],
                    rewards[2] - values[2]]
        torch.testing.assert_close(adv0, torch.tensor(expected))

    def test_reward_scaler_tracks_discounted_return_std(self):
        scaler = RewardScaler(gamma=0.0)
        self.assertEqual(scaler.scale, 1.0)
        scaler.update([2.0, -2.0, 2.0, -2.0])
        self.assertAlmostEqual(scaler.scale, 0.5, places=6)

    def test_multi_epoch_minibatch_update_with_gae_and_scaling(self):
        torch.manual_seed(0)
        random.seed(0)
        model, losses = train_geographic_ppo([self.scenario], updates=2, episodes_per_update=2,
                                             recurrent=True, ppo_epochs=3, minibatches=2, gae_lambda=0.95,
                                             reward_scaling=True, target_kl=0.05)
        self.assertEqual(len(losses), 2)
        self.assertTrue(all(torch.isfinite(torch.tensor(l)) for l in losses))

    def test_more_epochs_move_weights_further(self):
        import copy
        torch.manual_seed(0)
        model = SpatialActorCritic(grid_rows=15, grid_cols=15, action_dim=len(TACTICAL_ACTION_NAMES), recurrent=True)
        episodes = [rollout_geographic(model, self.scenario)[0] for _ in range(2)]
        moved = []
        for epochs in (1, 4):
            m = copy.deepcopy(model)
            ppo_update_geographic(m, torch.optim.Adam(m.parameters(), lr=1e-3), episodes, epochs=epochs)
            moved.append(sum(float((a - b).abs().sum()) for a, b in zip(m.parameters(), model.parameters())))
        self.assertGreater(moved[1], moved[0])

    def test_train_geographic_ppo_single_update(self):
        model, losses = train_geographic_ppo(
            [self.scenario],
            updates=1,
            episodes_per_update=1,
            recurrent=True,
        )
        self.assertEqual(len(losses), 1)
        self.assertTrue(torch.isfinite(torch.tensor(losses[0])))

    def test_missing_resume_checkpoint_fails_instead_of_starting_fresh(self):
        with tempfile.TemporaryDirectory() as temp:
            for invalid in (Path(temp) / "missing.pt", Path(temp)):
                with self.subTest(checkpoint=invalid), self.assertRaisesRegex(
                    FileNotFoundError, "Resume checkpoint file not found"
                ):
                    train_geographic_ppo([self.scenario], updates=0, resume_checkpoint=invalid)

    def test_resume_restores_raw_state_dict_exactly_without_updates(self):
        model = SpatialActorCritic(
            grid_rows=15,
            grid_cols=15,
            action_dim=len(TACTICAL_ACTION_NAMES),
            recurrent=True,
        )
        with tempfile.TemporaryDirectory() as temp:
            checkpoint = Path(temp) / "weights.pt"
            torch.save(model.state_dict(), checkpoint)
            resumed, losses = train_geographic_ppo(
                [self.scenario], updates=0, recurrent=True, resume_checkpoint=checkpoint
            )
            self.assertEqual(losses, [])
            self.assertEqual(model.state_dict().keys(), resumed.state_dict().keys())
            for name, expected in model.state_dict().items():
                self.assertTrue(torch.equal(expected, resumed.state_dict()[name]), name)

    def test_smoke_cli_trains_saves_and_reloads_reproducibly(self):
        from nightjar.geographic_family import build_synthetic_cems_fixture
        from nightjar.geographic_ppo import main

        fixture = build_synthetic_cems_fixture("EMSR578", rows=25, cols=25)
        scenarios = [
            generate_geographic_scenario(
                fixture,
                index=index,
                split="development",
                window_rows=15,
                window_cols=15,
                horizon_s=300.0,
            )
            for index in range(2)
        ]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            outputs = []
            for run_idx in range(2):
                output = root / f"smoke-{run_idx}.json"
                checkpoint = root / f"smoke-{run_idx}.pt"
                with patch("sys.argv", [
                    "geographic_ppo", "--smoke", "--synthetic",
                    "--scenarios-dir", str(root), "--output", str(output),
                    "--checkpoint", str(checkpoint), "--seed", "137",
                ]), patch("nightjar.geographic_ppo.load_geographic_split", return_value=scenarios) as loader:
                    main()
                loader.assert_called_once_with("development", root, synthetic=True)
                report = json.loads(output.read_text())
                self.assertEqual(report["seed"], 137)
                self.assertEqual(
                    [row["scenario_id"] for row in report["dataset"]["splits"]["development"]],
                    [scenario["scenario_id"] for scenario in scenarios],
                )
                self.assertTrue(math.isfinite(report["loss"]))
                self.assertEqual(report["evaluation_metrics"]["rejected_orders"], 0)
                self.assertTrue(checkpoint.is_file())
                state = torch.load(checkpoint, weights_only=True, map_location="cpu")
                self.assertTrue(state)
                self.assertTrue(all(torch.isfinite(value).all() for value in state.values()))

                reloaded = SpatialActorCritic(
                    grid_rows=15,
                    grid_cols=15,
                    action_dim=len(TACTICAL_ACTION_NAMES),
                    recurrent=True,
                )
                reloaded.load_state_dict(torch.load(checkpoint, weights_only=True, map_location="cpu"))
                _, reloaded_metrics = rollout_geographic(reloaded, scenarios[0], deterministic=True)
                self.assertEqual(reloaded_metrics, report["evaluation_metrics"])
                outputs.append((report["loss"], report["evaluation_metrics"], state))

            self.assertEqual(outputs[0][0], outputs[1][0])
            self.assertEqual(outputs[0][1], outputs[1][1])
            for name in outputs[0][2]:
                self.assertTrue(torch.equal(outputs[0][2][name], outputs[1][2][name]), name)


if __name__ == "__main__":
    unittest.main()
