"""Unit tests for geographic PPO training and evaluation."""

import json
from pathlib import Path
import tempfile
import unittest

from nightjar.geographic_family import (
    build_synthetic_cems_fixture,
    generate_geographic_scenario,
)
from nightjar.geographic_ppo import (
    HAS_TORCH,
    load_geographic_split,
    ppo_update_geographic,
    returns_and_advantages,
    rollout_geographic,
    train_geographic_ppo,
)
from nightjar.model import SpatialActorCritic, TACTICAL_ACTION_NAMES

if HAS_TORCH:
    import torch


class TestGeographicLoader(unittest.TestCase):
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

    def test_train_geographic_ppo_single_update(self):
        model, losses = train_geographic_ppo(
            [self.scenario],
            updates=1,
            episodes_per_update=1,
            recurrent=True,
        )
        self.assertEqual(len(losses), 1)
        self.assertTrue(torch.isfinite(torch.tensor(losses[0])))


if __name__ == "__main__":
    unittest.main()
