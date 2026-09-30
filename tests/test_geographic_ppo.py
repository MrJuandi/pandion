"""Unit tests for geographic PPO training and evaluation."""

import json
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
            reported = json.loads(output.read_text())["dataset"]["splits"]["development"]
            self.assertEqual([s["scenario_id"] for s in reported], expected)

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
