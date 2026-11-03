"""Regression checks for identity, portable inference and experiment integrity."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from nightjar.environment_family import _enrich_scenario
from nightjar.experiment_catalog import scan_catalog, write_catalog
from nightjar.geographic_family import build_synthetic_cems_fixture, generate_geographic_scenario, canonical_hash
from nightjar.geographic_ppo import HAS_TORCH, _dataset_report, build_policy_model, rollout_geographic, with_fleet
from nightjar.identity import audit_independence, content_hash, partition_holdout, scenario_entry
from nightjar.multi_ignition_family import derive_scenario
from nightjar.policy_artifact import load_policy_artifact, write_report
from nightjar.workspace import policy_replay, build_fold_workspace


def scenario(rows=12, horizon=180):
    source = build_synthetic_cems_fixture("EMSR578", utm_zone=31, rows=50, cols=50)
    result = generate_geographic_scenario(source, index=0, split="development",
                                          window_rows=rows, window_cols=rows, horizon_s=horizon)
    result["fire"]["fuel_grid"] = [[101] * rows for _ in range(rows)]
    result["fire"]["burn_mask"] = [[r == rows // 2 and c == rows // 2 for c in range(rows)] for r in range(rows)]
    result.pop("scenario_hash")
    result["scenario_hash"] = canonical_hash(result)
    return result


class IdentityTests(unittest.TestCase):
    def test_generation_clocks_are_separate_from_observation_times(self):
        first = {"environment": {"created_at_utc": "today"}, "acquired_at_utc": "2022-01-01", "scenario_hash": "old"}
        second = deepcopy(first); second["environment"]["created_at_utc"] = "tomorrow"; second["scenario_hash"] = "new"
        self.assertEqual(content_hash(first), content_hash(second))
        second["acquired_at_utc"] = "2022-01-02"
        self.assertNotEqual(content_hash(first), content_hash(second))
        self.assertEqual(first["scenario_hash"], "old")

    def test_new_environment_content_and_child_draws_are_reproducible(self):
        parent = scenario(rows=40, horizon=1800)
        parent["fire"]["burn_mask"] = [[8 <= r < 32 and 8 <= c < 32 for c in range(40)] for r in range(40)]
        parent.pop("scenario_hash"); parent["scenario_hash"] = canonical_hash(parent)
        def enrich(clock):
            return _enrich_scenario(parent, [[100.0] * 40 for _ in range(40)],
                {"provider": "fixture", "created_at_utc": clock}, audit=None, pilot_scenario=None,
                station_wind_factor=1, moisture_mode="fixed")
        a, b = enrich("today"), enrich("tomorrow")
        self.assertEqual(a, b)
        self.assertNotIn("created_at_utc", a["environment"])
        x, y = derive_scenario(a, 0, seed=42), derive_scenario(b, 0, seed=42)
        self.assertEqual(x, y)
        self.assertNotEqual(x["scenario_hash"], derive_scenario(a, 0, seed=43)["scenario_hash"])

    def test_holdout_excludes_all_aois_of_the_activation(self):
        items = [{"activation_id": a} for a in ("EMSR602_AOI02", "EMSR602_AOI04", "EMSR578")]
        training, held = partition_holdout(items, "EMSR602_AOI02")
        self.assertEqual([s["activation_id"] for s in training], ["EMSR578"])
        self.assertEqual(len(held), 2)
        self.assertTrue(audit_independence({"development": [{"incident_group": "EMSR602"}],
                                            "holdout": [{"incident_group": "EMSR602"}]}))


@unittest.skipUnless(HAS_TORCH, "PyTorch required")
class PolicyArtifactTests(unittest.TestCase):
    def _write(self, root, *, input_mode="legacy", action_mode="tactical", fleet=1, slot_features=False,
               thermal=False, skip_forced=False):
        import torch
        from test_policy_inputs import write_imagery
        parent = scenario()
        items = with_fleet([parent], fleet)
        imagery = root / "imagery"; imagery.mkdir(exist_ok=True)
        if input_mode == "image":
            write_imagery(imagery, parent, np.zeros((48, 48, 3), dtype=np.uint8))
        options = dict(input_mode=input_mode, action_mode=action_mode, recurrent=True,
                       imagery_dir=imagery if input_mode == "image" else None,
                       slot_features=slot_features, include_thermal=thermal, include_ros=False)
        model = build_policy_model(items[0], **options)
        checkpoint, report_path = root / "model.pt", root / "report.json"
        torch.save(model.state_dict(), checkpoint)
        report = {"schema_version": "nightjar-geographic-ppo-selection-0.1", "recurrent": True,
                  "input_mode": input_mode, "action_mode": action_mode, "fleet_size": fleet,
                  "slot_features": slot_features, "include_thermal": thermal, "include_ros": False,
                  "skip_forced": skip_forced, "imagery_dir": str(imagery) if input_mode == "image" else None,
                  "dataset": _dataset_report(root, {"development": items})}
        write_report(report, report_path, checkpoint)
        return parent, model, checkpoint, report_path, options

    def test_current_architectures_round_trip_and_share_evaluation_execution(self):
        cases = [("legacy", "tactical", 1, False, False), ("cells", "spatial", 1, False, False),
                 ("image", "parametric", 2, True, True)]
        for input_mode, action_mode, fleet, slots, thermal in cases:
            with self.subTest(input_mode=input_mode), tempfile.TemporaryDirectory() as tmp:
                parent, original, checkpoint, report, options = self._write(Path(tmp), input_mode=input_mode,
                    action_mode=action_mode, fleet=fleet, slot_features=slots, thermal=thermal, skip_forced=True)
                loaded, info = load_policy_artifact(checkpoint, report, [parent])
                self.assertEqual(info["checkpoint_report_binding"], "verified")
                replay = policy_replay(loaded, parent)
                replay_out = []
                rollout_geographic(original, with_fleet([parent], fleet)[0], deterministic=True,
                    inference_only=True, skip_forced=True, replay_out=replay_out,
                    **{k: v for k, v in options.items() if k not in ("action_mode", "recurrent", "slot_features")})
                self.assertEqual(replay["outcomes"], replay_out[0]["outcomes"])
                self.assertEqual(replay["decisions"], replay_out[0]["decisions"])
                self.assertEqual(len(replay["aircraft_initial_state"]), fleet)

    def test_swapped_weights_and_modified_report_are_rejected(self):
        import torch
        with tempfile.TemporaryDirectory() as tmp:
            parent, _, checkpoint, report, _ = self._write(Path(tmp))
            original = checkpoint.read_bytes()
            state = torch.load(checkpoint, weights_only=True)
            next(iter(state.values())).add_(1)
            torch.save(state, checkpoint)
            with self.assertRaisesRegex(ValueError, "binding mismatch"):
                load_policy_artifact(checkpoint, report, [parent])
            checkpoint.write_bytes(original)
            value = json.loads(report.read_text()); value["seed"] = 999; report.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, "sidecar/report"):
                load_policy_artifact(checkpoint, report, [parent])

    def test_catalog_verifies_weights_without_loading_them(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, _, checkpoint, report, _ = self._write(root)
            with patch("torch.load", side_effect=AssertionError("catalog must not load models")):
                catalog = scan_catalog([root])
            experiment = next(e for e in catalog["entries"] if e["kind"] == "experiment")
            self.assertEqual(experiment["binding"], "verified")
            self.assertEqual(experiment["checkpoints"], [str(checkpoint.resolve())])
            write_catalog(catalog, root / "catalog.json", root / "catalog.html")
            self.assertIn("Catálogo de experimentos", (root / "catalog.html").read_text())
            checkpoint.unlink()
            experiment = next(e for e in scan_catalog([root])["entries"] if e["kind"] == "experiment")
            self.assertEqual(experiment["binding"], "missing_checkpoint")


class FoldWorkspaceTests(unittest.TestCase):
    def _fixture(self, root):
        family = root / "family"; family.mkdir()
        (family / "manifest.json").write_text("{}")
        folds = root / "D-71"; folds.mkdir()
        cases = []
        for fold in ("EMSR578", "EMSR583"):
            case = scenario(horizon=60)
            case.update(scenario_id=fold + "-case", activation_id=fold, name=fold)
            case["scenario_hash"] = canonical_hash(case)
            cases.append(case)
            (folds / (fold + ".pt")).write_bytes(fold.encode())
            report = {"schema_version": "nightjar-geographic-ppo-selection-0.1",
                      "holdout_activation": fold, "fleet_size": 2,
                      "holdout": {"rows": [{"scenario_id": case["scenario_id"]}]},
                      "dataset": {"splits": {"holdout": [{"scenario_id": case["scenario_id"],
                                  "scenario_hash": with_fleet([case], 2)[0]["scenario_hash"]}]}}}
            (folds / (fold + ".json")).write_text(json.dumps(report))
        return family, folds, cases

    def test_each_case_uses_its_fold_checkpoint_and_matched_baseline(self):
        from nightjar.geographic_integrated import run_geographic_integrated
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); family, folds, cases = self._fixture(root)
            def load(checkpoint, report, scenarios, **options):
                self.assertEqual(checkpoint.stem, report.stem)
                return checkpoint.stem, {"inference": {"fleet_size": 2}}
            def replay(model, case):
                self.assertEqual(model, case["activation_id"])
                result = run_geographic_integrated(with_fleet([case], 2)[0], "no_intervention")
                result["policy"] = "PPO geográfico"
                return result
            with patch("nightjar.workspace.load_geographic_split", return_value=cases), \
                 patch("nightjar.workspace.load_policy", side_effect=load) as loader, \
                 patch("nightjar.workspace.policy_replay", side_effect=replay):
                manifest = build_fold_workspace(family, folds, root / "out", episodes=1, skip_forced=True)
            self.assertEqual(loader.call_count, 2)
            self.assertEqual(manifest["replay_count"], 8)
            self.assertEqual(manifest["episode_splits"], ["development"])
            self.assertEqual([t["checkpoint_name"] for t in manifest["training_runs"]], ["EMSR578.pt", "EMSR583.pt"])
            replays = [json.loads(p.read_text()) for p in (root / "out").glob("EMSR*.json")]
            self.assertTrue(all(len(r["aircraft_initial_state"]) == 2 for r in replays))

    def test_mismatched_holdout_is_rejected_before_loading_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); family, folds, cases = self._fixture(root)
            report = folds / "EMSR578.json"; value = json.loads(report.read_text())
            value["dataset"]["splits"]["holdout"][0]["scenario_hash"] = "wrong"
            report.write_text(json.dumps(value))
            with patch("nightjar.workspace.load_geographic_split", return_value=cases), \
                 patch("nightjar.workspace.load_policy") as loader:
                with self.assertRaisesRegex(ValueError, "hash mismatch"):
                    build_fold_workspace(family, folds, root / "out")
                loader.assert_not_called()


if __name__ == "__main__":
    unittest.main()
