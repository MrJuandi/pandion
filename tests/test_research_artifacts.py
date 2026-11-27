"""Regression checks for identity, portable inference and experiment integrity."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from pandion.environment_family import _enrich_scenario
from pandion.experiment_catalog import scan_catalog, write_catalog
from pandion.geographic_family import build_synthetic_cems_fixture, generate_geographic_scenario, canonical_hash
from pandion.geographic_ppo import HAS_TORCH, _dataset_report, build_policy_model, main, rollout_geographic, with_fleet
from pandion.geographic_training_env import GeographicTrainingEnv
from pandion.identity import audit_independence, content_hash, partition_holdout
from pandion.multi_ignition_family import derive_scenario
from pandion.parametric_actions import num_actions
from pandion.policy_artifact import file_hash, load_policy_artifact, write_report
from pandion.workspace import policy_replay, build_fold_workspace


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

    def test_catalog_label_does_not_identify_packages_across_activations(self):
        splits = {"development": [{"activation_id": "EMSR578", "source_product_code": "PRODUCT",
                                   "source_zip_sha256": "a" * 64}],
                  "holdout": [{"activation_id": "EMSR583", "source_product_code": "PRODUCT",
                               "source_zip_sha256": "b" * 64}]}
        self.assertEqual(audit_independence(splits), [])
        self.assertEqual(audit_independence({"development": [{"source_product_code": "PRODUCT"}],
                                             "holdout": [{"source_product_code": "PRODUCT"}]}), [])

    def test_same_source_zip_is_detected_across_activations(self):
        digest = "a" * 64
        splits = {"development": [{"activation_id": "EMSR578", "source_zip_sha256": digest}],
                  "holdout": [{"activation_id": "EMSR583", "source_zip_sha256": digest}]}
        self.assertEqual(audit_independence(splits), [f"source_zip {digest} shared by development and holdout"])

    def test_same_activation_and_product_are_detected_without_zip_hashes(self):
        entry = {"activation_id": "EMSR578", "source_product_code": "PRODUCT"}
        self.assertEqual(audit_independence({"development": [entry], "holdout": [entry]}),
                         ["source_product EMSR578|PRODUCT shared by development and holdout"])


@unittest.skipUnless(HAS_TORCH, "PyTorch required")
class PolicyArtifactTests(unittest.TestCase):
    def _write(self, root, *, input_mode="legacy", action_mode="tactical", fleet=1, slot_features=False,
               thermal=False, skip_forced=False, drop_distances=(0,), action_features=False, action_skip=False,
               measured_intensity=False):
        import torch
        from test_policy_inputs import write_imagery
        parent = scenario()
        items = with_fleet([parent], fleet)
        imagery = root / "imagery"; imagery.mkdir(exist_ok=True)
        if input_mode == "image":
            write_imagery(imagery, parent, np.zeros((48, 48, 3), dtype=np.uint8))
        options = dict(input_mode=input_mode, action_mode=action_mode, recurrent=True,
                       imagery_dir=imagery if input_mode == "image" else None,
                       slot_features=slot_features, include_thermal=thermal, include_ros=False,
                       measured_intensity=measured_intensity)
        model = build_policy_model(items[0], **options, drop_distances=drop_distances,
                                   action_features=action_features, action_skip=action_skip)
        checkpoint, report_path = root / "model.pt", root / "report.json"
        torch.save(model.state_dict(), checkpoint)
        report = {"schema_version": "nightjar-geographic-ppo-selection-0.1", "recurrent": True,
                  "input_mode": input_mode, "action_mode": action_mode, "fleet_size": fleet,
                  "slot_features": slot_features, "include_thermal": thermal, "include_ros": False,
                  "skip_forced": skip_forced, "imagery_dir": str(imagery) if input_mode == "image" else None,
                  "drop_distances": list(drop_distances), "action_features": action_features,
                  "action_skip": action_skip, "measured_intensity": measured_intensity,
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

    def test_eval_records_and_reloads_effective_action_and_input_configuration(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            distances = (0, 1, 2)
            parent, original, checkpoint, report, options = self._write(
                root, input_mode="image", action_mode="parametric", fleet=2, slot_features=True,
                thermal=True, skip_forced=True, drop_distances=distances, action_features=True,
                action_skip=True, measured_intensity=True)
            output = root / "eval.json"
            sidecar = Path(f"{checkpoint}.artifact.json")
            training_binding = sidecar.read_bytes()
            argv = ["geographic_ppo", "--eval", "--synthetic", "--split", "validation", "--feedforward",
                    "--scenarios-dir", str(root), "--checkpoint", str(checkpoint),
                    "--training-report", str(report), "--output", str(output)]
            with patch("sys.argv", argv), patch("pandion.geographic_ppo.load_geographic_split", return_value=[parent]):
                main()

            recorded = json.loads(output.read_text())
            expected = {"input_mode": "image", "action_mode": "parametric", "recurrent": True, "fleet_size": 2,
                        "imagery_dir": str(options["imagery_dir"]), "include_thermal": True, "include_ros": False,
                        "measured_intensity": True, "skip_forced": True, "slot_features": True,
                        "action_features": True, "action_skip": True, "drop_distances": list(distances)}
            for config in (recorded, recorded["artifact"]["inference"]):
                self.assertEqual({key: config[key] for key in expected}, expected)
            self.assertEqual(set(recorded["dataset"]["splits"]), {"validation"})
            self.assertEqual(sidecar.read_bytes(), training_binding)
            reloaded, _ = load_policy_artifact(checkpoint, output, [parent])
            self.assertEqual(reloaded.drop_distances, original.drop_distances)
            self.assertEqual(reloaded.action_dim, original.action_dim)
            env = GeographicTrainingEnv(with_fleet([parent], 2)[0], action_mode=reloaded.action_mode,
                                        drop_distances=reloaded.drop_distances)
            self.assertEqual(len(env.action_mapper().compute_mask(env.reset())), num_actions(distances))

    def test_legacy_binding_can_omit_default_drop_distances(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent, _, checkpoint, report, _ = self._write(Path(tmp))
            value = json.loads(report.read_text())
            del value["artifact"]["inference"]["drop_distances"]
            report.write_text(json.dumps(value))
            Path(f"{checkpoint}.artifact.json").unlink()
            _, info = load_policy_artifact(checkpoint, report, [parent])
            self.assertEqual(info["checkpoint_report_binding"], "verified")
            self.assertEqual(info["inference"]["drop_distances"], [0])

    def test_report_records_distinct_final_and_best_checkpoint_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, _, checkpoint, report, _ = self._write(Path(tmp))
            value = json.loads(report.read_text())
            self.assertNotIn("best_checkpoint_sha256", value["artifact"])
            best = Path(f"{checkpoint}.best")
            best.write_bytes(b"different checkpoint bytes")
            write_report(value, report, checkpoint)
            artifact = json.loads(report.read_text())["artifact"]
            self.assertEqual(artifact["checkpoint_sha256"], file_hash(checkpoint))
            self.assertEqual(artifact["best_checkpoint_sha256"], file_hash(best))
            self.assertNotEqual(artifact["checkpoint_sha256"], artifact["best_checkpoint_sha256"])

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
            self.assertIn("Experiment catalog", (root / "catalog.html").read_text())
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
        from pandion.geographic_integrated import run_geographic_integrated
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); family, folds, cases = self._fixture(root)
            def load(checkpoint, report, scenarios, **options):
                self.assertEqual(checkpoint.stem, report.stem)
                return checkpoint.stem, {"inference": {"fleet_size": 2}}
            def replay(model, case):
                self.assertEqual(model, case["activation_id"])
                result = run_geographic_integrated(with_fleet([case], 2)[0], "no_intervention")
                result["policy"] = "Geographic PPO"
                return result
            with patch("pandion.workspace.load_geographic_split", return_value=cases), \
                 patch("pandion.workspace.load_policy", side_effect=load) as loader, \
                 patch("pandion.workspace.policy_replay", side_effect=replay):
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
            with patch("pandion.workspace.load_geographic_split", return_value=cases), \
                 patch("pandion.workspace.load_policy") as loader:
                with self.assertRaisesRegex(ValueError, "hash mismatch"):
                    build_fold_workspace(family, folds, root / "out")
                loader.assert_not_called()


if __name__ == "__main__":
    unittest.main()
