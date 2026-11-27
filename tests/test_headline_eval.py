"""Checks for packaged headline evaluation and reference statistics."""
from contextlib import redirect_stdout
from hashlib import sha256
from io import StringIO
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from pandion.geographic_family import build_synthetic_cems_fixture, canonical_hash, generate_geographic_scenario
from pandion.headline_eval import bootstrap_ci, main
from pandion.model import HAS_TORCH


def _reference_ci(diff):
    means = np.random.default_rng(0).choice(diff, size=(10000, len(diff))).mean(axis=1)
    return [round(float(np.percentile(means, p)), 2) for p in (2.5, 97.5)]


class HeadlineSummaryTests(unittest.TestCase):
    def test_bootstrap_matches_reference(self):
        diff = np.array([-9.3, -2.0, 0.9, 2.1, 6.7])
        self.assertEqual(bootstrap_ci(diff), _reference_ci(diff))

    def test_summarize_merges_by_scenario_and_matches_reference(self):
        refs = {"heuristic": np.array([9.5, 12.0, 6.5]), "hold_aware_heuristic": np.array([7.0, 10.0, 5.0])}
        a = {"a42": np.array([8., 13., 7.]), "a43": np.array([6., 10., 5.]), "a44": np.array([7., 11., 6.])}
        b = {"b42": np.array([9., 12., 8.]), "b43": np.array([7., 9., 6.]), "b44": np.array([5., 8., 4.])}
        ids = ["s1", "s2", "s3"]
        splits = ("holdout", "development_sample")

        def evaluation(indices, columns):
            rows = [
                {"scenario_id": ids[i], **{key: float(values[i]) for key, values in columns.items()}}
                for i in indices
            ]
            return {split: {"rows": rows} for split in splits}

        with tempfile.TemporaryDirectory() as tmp:
            first, second, output = [Path(tmp) / name for name in ("first.json", "second.json", "summary.json")]
            first.write_text(json.dumps(evaluation([2, 0, 1], {**refs, "a42": a["a42"], "b42": b["b42"]})))
            more_seeds = evaluation([1, 2, 0], {**refs, **{k: v for k, v in (a | b).items() if not k.endswith("42")}})
            second.write_text(json.dumps(more_seeds))
            argv = [
                "summarize", "--evaluation", str(first), "--evaluation", str(second), "--policy", "a=a42,a43,a44",
                "--policy", "b=b42,b43,b44", "--pair", "a", "b", "--output", str(output),
            ]
            with redirect_stdout(StringIO()) as stdout:
                main(argv)
            result = json.loads(output.read_text())
            self.assertEqual(json.loads(stdout.getvalue()), result)
            self.assertEqual(set(result), set(splits))
            for summary in result.values():
                self.assertEqual(summary["n"], 3)
                for name, seeds in (("a", a), ("b", b)):
                    mean_policy = np.mean(list(seeds.values()), axis=0)
                    expected = {"policy": round(float(mean_policy.mean()), 2)}
                    for ref, reference in refs.items():
                        diff = mean_policy - reference
                        expected[f"minus_{ref}"] = round(float(diff.mean()), 2)
                        expected[f"ci95_{ref}"] = _reference_ci(diff)
                        expected[f"per_seed_minus_{ref}"] = {
                            label: round(float((values - reference).mean()), 2) for label, values in seeds.items()
                        }
                        expected[f"bew_{ref}"] = [np.count_nonzero(diff < 0), np.count_nonzero(diff == 0),
                                                   np.count_nonzero(diff > 0)]
                    self.assertEqual(summary[name], expected)
                pair_diffs = {label: a[label] - other for label, other in zip(a, b.values(), strict=True)}
                pair_mean = np.mean(list(pair_diffs.values()), axis=0)
                self.assertEqual(summary["a_minus_b"], {
                    "mean": round(float(pair_mean.mean()), 2),
                    "ci95": _reference_ci(pair_mean),
                    "per_seed": {label: round(float(values.mean()), 2) for label, values in pair_diffs.items()},
                })

            more_seeds["holdout"]["rows"][0]["scenario_id"] = "different"
            second.write_text(json.dumps(more_seeds))
            with self.assertRaisesRegex(ValueError, "mismatched scenario IDs for holdout"):
                main(argv)


@unittest.skipUnless(HAS_TORCH, "PyTorch required")
class HeadlineEvaluationTests(unittest.TestCase):
    def test_evaluate_synthetic_family(self):
        import torch

        from pandion.geographic_ppo import build_policy_model

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            family, imagery = root / "family", root / "imagery"
            development = family / "development"
            development.mkdir(parents=True)
            imagery.mkdir()
            entries, ids = [], []
            for activation, index in (("EMSR578", 0), ("EMSR583", 0), ("EMSR583", 1)):
                source = build_synthetic_cems_fixture(activation, utm_zone=31 if activation == "EMSR578" else 30)
                scenario = generate_geographic_scenario(
                    source, index, "development", window_rows=12, window_cols=12, horizon_s=180,
                )
                scenario["fire"]["fuel_grid"] = [[101] * 12 for _ in range(12)]
                scenario["fire"]["burn_mask"] = [[r == 6 and c == 6 for c in range(12)] for r in range(12)]
                del scenario["scenario_hash"]
                scenario["scenario_hash"] = canonical_hash(scenario)
                scenario_id = scenario["scenario_id"]
                ids.append(scenario_id)
                (development / f"{scenario_id}.json").write_text(json.dumps(scenario))
                entries.append({
                    "scenario_id": scenario_id, "activation_id": activation,
                    "hash": scenario["scenario_hash"], "source_kind": scenario["source_kind"],
                })
            (family / "manifest.json").write_text(json.dumps({
                "source_kind": "synthetic_fixture", "scenarios": {"development": entries},
            }))

            torch.manual_seed(0)
            model = build_policy_model(scenario, recurrent=False)
            checkpoint, report, output = root / "model.pt", root / "training.json", root / "evaluation.json"
            torch.save(model.state_dict(), checkpoint)
            report.write_text(json.dumps({
                "input_mode": "legacy", "include_ros": False, "include_thermal": False, "action_mode": "tactical",
                "recurrent": False, "slot_features": False, "drop_distances": [0], "action_features": False,
                "skip_forced": True, "fleet_size": 1, "holdout_activation": "EMSR578", "imagery_dir": str(imagery),
            }))
            proc = subprocess.run([
                sys.executable, "-m", "pandion.headline_eval", "evaluate", "--report", str(report),
                "--scenarios-dir", str(family), "--imagery-dir", str(imagery),
                "--checkpoint", f"untrained={checkpoint}", "--checkpoint", f"same-model={checkpoint}",
                "--workers", "1", "--sample", "1", "--sample-seed", "7", "--synthetic", "--output", str(output),
            ], capture_output=True, text=True, timeout=18)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertEqual([line.split(" ", 1)[0] for line in proc.stdout.splitlines()],
                             ["holdout", "development_sample"])
            raw = output.read_text()
            result = json.loads(raw)
            self.assertNotIn(str(root), raw)
            self.assertEqual(result["package_version"], "0.9.0")
            self.assertTrue(result["synthetic"])
            self.assertEqual((result["sample"], result["sample_seed"]), (1, 7))
            self.assertEqual(result["report"], {"filename": report.name, "sha256": sha256(report.read_bytes()).hexdigest()})
            self.assertEqual(result["checkpoints"], {
                label: {"filename": checkpoint.name, "sha256": sha256(checkpoint.read_bytes()).hexdigest()}
                for label in ("untrained", "same-model")
            })
            train = sorted(ids[1:])
            random.Random(7).shuffle(train)
            for split, scenario_id in (("holdout", ids[0]), ("development_sample", train[0])):
                rows, summary = result[split]["rows"], result[split]["summary"]
                self.assertEqual(summary["n"], 1)
                self.assertEqual(len(rows), 1)
                self.assertEqual(rows[0]["scenario_id"], scenario_id)
                self.assertEqual(rows[0]["untrained"], rows[0]["same-model"])
                self.assertEqual(summary["untrained"], summary["same-model"])
                self.assertIn("no_intervention", rows[0])
                self.assertEqual(set(summary["untrained"]), {
                    "policy", "minus_heuristic", "ci95_heuristic", "better_equal_worse_heuristic",
                    "minus_hold_aware_heuristic", "ci95_hold_aware_heuristic", "better_equal_worse_hold_aware_heuristic",
                })
