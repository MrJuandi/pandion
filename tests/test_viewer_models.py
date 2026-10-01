from copy import deepcopy
from hashlib import sha256
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path
import tempfile
import unittest

from nightjar.discrimination_family import build_scenarios
from nightjar.integrated import run_integrated
from nightjar.viewer_models import controller_replay, load_controller_checkpoint, physical_model_record

HAS_TORCH = importlib.util.find_spec("torch") is not None


class PhysicalModelRecordTests(unittest.TestCase):
    def test_records_supplied_physics_and_same_scenario_hash(self):
        scenario = build_scenarios()[0][0]
        record = physical_model_record(scenario)
        replay = run_integrated(scenario, "no_intervention")
        self.assertEqual(record["scenario_sha256"], replay["scenario_sha256"])
        details = {row["label"]: row["value"] for row in record["details"]}
        self.assertEqual(details["Scenario seed"], 500000)
        self.assertEqual(details["Midflame wind to [east, north] (m/s)"], scenario["fire"]["midflame_wind_to_m_s"])
        self.assertEqual(details["Water-effect radius (m)"], 35)
        self.assertIn("unsourced", details["Water-effect assumption"])
        self.assertEqual(record["config"]["horizon_s"], 1200)
        self.assertTrue(all(len(source["sha256"]) == 64 and not Path(source["file"]).is_absolute()
                            for source in record["source_files"]))

    def test_rejects_unsupported_geographic_scenario(self):
        with self.assertRaisesRegex(ValueError, "integrated-synthetic"):
            physical_model_record({"schema_version": "nightjar-geographic-scenario-0.1"})


@unittest.skipUnless(HAS_TORCH, "PyTorch unavailable; checkpoint tests require requirements-rl.txt")
class ControllerModelRecordTests(unittest.TestCase):
    def setUp(self):
        import torch
        from nightjar.model import SpatialActorCritic
        self.torch = torch
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(123)
            self.model = SpatialActorCritic(grid_rows=11, grid_cols=11, hidden_dim=16,
                                           recurrent=True, lstm_hidden_dim=8, action_dim=9)
        self.config = {"grid_rows": 11, "grid_cols": 11, "recurrent": True,
                       "action_mode": "tactical", "action_dim": 9}
        self.bundle = {"schema_version": "nightjar-ppo-selection-0.1",
                       "model_config": self.config, "model_state_dict": self.model.state_dict(),
                       "selected_seed": 123, "training_config": {"updates": 1}}
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "model.pt"
        torch.save(self.bundle, self.path)

    def test_binds_exact_bytes_architecture_and_weights(self):
        loaded, record = load_controller_checkpoint(self.path)
        self.assertEqual(record["checkpoint"]["sha256"], sha256(self.path.read_bytes()).hexdigest())
        self.assertEqual(record["checkpoint"]["filename"], "model.pt")
        self.assertEqual(record["selected_seed"], 123)
        self.assertEqual(record["config"]["in_channels"], 10)
        self.assertEqual(record["config"]["hidden_dim"], 16)
        self.assertEqual(record["config"]["lstm_hidden_dim"], 8)
        self.assertEqual(record["parameter_count"], sum(p.numel() for p in loaded.parameters()))
        self.assertFalse(loaded.training)
        for key, value in self.model.state_dict().items():
            self.assertTrue(self.torch.equal(value, loaded.state_dict()[key]))

    def test_raw_weights_require_explicit_configuration(self):
        self.torch.save(self.model.state_dict(), self.path)
        with self.assertRaisesRegex(ValueError, "model_config"):
            load_controller_checkpoint(self.path)
        loaded, record = load_controller_checkpoint(self.path, self.config)
        self.assertEqual(loaded.action_mode, "tactical")
        self.assertIsNone(record["selected_seed"])
        self.assertIsNone(record["training_config"])

    def test_rejects_conflicting_configuration_and_nonfinite_weights(self):
        for config in ({**self.config, "action_dim": 8}, {**self.config, "hidden_dim": 32},
                       {**self.config, "grid_rows": 10}, {**self.config, "action_mode": "unknown"},
                       {**self.config, "recurrent": False}, {**self.config, "grid_cols": True}):
            with self.subTest(config=config), self.assertRaises(ValueError):
                load_controller_checkpoint(self.path, config)
        invalid = deepcopy(self.bundle)
        invalid["model_state_dict"]["actor_head.weight"][0, 0] = float("nan")
        self.torch.save(invalid, self.path)
        with self.assertRaisesRegex(ValueError, "finite"):
            load_controller_checkpoint(self.path)

    def test_records_deterministic_development_episode_without_modifying_weights(self):
        model, record = load_controller_checkpoint(self.path)
        before = {key: value.clone() for key, value in model.state_dict().items()}
        scenario = build_scenarios()[0][0]
        first = controller_replay(model, record, scenario)
        second = controller_replay(model, record, scenario)
        self.assertEqual(first, second)
        self.assertEqual(first["scenario_id"], "discrimination-development-00")
        self.assertEqual(first["policy"], "ppo")
        self.assertEqual(first["controller_id"], record["id"])
        self.assertEqual(first["checkpoint_sha256"], record["checkpoint"]["sha256"])
        self.assertEqual(first["aircraft_metrics"]["rejected_orders"], 0)
        for key, value in model.state_dict().items():
            self.assertTrue(self.torch.equal(value, before[key]))
        self.assertFalse(model.training)
        wrong_grid = deepcopy(record)
        wrong_grid["config"]["grid_rows"] = 10
        with self.assertRaisesRegex(ValueError, "scenario grid"):
            controller_replay(model, wrong_grid, scenario)

    def test_cli_binds_checkpoint_and_physics_to_same_development_replays(self):
        output = Path(self.directory.name) / "viewer.html"
        subprocess.run([sys.executable, "-m", "nightjar.viewer", "--demo", "--checkpoint", str(self.path),
                        "--output", str(output)], cwd=Path(__file__).parents[1],
                       check=True, text=True, capture_output=True)
        page = output.read_text()
        replays = json.loads(re.search(r"const INITIAL_REPLAYS = (.*);", page).group(1))
        models = json.loads(re.search(r"const INITIAL_MODELS = (.*);", page).group(1))
        self.assertEqual({replay["scenario_id"] for replay in replays}, {"discrimination-development-00"})
        self.assertEqual({replay["policy"] for replay in replays}, {"ppo", "heuristic", "containment_heuristic", "no_intervention"})
        controller = next(record for record in models if record["role"] == "controller")
        physical = next(record for record in models if record["role"] == "physical")
        ppo = next(replay for replay in replays if replay["policy"] == "ppo")
        self.assertEqual(ppo["controller_id"], controller["id"])
        self.assertEqual(ppo["checkpoint_sha256"], controller["checkpoint"]["sha256"])
        self.assertTrue(all(replay["scenario_sha256"] == physical["scenario_sha256"] for replay in replays))
        self.assertTrue(all(replay["physical_model_id"] == physical["id"] for replay in replays))


if __name__ == "__main__":
    unittest.main()
