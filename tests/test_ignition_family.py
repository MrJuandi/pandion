"""Tests for the initial-attack ignition family derived from geographic scenarios."""

import json
from pathlib import Path
import tempfile
import unittest

from pandion.geographic_family import build_geographic_family, canonical_hash
from pandion.geographic_ppo import load_geographic_split
from pandion.ignition_family import (
    EDGE_MARGIN_CELLS,
    IGNITION_RADII_CELLS,
    build_ignition_family,
    derive_split,
    ignite,
    ignition_candidates,
)


def _make_burnable(scenario):
    """Give a synthetic parent a wide burnable mapped area so ignitions can be sampled."""
    fire = scenario["fire"]
    rows, cols = len(fire["fuel_grid"]), len(fire["fuel_grid"][0])
    for r in range(10, rows - 10):
        for c in range(10, cols - 10):
            fire["fuel_grid"][r][c] = 145
            fire["burn_mask"][r][c] = True
    scenario.pop("scenario_hash", None)
    scenario["scenario_hash"] = canonical_hash(scenario)
    return scenario


def _write_family(root):
    family = build_geographic_family(num_dev=2, num_val=1, num_res=2, synthetic=True)
    manifest = family["manifest"]
    for split, scenarios in family["scenarios"].items():
        (root / split).mkdir()
        for scenario, entry in zip(scenarios, manifest["scenarios"][split]):
            _make_burnable(scenario)
            entry["hash"] = scenario["scenario_hash"]
            (root / split / f"{scenario['scenario_id']}.json").write_text(json.dumps(scenario))
    (root / "manifest.json").write_text(json.dumps(manifest))
    return family


class IgnitionFamilyTests(unittest.TestCase):
    def setUp(self):
        family = build_geographic_family(num_dev=1, num_val=1, num_res=2, synthetic=True)
        self.parent = _make_burnable(family["scenarios"]["development"][0])

    def test_candidates_are_inside_mapped_burn_and_away_from_edges(self):
        fire = self.parent["fire"]
        rows = len(fire["fuel_grid"])
        candidates = ignition_candidates(fire)
        self.assertTrue(candidates)
        for r, c in candidates:
            self.assertTrue(fire["burn_mask"][r][c])
            self.assertTrue(EDGE_MARGIN_CELLS <= r < rows - EDGE_MARGIN_CELLS)

    def test_ignition_replaces_cems_burn_and_keeps_landscape(self):
        child = ignite(self.parent, 3, (20, 20), 2)
        self.assertEqual(sum(map(sum, child["fire"]["burn_mask"])), 25)
        self.assertEqual(child["fire"]["fuel_grid"], self.parent["fire"]["fuel_grid"])
        self.assertEqual(child["observation_schedule"], self.parent["observation_schedule"])
        self.assertEqual(child["scenario_id"], self.parent["scenario_id"] + "-ig03")
        self.assertEqual(child["provenance"]["ignition_parent_scenario_hash"], self.parent["scenario_hash"])
        self.assertEqual(child["scenario_hash"], canonical_hash({k: v for k, v in child.items() if k != "scenario_hash"}))

    def test_ignition_skips_unburnable_cells(self):
        self.parent["fire"]["fuel_grid"][20][21] = 0
        child = ignite(self.parent, 0, (20, 20), 1)
        self.assertFalse(child["fire"]["burn_mask"][20][21])
        self.assertEqual(sum(map(sum, child["fire"]["burn_mask"])), 8)

    def test_derivation_is_deterministic_and_distinct(self):
        first = derive_split([self.parent], 4)
        second = derive_split([self.parent], 4)
        self.assertEqual([s["scenario_hash"] for s in first], [s["scenario_hash"] for s in second])
        centres = {tuple(s["provenance"]["ignition"]["centre_row_col"]) for s in first}
        self.assertEqual(len(centres), 4)
        self.assertTrue(all(s["provenance"]["ignition"]["radius_cells"] in IGNITION_RADII_CELLS for s in first))

    def test_built_family_loads_with_inherited_splits(self):
        with tempfile.TemporaryDirectory() as temp:
            parent_dir, out_dir = Path(temp) / "parent", Path(temp) / "out"
            parent_dir.mkdir()
            _write_family(parent_dir)
            result = build_ignition_family(parent_dir, out_dir, {"development": 3, "validation": 2, "reserved": 1},
                                           synthetic=True)
            self.assertEqual(result["manifest"]["counts"], {"development": 6, "validation": 2, "reserved": 2, "total": 10})
            development = load_geographic_split("development", out_dir, synthetic=True)
            self.assertEqual({s["activation_id"] for s in development}, {"EMSR578"})
            self.assertEqual(len(development), 6)
            with self.assertRaises(FileExistsError):
                build_ignition_family(parent_dir, out_dir, synthetic=True)


if __name__ == "__main__":
    unittest.main()
