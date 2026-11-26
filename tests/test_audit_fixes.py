"""Regression tests for repository audit fixes."""

from __future__ import annotations

import ast
import contextlib
import io
import sys
import tempfile
import typing
import unittest
from pathlib import Path
from unittest.mock import patch

from pandion import evaluate_reserved
from pandion.geographic_benchmark import run_geographic_benchmark


REPO_ROOT = Path(__file__).resolve().parents[1]


def _is_torch_load(node: ast.Call) -> bool:
    func = node.func
    return (
        isinstance(func, ast.Attribute)
        and func.attr == "load"
        and isinstance(func.value, ast.Name)
        and func.value.id == "torch"
    )


def _has_weights_only_true(node: ast.Call) -> bool:
    for kw in node.keywords:
        if kw.arg == "weights_only":
            return isinstance(kw.value, ast.Constant) and kw.value.value is True
    return False


class AuditFixesTests(unittest.TestCase):
    def test_torch_load_calls_require_weights_only_true(self):
        pandion_dir = REPO_ROOT / "pandion"
        missing: list[str] = []
        for py_path in sorted(pandion_dir.rglob("*.py")):
            tree = ast.parse(py_path.read_text(encoding="utf-8"), filename=str(py_path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and _is_torch_load(node):
                    if not _has_weights_only_true(node):
                        rel = py_path.relative_to(REPO_ROOT)
                        missing.append(f"{rel}:{node.lineno}")
        self.assertEqual(missing, [], f"torch.load calls missing weights_only=True: {missing}")

    def test_geographic_benchmark_type_hints_resolve(self):
        hints = typing.get_type_hints(run_geographic_benchmark)
        self.assertIn("random_seeds", hints)
        self.assertIn("return", hints)

    def test_evaluate_reserved_cli_missing_checkpoint_reports_clean_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            missing_ckpt = Path(tmpdir) / "runs" / "ppo-tactical-v2.pt"
            stderr = io.StringIO()
            with (
                patch.object(evaluate_reserved, "DEFAULT_CHECKPOINT", missing_ckpt),
                patch.object(sys, "argv", ["pandion.evaluate_reserved", "--split-version", "v2"]),
                contextlib.redirect_stderr(stderr),
            ):
                with self.assertRaises(SystemExit) as ctx:
                    evaluate_reserved.main()
            self.assertEqual(ctx.exception.code, 2)
            message = stderr.getvalue()
            self.assertIn("ppo-tactical-v2.pt", message)
            self.assertIn("pandion.ppo --select", message)


if __name__ == "__main__":
    unittest.main()
