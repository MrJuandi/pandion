"""Index existing reports, datasets, checkpoints and replays without executing them."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import html
import json
from pathlib import Path

from .identity import audit_independence, content_hash
from .policy_artifact import file_hash, verify_binding

SCHEMA_VERSION = "nightjar-experiment-catalog-0.1"


def scan_catalog(roots: list[Path]) -> dict:
    files = set()
    for root in roots:
        root = root.resolve()
        if not root.exists():
            raise FileNotFoundError(root)
        files.update([root] if root.is_file() else (p for p in root.rglob("*") if p.is_file()
                     and not any(part in ("reserved", "development", "validation", "raw", ".venv", ".git")
                                 for part in p.relative_to(root).parts[:-1])))
    weights, entries, errors = {}, [], []
    for path in sorted(files):
        if path.suffix in (".pt", ".pth", ".best", ".partial"):
            digest = file_hash(path)
            weights.setdefault(digest, []).append(str(path))
            entries.append({"kind": "checkpoint", "path": str(path), "sha256": digest,
                            "binding": "unbound", "bytes": path.stat().st_size})
    for path in sorted(files):
        if path.suffix != ".json":
            continue
        try:
            value = json.loads(path.read_text())
            if not isinstance(value, dict):
                continue
            schema = value.get("schema_version", "")
            if schema.startswith("nightjar-geographic-ppo-"):
                kind = "experiment"
            elif "family" in schema and isinstance(value.get("scenarios"), dict):
                kind = "dataset"
            elif "replay" in schema:
                kind = "replay"
            elif schema.startswith("nightjar-workspace-"):
                kind = "workspace"
            else:
                continue
            entry = {"kind": kind, "path": str(path), "sha256": file_hash(path), "schema_version": schema}
            if kind == "experiment":
                artifact = value.get("artifact", {})
                digest = artifact.get("checkpoint_sha256")
                matching = weights.get(digest, [])
                binding_errors = []
                verified = []
                for checkpoint in matching:
                    try:
                        if verify_binding(Path(checkpoint), path, value):
                            verified.append(checkpoint)
                    except (OSError, ValueError) as exc:
                        binding_errors.append(str(exc))
                entry.update({"seed": value.get("selected_seed", value.get("seed")), "input_mode": value.get("input_mode", "legacy"),
                              "action_mode": value.get("action_mode", "tactical"), "fleet_size": value.get("fleet_size", 1),
                              "configuration": {key: value.get(key) for key in ("recurrent", "include_ros", "include_thermal", "slot_features", "skip_forced", "ppo", "training_settings", "updates", "episodes_per_update")},
                              "dataset": value.get("dataset", {}), "execution": value.get("execution"),
                              "checkpoints": matching, "checkpoint_sha256": digest,
                              "binding": "invalid_binding" if binding_errors else "verified" if verified else "missing_checkpoint" if digest else "legacy_unbound",
                              "binding_errors": binding_errors,
                              "independence_issues": audit_independence(value.get("dataset", {}).get("splits", {})),
                              "metrics": {key: value[key] for key in ("evaluation", "evaluation_metrics", "holdout", "best_mean_burned_ha", "reports") if key in value}})
                for weight in entries:
                    if weight["kind"] == "checkpoint" and weight["path"] in verified:
                        weight.update(binding="verified", report=str(path), report_sha256=entry["sha256"])
                replays = [e for e in value.get("replays", []) if isinstance(e, dict)]
                entry["replays"] = replays
            elif kind == "dataset":
                entry.update(content_sha256=content_hash(value), counts=value.get("counts"),
                             scenario_hashes={split: [{"scenario_id": s.get("scenario_id"), "hash": s.get("hash"),
                                                       "activation_id": s.get("activation_id")} for s in scenarios]
                                              for split, scenarios in value["scenarios"].items()})
            elif kind == "replay":
                entry.update(scenario_hash=value.get("scenario_hash"), policy=value.get("policy"),
                             metrics=value.get("outcomes"), training=value.get("metadata", {}).get("training"))
            else:
                entry.update(training=value.get("training"), artifacts=value.get("artifacts", []),
                             family_manifest_sha256=value.get("family_manifest_sha256"))
            entries.append(entry)
        except (OSError, ValueError, TypeError, AttributeError) as exc:
            errors.append({"path": str(path), "error": str(exc)})
    return {"schema_version": SCHEMA_VERSION, "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "roots": [str(p.resolve()) for p in roots], "entries": entries, "errors": errors,
            "scope": "Existing artifacts only; no model loading, simulation or reserved outcome generation."}


def write_catalog(catalog: dict, output: Path, html_output: Path | None = None) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(catalog, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    if html_output is None:
        return
    rows = []
    for entry in catalog["entries"]:
        fields = [entry["kind"], Path(entry["path"]).name, entry.get("binding", "—"),
                  entry.get("input_mode", "—"), entry.get("action_mode", "—"), entry.get("fleet_size", "—"),
                  entry.get("seed", "—"), entry["sha256"][:12]]
        cells = "".join(f"<td>{html.escape(str(v))}</td>" for v in fields)
        details = html.escape(json.dumps(entry, indent=2, ensure_ascii=False))
        rows.append(f"<tr>{cells}<td><details><summary>View record</summary><pre>{details}</pre></details></td></tr>")
    html_output.parent.mkdir(parents=True, exist_ok=True)
    html_output.write_text('''<!doctype html><html lang="en"><meta charset="utf-8"><title>Pandion · Experiments</title>
<style>body{font:15px system-ui;margin:32px;background:#101923;color:#e7eef5}input{padding:12px;width:50%;margin-bottom:24px}table{border-collapse:collapse;width:100%}th,td{text-align:left;padding:10px;border-bottom:1px solid #344455;vertical-align:top}pre{max-width:55vw;max-height:500px;overflow:auto;white-space:pre-wrap}summary{cursor:pointer}th{color:#9ac9ed}</style>
<h1>Pandion · Experiment catalog</h1><p>Existing artifacts, configuration, results and provenance.</p>
<input id="search" aria-label="Search artifacts" placeholder="Search incident, file, model, hash…">
<table><thead><tr><th>Type</th><th>File</th><th>Binding</th><th>Inputs</th><th>Actions</th><th>Fleet</th><th>Seed</th><th>Hash</th><th>Details</th></tr></thead><tbody>''' + "".join(rows) + '''</tbody></table>
<script>document.querySelector('#search').addEventListener('input',e=>{const q=e.target.value.toLowerCase();document.querySelectorAll('tbody tr').forEach(r=>r.hidden=!r.textContent.toLowerCase().includes(q));});</script></html>''')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--html", type=Path)
    args = parser.parse_args()
    catalog = scan_catalog(args.roots)
    write_catalog(catalog, args.output, args.html)
    print(f"Indexed {len(catalog['entries'])} artifacts; {len(catalog['errors'])} unreadable files")


if __name__ == "__main__":
    main()
