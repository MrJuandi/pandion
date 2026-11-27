"""Verified checkpoint/report binding and shared geographic inference loading."""
from __future__ import annotations

from hashlib import sha256
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
from typing import Any

from .identity import content_hash

SCHEMA_VERSION = "nightjar-policy-artifact-0.1"
REPORT_SCHEMAS = {f"nightjar-geographic-ppo-{kind}-0.1" for kind in ("smoke", "train", "selection", "eval")}


def file_hash(path: Path | str) -> str:
    return sha256(Path(path).read_bytes()).hexdigest()


def execution_provenance() -> dict:
    root = Path(__file__).resolve().parent
    source = {p.name: file_hash(p) for p in sorted(root.glob("*.py"))}
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True,
                                         stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    packages = {}
    for name in ("torch", "numpy", "pyretechnics"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {"source_sha256": content_hash(source), "source_files": source, "commit": commit,
            "python": platform.python_version(), "packages": packages}


def inference_config(report: dict) -> dict:
    recurrent = report.get("recurrent")
    if not isinstance(recurrent, bool):
        raise ValueError("training report must specify recurrent configuration")
    input_mode = report.get("input_mode", "legacy")
    action_mode = report.get("action_mode", "tactical")
    fleet_size = report.get("fleet_size", 1)
    if input_mode not in ("legacy", "image", "cells") or action_mode not in ("tactical", "spatial", "parametric", "frontier"):
        raise ValueError("unsupported policy input/action configuration")
    if isinstance(fleet_size, bool) or not isinstance(fleet_size, int) or fleet_size < 1:
        raise ValueError("invalid policy fleet_size")
    for key in ("include_ros", "include_thermal", "slot_features", "skip_forced"):
        if report.get(key) is not None and not isinstance(report[key], bool):
            raise ValueError(f"{key} must be boolean")
    inference = {"input_mode": input_mode, "action_mode": action_mode, "recurrent": recurrent,
                 "fleet_size": fleet_size, "include_ros": report.get("include_ros", False),
                 "include_thermal": report.get("include_thermal", False),
                 "slot_features": report.get("slot_features"),
                 "skip_forced": report.get("skip_forced", report.get("ppo", {}).get("skip_forced")),
                 "imagery_dir": report.get("imagery_dir"),
                 "drop_distances": list(report.get("drop_distances") or [0])}
    # Other optional keys appear only when enabled, preserving older bindings.
    if report.get("action_features"):
        inference["action_features"] = True
    if report.get("action_skip"):
        inference["action_skip"] = True
    if report.get("measured_intensity"):
        inference["measured_intensity"] = True
    for key in ("learned_speed", "learned_hold", "learned_burn"):
        if report.get(key):
            inference[key] = report[key]
    return inference


def write_report(report: dict, report_path: Path, checkpoint: Path | None = None, *, attach_sidecar: bool = True) -> None:
    """Bind newly written reports to weights; keep checkpoint tensor format intact."""
    report["execution"] = execution_provenance()
    imagery = report.get("imagery_dir")
    if imagery:
        sources = Path(imagery) / "sources.json"
        if sources.is_file():
            report.setdefault("dataset", {})["imagery_content_sha256"] = content_hash(json.loads(sources.read_text()))
    if checkpoint is not None and checkpoint.is_file():
        report["artifact"] = {"schema_version": SCHEMA_VERSION,
                              "checkpoint_sha256": file_hash(checkpoint),
                              "inference": inference_config(report)}
        best_checkpoint = Path(f"{checkpoint}.best")
        if best_checkpoint.is_file():
            report["artifact"]["best_checkpoint_sha256"] = file_hash(best_checkpoint)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if "artifact" in report and attach_sidecar:
        sidecar = {**report["artifact"], "checkpoint": str(checkpoint.resolve()),
                   "report": str(report_path.resolve()), "report_sha256": file_hash(report_path)}
        Path(str(checkpoint) + ".artifact.json").write_text(json.dumps(sidecar, indent=2) + "\n")


def verify_binding(checkpoint: Path, report_path: Path, report: dict) -> bool:
    """Validate weights and any exact-report sidecar without importing torch."""
    artifact = report.get("artifact")
    digest = file_hash(checkpoint)
    if artifact:
        bound = artifact.get("inference")
        # Older bindings implied a zero-cell distance when the field was absent.
        if (artifact.get("schema_version") != SCHEMA_VERSION
                or artifact.get("checkpoint_sha256") != digest
                or not isinstance(bound, dict)
                or {"drop_distances": [0], **bound} != inference_config(report)):
            raise ValueError("checkpoint/report binding mismatch")
    sidecar_path = Path(str(checkpoint) + ".artifact.json")
    if sidecar_path.exists():
        sidecar = json.loads(sidecar_path.read_text())
        if sidecar.get("checkpoint_sha256") != digest:
            raise ValueError("checkpoint sidecar/report binding mismatch")
        # Evaluation reports are separate consumers of the training artifact;
        # writing/reading them must not replace its original report binding.
        if report.get("schema_version") != "nightjar-geographic-ppo-eval-0.1" and sidecar.get("report_sha256") != file_hash(report_path):
            raise ValueError("checkpoint sidecar/report binding mismatch")
    return bool(artifact)


def load_policy_artifact(checkpoint: Path, report_path: Path, scenarios: list[dict], *,
                         imagery_dir: Path | None = None, device: str = "cpu",
                         skip_forced: bool | None = None) -> tuple[Any, dict]:
    """Validate dataset/report identity before constructing a model or reading weights."""
    report = json.loads(report_path.read_text())
    if report.get("schema_version") not in REPORT_SCHEMAS:
        raise ValueError("workspace requires a geographic report with dataset provenance")
    training_dataset = report.get("dataset", {})
    if report["schema_version"] == "nightjar-geographic-ppo-eval-0.1":
        training_dataset = report.get("training_artifact", {}).get("dataset", training_dataset)
    entries = training_dataset.get("splits", {}).get("development", [])
    config = inference_config(report)
    from .geographic_ppo import with_fleet, build_policy_model
    available = {s["scenario_id"]: s["scenario_hash"] for s in with_fleet(scenarios, config["fleet_size"])}
    if not entries or len({e.get("scenario_id") for e in entries}) != len(entries) or any(
            available.get(e.get("scenario_id")) != e.get("scenario_hash") for e in entries):
        raise ValueError("training report does not match the verified development family")
    digest = file_hash(checkpoint)
    artifact = report.get("artifact")
    verify_binding(checkpoint, report_path, report)
    import torch
    state = torch.load(checkpoint, weights_only=True, map_location=device)
    warnings = []
    if not artifact:
        warnings.append("Legacy weights: checkpoint/report association is caller supplied, not historically verified.")
    if config["slot_features"] is None:
        # Old reports omitted this flag; the parametric head shape distinguishes it.
        weight = state.get("slot_actor_head.0.weight")
        # Standard width=32, recurrent/nonrecurrent head=128: base input is 161.
        config["slot_features"] = bool(weight is not None and weight.shape[1] > 161)
        if weight is not None:
            warnings.append("slot_features reconstructed from legacy checkpoint architecture.")
    if skip_forced is not None:
        if artifact and skip_forced != config["skip_forced"]:
            raise ValueError("skip_forced override conflicts with bound inference configuration")
        config["skip_forced"] = skip_forced
    if config["skip_forced"] is None:
        config["skip_forced"] = False
        warnings.append("Legacy report omitted skip_forced; using False. Supply an explicit override if needed.")
    image_path = imagery_dir or (Path(config["imagery_dir"]) if config["imagery_dir"] else None)
    if config["input_mode"] == "image" and image_path is None:
        raise ValueError("image policy requires imagery_dir")
    if image_path is not None and config["input_mode"] == "image":
        manifest = image_path / "sources.json"
        expected = report.get("dataset", {}).get("imagery_content_sha256")
        if expected and content_hash(json.loads(manifest.read_text())) != expected:
            raise ValueError("imagery manifest content differs from training report")
    member_ids = {e["scenario_id"] for e in entries}
    training_scenario = next(s for s in scenarios if s["scenario_id"] in member_ids)
    scenario = with_fleet([training_scenario], config["fleet_size"])[0]
    model = build_policy_model(scenario, **{k: config[k] for k in
                               ("input_mode", "action_mode", "recurrent", "include_ros", "include_thermal", "slot_features")},
                               imagery_dir=image_path,
                               drop_distances=tuple(config["drop_distances"]),
                               action_features=bool(config.get("action_features", False)),
                               **({"action_skip": True} if config.get("action_skip") else {}),
                               measured_intensity=bool(config.get("measured_intensity", False)),
                               learned_speed=config.get("learned_speed"),
                               learned_hold=config.get("learned_hold"), learned_burn=config.get("learned_burn"))
    try:
        model.load_state_dict(state)
    except RuntimeError as exc:
        raise ValueError(f"checkpoint architecture does not match report: {exc}") from exc
    model.to(device).eval()
    model._pandion_inference = {**config, "imagery_dir": image_path, "device": device}
    if report.get("execution", {}).get("source_sha256") and report["execution"]["source_sha256"] != execution_provenance()["source_sha256"]:
        warnings.append("Current source differs from the report; this is inference under the current simulator version.")
    info = {"label": "Geographic PPO", "seed": report.get("selected_seed", report.get("seed")),
            "updates": report.get("updates"), "status": "Exploratory training",
            "checkpoint_sha256": digest, "report_sha256": file_hash(report_path),
            "checkpoint_report_binding": "verified" if artifact else "caller_supplied_legacy",
            "inference": {**config, "imagery_dir": str(image_path) if image_path else None},
            "dataset": training_dataset, "execution": report.get("execution"), "warnings": warnings}
    return model, info
