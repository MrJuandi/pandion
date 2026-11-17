"""Build an offline research workspace from a verified geographic family.

Only development episodes are run. Validation/reserved entries are static maps.
Optional pilot terrain is a display layer and never changes policy inputs/physics.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path

from .geographic_ppo import load_geographic_split
from .geographic_integrated import run_geographic_integrated
from .pilot_terrain import load_terrain
from .viewer import write_viewer_html

SCHEMA_VERSION = "nightjar-workspace-0.1"


def file_hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def terrain_crop(scenario: dict, terrain: dict) -> list | None:
    """Align a display DEM by metric origin; never borrow relief from another CRS."""
    if scenario["activation_id"] != "EMSR578":
        return None
    grid = scenario["grid"]
    tg = terrain["grid"]
    if scenario["crs"]["code"] != terrain["spatial_reference"]["crs"]:
        raise ValueError("terrain and scenario CRS differ")
    size = grid["cell_size_m"]
    if size != tg["cell_size_m"]:
        raise ValueError("terrain and scenario resolution differ")
    offsets = [(grid["origin_utm_easting_m"] - tg["lower_left_origin_m"]["x"]) / size,
               (grid["origin_utm_northing_m"] - tg["lower_left_origin_m"]["y"]) / size]
    # Pilot origin is rounded to 0.1 m; family crop origin to 0.01 m.
    if any(abs(v - round(v)) * size > 0.1 for v in offsets):
        raise ValueError("terrain and scenario cell boundaries do not align within 0.1 m")
    c0, r0 = [round(v) for v in offsets]
    rows, cols = grid["rows"], grid["cols"]
    if r0 < 0 or c0 < 0 or r0 + rows > tg["rows"] or c0 + cols > tg["cols"]:
        raise ValueError("scenario crop is outside terrain coverage")
    return [row[c0:c0 + cols] for row in terrain["elevation_m"][r0:r0 + rows]]


def decorate(replay: dict, scenario: dict, terrain: dict | None = None, training: dict | None = None) -> dict:
    out = deepcopy(replay)
    p = scenario["provenance"]
    out.update({"name": scenario["name"], "source_kind": scenario["source_kind"],
                "split": scenario["split"], "seed": scenario["seed"],
                "scenario_hash": replay.get("scenario_hash", scenario["scenario_hash"]), "provenance": deepcopy(p)})
    out["metadata"] = {
        "name": scenario["name"], "source_kind": scenario["source_kind"],
        "split": scenario["split"], "seed": scenario["seed"],
        "source": p.get("source", ""), "assumptions": p.get("assumptions", {}),
        "scenario_hash": replay.get("scenario_hash", scenario["scenario_hash"]), "acquired_at_utc": p.get("image_acquired_at_utc"),
        "wind_to_m_s": scenario["fire"].get("wind_to_m_s"),
        "terrain": {"label": "Terreno plano · sin DEM en la simulación", "source": "", "display_only": False},
    }
    environment = scenario.get("environment")
    if environment:
        out["environment"] = deepcopy(environment)
        out["metadata"]["terrain"] = {
            "label": "Relieve real · cartografía sin episodio" if replay.get("horizon_s") == 0 else "Relieve real · usado en propagación y observaciones del PPO",
            "source": environment.get("terrain", {}).get("source", {}), "display_only": False,
        }
    if terrain is not None and not environment:
        cropped = terrain_crop(scenario, terrain)
        if cropped is not None:
            out["display_elevation_m"] = cropped
            out["metadata"]["terrain"] = {
                "label": "Relieve IGN/CNIG · capa visual; simulación y PPO sobre terreno plano",
                "source": terrain["source"], "display_only": True,
                "coverage": terrain["coverage"], "scenario_sha256": terrain["scenario_sha256"],
                "alignment_tolerance_m": 0.1,
            }
    if training:
        out["metadata"]["training"] = training
    return out


def static_map(scenario: dict) -> dict:
    fire, grid = scenario["fire"], scenario["grid"]
    return {
        "schema_version": "nightjar-static-map-0.1", "scenario_id": scenario["scenario_id"],
        "policy": "Cartografía · sin episodio", "horizon_s": 0,
        "affected_area_ha": None, "grid": grid, "crs": scenario["crs"],
        "units": {"time": "s", "coordinates": "m", "elevation": "m", "area": "ha"},
        "origin_utm": [grid["origin_utm_easting_m"], grid["origin_utm_northing_m"]],
        "arrival_time_s": [[0 if value else None for value in row] for row in fire["burn_mask"]],
        "fuel_grid": fire["fuel_grid"], "water_mask": fire["water_mask"],
        "elevation_m": deepcopy(fire.get("elevation_m")),
        "time_origin_utc": scenario.get("provenance", {}).get("image_acquired_at_utc"),
        "decisions": [], "drops": [], "aircraft_metrics": {},
    }


def pilot_map(pilot: dict, terrain: dict) -> dict:
    """Full observed initial landscape, with no simulated trajectory or result."""
    origin = terrain["grid"]["lower_left_origin_m"]
    return {
        "schema_version": "nightjar-static-map-0.1", "scenario_id": "emsr578-paisaje-completo",
        "policy": "Cartografía · sin episodio", "horizon_s": 0, "affected_area_ha": None,
        "seed": None, "grid": pilot["grid"], "crs": pilot["spatial_reference"]["crs"],
        "origin_m": [origin["x"], origin["y"]], "elevation_m": terrain["elevation_m"],
        "fuel_grid": pilot["fuel_model"], "water_mask": pilot["water_mask"],
        "arrival_time_s": [[0 if v else None for v in row] for row in pilot["initial_burn_mask"]],
        "decisions": [], "drops": [], "aircraft_metrics": {},
        "metadata": {"name": "Artesa de Segre · paisaje completo", "source": pilot["source"],
                     "source_kind": "copernicus_cems", "split": "development", "seed": None,
                     "scenario_hash": terrain["scenario_sha256"],
                     "acquired_at_utc": pilot["observation_timestamps"]["t0_initial_utc"],
                     "assumptions": {"fuel_crosswalk": "inferred", "episode": "none; initial observed burned footprint only"},
                     "terrain": {"label": "Relieve IGN/CNIG · cartografía histórica, sin episodio",
                                 "source": terrain["source"], "coverage": terrain["coverage"], "display_only": False}},
    }


def load_policy(checkpoint: Path, report_path: Path, scenarios: list[dict], **options):
    from .policy_artifact import load_policy_artifact
    return load_policy_artifact(checkpoint, report_path, scenarios, **options)


def policy_replay(model, scenario: dict) -> dict:
    from .geographic_ppo import rollout_geographic, with_fleet
    config = getattr(model, "_nightjar_inference", {})
    scenario = with_fleet([scenario], config.get("fleet_size", 1))[0]
    replays = []
    rollout_geographic(model, scenario, deterministic=True, inference_only=True, replay_out=replays,
                       **{key: config[key] for key in ("device", "input_mode", "imagery_dir",
                          "include_ros", "include_thermal", "skip_forced") if key in config})
    replay = replays[0]
    replay["policy"] = "PPO geográfico"
    return replay


def build_workspace(scenarios_dir: Path, output_dir: Path, *, episodes: int = 2,
                    terrain_path: Path | None = None, terrain_scenario: Path | None = None,
                    checkpoint: Path | None = None, training_report: Path | None = None,
                    synthetic: bool = False, imagery_dir: Path | None = None,
                    device: str = "cpu", skip_forced: bool | None = None) -> dict:
    if episodes < 0:
        raise ValueError("episodes must be nonnegative")
    if bool(terrain_path) != bool(terrain_scenario):
        raise ValueError("--terrain and --terrain-scenario must be supplied together")
    if bool(checkpoint) != bool(training_report):
        raise ValueError("--checkpoint and --training-report must be supplied together")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("output-dir must be new or empty")
    splits = {s: load_geographic_split(s, scenarios_dir, synthetic=synthetic) for s in ("development", "validation", "reserved")}
    pilot = json.loads(terrain_scenario.read_text()) if terrain_path else None
    terrain = load_terrain(terrain_path, pilot) if terrain_path else None
    model, training = load_policy(checkpoint, training_report, splits["development"], imagery_dir=imagery_dir,
                                  device=device, skip_forced=skip_forced) if checkpoint else (None, None)
    output_dir.mkdir(parents=True, exist_ok=True)
    replays = [pilot_map(pilot, terrain)] if terrain else []
    if replays:
        (output_dir / "emsr578-paisaje-completo.json").write_text(json.dumps(replays[0], ensure_ascii=False, allow_nan=False))
    for split, scenarios in splits.items():
        for index, scenario in enumerate(scenarios):
            episodes_here = []
            if split == "development" and index < episodes:
                if model is not None:
                    episodes_here.append(policy_replay(model, scenario))
                from .geographic_ppo import with_fleet
                baseline_scenario = with_fleet([scenario], training["inference"]["fleet_size"] if training else 1)[0]
                episodes_here.extend(run_geographic_integrated(baseline_scenario, policy=p) for p in
                                     ("heuristic", "containment_heuristic", "no_intervention"))
            else:
                episodes_here.append(static_map(scenario))
            for replay in episodes_here:
                decorated = decorate(replay, scenario, terrain, training if replay["policy"] == "PPO geográfico" else None)
                replays.append(decorated)
                name = f"{scenario['scenario_id']}-{len(replays):02d}.json"
                (output_dir / name).write_text(json.dumps(decorated, ensure_ascii=False, allow_nan=False))
            print(f"{scenario['scenario_id']}: {len(episodes_here)} vista(s)", flush=True)
    manifest = {"schema_version": SCHEMA_VERSION, "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "units": {"coordinates": "m", "time": "s", "area": "ha", "elevation": "m"},
                "seed": None, "coordinate_reference_systems": sorted({s["crs"]["code"] for v in splits.values() for s in v}),
                "family_manifest_sha256": file_hash(scenarios_dir / "manifest.json"),
                "terrain_sha256": file_hash(terrain_path) if terrain_path else None,
                "training": training, "scenario_count": sum(map(len, splits.values())) + bool(terrain),
                "replay_count": sum(r["horizon_s"] > 0 for r in replays),
                "static_map_count": sum(r["horizon_s"] == 0 for r in replays),
                "episode_splits": ["development"] if episodes else [],
                "scenario_hashes": {s["scenario_id"]: s["scenario_hash"] for v in splits.values() for s in v}}
    manifest["artifacts"] = [{"path": p.name, "sha256": file_hash(p)} for p in sorted(output_dir.glob("*.json"))]
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False))
    write_viewer_html(replays, output_dir / "index.html", title="Nightjar · Laboratorio de incendios",
                      imagery_dirs=[imagery_dir] if imagery_dir else None)
    from .experiment_catalog import scan_catalog, write_catalog
    roots = [output_dir] + ([checkpoint, training_report] if checkpoint else [])
    write_catalog(scan_catalog(roots), output_dir / "experiments.json", output_dir / "experiments.html")
    return manifest


def build_fold_workspace(scenarios_dir: Path, folds_dir: Path, output_dir: Path, *,
                         episodes: int = 2, synthetic: bool = False,
                         imagery_dir: Path | None = None, device: str = "cpu",
                         skip_forced: bool | None = None) -> dict:
    """Replay each fold's held-out development cases with that fold's weights."""
    from .geographic_ppo import with_fleet
    from .identity import incident_group
    if episodes < 1:
        raise ValueError("fold workspace requires at least one episode per fold")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("output-dir must be new or empty")
    scenarios = load_geographic_split("development", scenarios_dir, synthetic=synthetic)
    by_id = {s["scenario_id"]: s for s in scenarios}
    plans = []
    for report_path in sorted(folds_dir.glob("*.json")):
        report = json.loads(report_path.read_text())
        if report.get("schema_version") != "nightjar-geographic-ppo-selection-0.1" or not report.get("holdout"):
            continue
        checkpoint = report_path.with_suffix(".pt")
        if not checkpoint.is_file():
            raise ValueError(f"missing fold checkpoint: {checkpoint}")
        fold = report["holdout_activation"]
        rows = report["holdout"]["rows"]
        held = report.get("dataset", {}).get("splits", {}).get("holdout", [])
        hashes = {e["scenario_id"]: e["scenario_hash"] for e in held}
        selected = []
        for row in rows[:episodes]:
            case = by_id.get(row["scenario_id"])
            if case is None or incident_group(case["activation_id"]) != incident_group(fold):
                raise ValueError(f"fold {fold}: held-out scenario is not in the development family")
            executed = with_fleet([case], report.get("fleet_size", 1))[0]
            if hashes.get(case["scenario_id"]) != executed["scenario_hash"]:
                raise ValueError(f"fold {fold}: held-out scenario hash mismatch")
            selected.append(case)
        if not selected:
            raise ValueError(f"fold {fold}: no held-out rows")
        plans.append((fold, checkpoint, report_path, selected))
    if not plans:
        raise ValueError("no completed fold reports found")
    output_dir.mkdir(parents=True, exist_ok=True)
    replays, training_runs = [], []
    for fold, checkpoint, report_path, selected in plans:
        model, training = load_policy(checkpoint, report_path, scenarios, imagery_dir=imagery_dir,
                                      device=device, skip_forced=skip_forced)
        training.update(label=f"PPO · {folds_dir.name} · fold {fold}",
                        checkpoint_name=checkpoint.name, fold=fold)
        training_runs.append(training)
        for case in selected:
            episodes_here = [policy_replay(model, case)]
            baseline = with_fleet([case], training["inference"]["fleet_size"])[0]
            episodes_here.extend(run_geographic_integrated(baseline, policy=p) for p in
                                 ("heuristic", "containment_heuristic", "no_intervention"))
            for replay in episodes_here:
                decorated = decorate(replay, case, training=training if replay["policy"] == "PPO geográfico" else None)
                replays.append(decorated)
                name = f"{case['scenario_id']}-{len(replays):02d}.json"
                (output_dir / name).write_text(json.dumps(decorated, ensure_ascii=False, allow_nan=False))
            print(f"{fold}: {case['scenario_id']} · {checkpoint.name}", flush=True)
        del model
    manifest = {"schema_version": SCHEMA_VERSION, "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "units": {"coordinates": "m", "time": "s", "area": "ha", "elevation": "m"},
                "family_manifest_sha256": file_hash(scenarios_dir / "manifest.json"),
                "mode": "held_out_folds", "training_runs": training_runs,
                "scenario_count": sum(len(p[3]) for p in plans), "replay_count": len(replays),
                "static_map_count": 0, "episode_splits": ["development"],
                "scenario_hashes": {r["scenario_id"]: r["scenario_hash"] for r in replays}}
    manifest["artifacts"] = [{"path": p.name, "sha256": file_hash(p)} for p in sorted(output_dir.glob("*.json"))]
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False))
    write_viewer_html(replays, output_dir / "index.html", title=f"Nightjar · {folds_dir.name}",
                      imagery_dirs=[imagery_dir] if imagery_dir else None)
    return manifest


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scenarios-dir", type=Path, default=Path("runs/geographic-family-real-v2"))
    p.add_argument("--output-dir", type=Path, default=Path("runs/workspace"))
    p.add_argument("--episodes", type=int, default=2, help="Number of development scenarios to replay; 0 builds only maps")
    p.add_argument("--terrain", type=Path)
    p.add_argument("--terrain-scenario", type=Path)
    p.add_argument("--folds-dir", type=Path, help="Completed fold reports and matching .pt checkpoints; replays held-out development cases per fold")
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--training-report", type=Path)
    p.add_argument("--imagery-dir", type=Path, help="Relocated imagery directory, verified against report when bound; "
                   "also embedded in the viewer as a display-only Sentinel-2 base layer for matching grids")
    p.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    p.add_argument("--skip-forced", action=argparse.BooleanOptionalAction, default=None,
                   help="Explicit recurrent-state behavior for legacy reports missing this setting")
    p.add_argument("--synthetic", action="store_true", help="Explicitly use a generated synthetic fixture family")
    args = p.parse_args()
    try:
        if args.folds_dir:
            if args.checkpoint or args.training_report or args.terrain or args.terrain_scenario:
                raise ValueError("--folds-dir cannot be combined with a single checkpoint or pilot terrain")
            manifest = build_fold_workspace(args.scenarios_dir, args.folds_dir, args.output_dir,
                episodes=args.episodes, synthetic=args.synthetic, imagery_dir=args.imagery_dir,
                device=args.device, skip_forced=args.skip_forced)
        else:
            manifest = build_workspace(args.scenarios_dir, args.output_dir, episodes=args.episodes,
                                       terrain_path=args.terrain, terrain_scenario=args.terrain_scenario,
                                       checkpoint=args.checkpoint, training_report=args.training_report, synthetic=args.synthetic, imagery_dir=args.imagery_dir,
                                       device=args.device, skip_forced=args.skip_forced)
    except (OSError, ValueError) as exc:
        p.error(str(exc))
    print(f"Workspace: {args.output_dir / 'index.html'} ({manifest['scenario_count']} scenarios)")


if __name__ == "__main__":
    main()
