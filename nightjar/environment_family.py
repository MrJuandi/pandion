"""Attach aligned IGN terrain and optional EMSR578 station forcing to a family.

This derives a new scenario family from the verified four-incident geographic
family. DEM input is acquired separately; this module never downloads or fills
terrain values.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

from .identity import IDENTITY_VERSION, without_generation_times
from .cems_catalog import ACTIVATIONS
from .geographic_family import canonical_hash
from .geographic_ppo import load_geographic_split
from .pilot_terrain import _parse_ascii

SCHEMA_VERSION = "nightjar-geographic-environment-family-0.2"
SCENARIO_SCHEMA_VERSION = "nightjar-geographic-scenario-0.2"
TERRAIN_FILES = {
    "EMSR578": "emsr578.asc",
    "EMSR590": "emsr590.asc",
    "EMSR592": "emsr592.asc",
    "EMSR600": "emsr600.asc",
}
CRS_BY_ACTIVATION = {key: info["epsg"] for key, info in ACTIVATIONS.items()}


def terrain_filename(activation_id: str) -> str:
    return TERRAIN_FILES.get(activation_id, activation_id.lower().replace("_", "-") + ".asc")
ALIGNMENT_TOLERANCE_M = 0.1
WCS_BASE = "https://servicios.idee.es/wcs-inspire/mdt"


def _validate_grid(scenario: dict[str, Any], activation_id: str) -> tuple[int, int, float, float, float, str]:
    expected_crs = CRS_BY_ACTIVATION[activation_id]
    crs_metadata = scenario.get("crs", {})
    # Geographic-family 0.1 records the full authority/code pair. Accept the
    # short spelling too for small external fixtures, but require the same
    # explicit EPSG identifier either way.
    crs = crs_metadata.get("code", crs_metadata.get("epsg"))
    if crs != expected_crs:
        raise ValueError(f"{activation_id} scenario CRS must be {expected_crs}")
    grid = scenario.get("grid", {})
    try:
        rows, cols = int(grid["rows"]), int(grid["cols"])
        cellsize = float(grid["cell_size_m"])
        x0 = float(grid["origin_utm_easting_m"])
        y0 = float(grid["origin_utm_northing_m"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{activation_id} scenario has invalid grid metadata") from exc
    if rows <= 0 or cols <= 0 or any(isinstance(grid.get(key), bool) for key in ("rows", "cols")):
        raise ValueError(f"{activation_id} scenario dimensions must be positive integers")
    if rows != grid["rows"] or cols != grid["cols"]:
        raise ValueError(f"{activation_id} scenario dimensions must be integers")
    if not all(math.isfinite(v) for v in (cellsize, x0, y0)) or cellsize <= 0:
        raise ValueError(f"{activation_id} scenario grid values must be finite and cell size positive")
    return rows, cols, cellsize, x0, y0, expected_crs


def _assert_family_grids(scenarios: list[dict[str, Any]], activation_id: str) -> tuple[int, int, float, float, float, str]:
    if not scenarios:
        raise ValueError(f"No scenarios found for {activation_id}")
    reference = _validate_grid(scenarios[0], activation_id)
    for scenario in scenarios[1:]:
        candidate = _validate_grid(scenario, activation_id)
        if candidate[:2] != reference[:2] or any(
            abs(candidate[index] - reference[index]) > ALIGNMENT_TOLERANCE_M
            for index in (2, 3, 4)
        ) or candidate[5] != reference[5]:
            raise ValueError(f"{activation_id} variant grid metadata disagree; terrain crop cannot be reused")
    return reference


def _dem_url(activation_id: str, grid: tuple[int, int, float, float, float, str]) -> str:
    rows, cols, cellsize, x0, y0, crs = grid
    bbox = ",".join(f"{v:.1f}" for v in (x0, y0, x0 + cols * cellsize, y0 + rows * cellsize))
    params = [("service", "WCS"), ("version", "1.0.0"), ("request", "GetCoverage"),
              ("coverage", "Elevacion25830_25"), ("crs", crs), ("response_crs", crs),
              ("bbox", bbox), ("width", str(cols)), ("height", str(rows)), ("format", "ArcGrid")]
    return WCS_BASE + "?" + urlencode(params)


def _default_source(path: Path, grid: tuple[int, int, float, float, float, str], activation_id: str,
                    header: dict[str, Any]) -> dict[str, Any]:
    from hashlib import sha256

    return {
        "raw_sha256": sha256(path.read_bytes()).hexdigest(),
        "url": _dem_url(activation_id, grid),
        "provider": "IGN/CNIG",
        "dataset": "Elevacion25830_25",
        "native_resolution_m": 25,
        "vertical_reference": "orthometric",
        "acquisition_epoch": None,
        "source_footprint": "unknown" if "nodata_value" not in header else "nodata_encoded",
        "crs": grid[5],
        "crs_assertion": "caller_asserted_from_source_request; ESRI ASCII has no CRS field",
        "resampling": "nearest-neighbor",
    }


def _read_dem(path: Path, grid: tuple[int, int, float, float, float, str], activation_id: str,
              source_metadata: dict[str, Any] | None = None) -> tuple[list[list[float]], dict[str, Any]]:
    rows, cols, cellsize, x0, y0, crs = grid
    header, values = _parse_ascii(path)
    if (header["nrows"], header["ncols"]) != (rows, cols):
        raise ValueError(f"{path}: terrain dimensions do not match {activation_id} tactical grid")
    for actual, expected, field in ((float(header["cellsize"]), cellsize, "cellsize"),
                                    (float(header["xllcorner"]), x0, "x origin"),
                                    (float(header["yllcorner"]), y0, "y origin")):
        if abs(actual - expected) > ALIGNMENT_TOLERANCE_M:
            raise ValueError(f"{path}: terrain {field} does not align with {activation_id} tactical grid")
    # _parse_ascii returns south-to-north order. Nulls include NODATA and must
    # fail here: simulator elevation has no missing-cell substitution rule.
    if len(values) != rows or any(len(row) != cols for row in values):
        raise ValueError(f"{path}: terrain array dimensions do not match {activation_id} tactical grid")
    elevation: list[list[float]] = []
    for row in values:
        if any(value is None or not math.isfinite(float(value)) for value in row):
            raise ValueError(f"{path}: missing or nonfinite elevation; refusing to substitute terrain")
        elevation.append([float(value) for value in row])
    from hashlib import sha256

    raw_sha256 = sha256(path.read_bytes()).hexdigest()
    if source_metadata is None:
        source = _default_source(path, grid, activation_id, header)
    else:
        source = deepcopy(source_metadata)
        declared_hash = source.get("raw_sha256")
        if declared_hash != raw_sha256:
            raise ValueError(f"{path}: raw_sha256 does not match terrain sources manifest")
        if source.get("crs") != crs:
            raise ValueError(f"{path}: source CRS does not match scenario CRS {crs}")
        source.setdefault("crs_assertion", "caller_asserted_from_source_request; ESRI ASCII has no CRS field")
        source.setdefault("source_footprint", "unknown" if "nodata_value" not in header else "nodata_encoded")
        source["raw_sha256"] = raw_sha256
    return elevation, source


def _scenario_environment(scenario: dict[str, Any], terrain: dict[str, Any], *,
                          audit: dict[str, Any] | None, pilot_scenario: dict[str, Any] | None,
                          station_wind_factor: float, moisture_mode: str) -> dict[str, Any]:
    activation = scenario["activation_id"]
    if activation == "EMSR578" and scenario["split"] == "development" and audit is not None:
        # Imported on use because the weather integration is maintained in its
        # own module and may be added alongside this family builder.
        from .environment import build_station_environment

        if pilot_scenario is None:
            raise ValueError("pilot_scenario is required when weather audit is supplied")
        from .pilot_terrain import scenario_sha256

        if audit.get("scenario_sha256") != scenario_sha256(pilot_scenario):
            raise ValueError("weather audit scenario_sha256 does not match --pilot-scenario")
        payload = build_station_environment(scenario, audit,
                                            station_wind_factor=station_wind_factor,
                                            moisture_mode=moisture_mode)
        if not isinstance(payload, dict):
            raise ValueError("build_station_environment must return an environment object")
        environment = deepcopy(payload)
        environment["terrain"] = deepcopy(terrain)
        return environment
    return {
        "schema_version": "nightjar-environment-0.1",
        "terrain": deepcopy(terrain),
        "weather_mode": "constant_synthetic",
        "forcing": [],
    }


def _enrich_scenario(scenario: dict[str, Any], elevation: list[list[float]], source: dict[str, Any], *,
                     audit: dict[str, Any] | None, pilot_scenario: dict[str, Any] | None,
                     station_wind_factor: float, moisture_mode: str) -> dict[str, Any]:
    result = deepcopy(scenario)
    parent_hash = result.get("scenario_hash")
    if not isinstance(parent_hash, str) or len(parent_hash) != 64:
        raise ValueError(f"{result.get('scenario_id')}: missing original scenario hash")
    result["schema_version"] = SCENARIO_SCHEMA_VERSION
    result["fire"]["elevation_m"] = deepcopy(elevation)
    assumptions = result.setdefault("provenance", {}).setdefault("assumptions", {})
    assumptions["topography"] = "sourced_dem"
    if audit is not None and result.get("activation_id") == "EMSR578" and result.get("split") == "development":
        assumptions["wind"] = "station_2m_proxy_time_varying"
        assumptions["moisture"] = "modeled_emc_lag" if moisture_mode == "emc_lag" else "fixed"
    provenance = result["provenance"]
    provenance["environment_parent_scenario_hash"] = parent_hash
    terrain = {
        "source": deepcopy(source),
        "grid": {"rows": result["grid"]["rows"], "cols": result["grid"]["cols"],
                 "cell_size_m": result["grid"]["cell_size_m"],
                 "origin_utm_easting_m": result["grid"]["origin_utm_easting_m"],
                 "origin_utm_northing_m": result["grid"]["origin_utm_northing_m"],
                 "row_order": "south_to_north", "column_order": "west_to_east"},
        "units": "m",
    }
    result["environment"] = _scenario_environment(
        result, terrain, audit=audit, pilot_scenario=pilot_scenario,
        station_wind_factor=station_wind_factor, moisture_mode=moisture_mode,
    )
    result = without_generation_times(result)
    result["provenance"]["identity_version"] = IDENTITY_VERSION
    result.pop("scenario_hash", None)
    result["scenario_hash"] = canonical_hash(result)
    return result


def _summarize_sources(sources: dict[str, dict[str, Any]]) -> dict[str, Any]:
    def common(key: str) -> Any:
        values = {source.get(key) for source in sources.values()}
        return next(iter(values)) if len(values) == 1 else "mixed"

    resolutions = sorted({source.get("native_resolution_m") for source in sources.values()
                          if source.get("native_resolution_m") is not None})
    return {
        "terrain_provider": common("provider"),
        "terrain_dataset": common("dataset"),
        "native_resolution_m": resolutions[0] if len(resolutions) == 1 else None,
        "native_resolutions_m": resolutions,
        "vertical_reference": common("vertical_reference"),
        "acquisition_epoch": common("acquisition_epoch"),
        "source_footprint": common("source_footprint"),
        "terrain_sources": deepcopy(sources),
    }


def build_environment_family(
    scenarios_dir: Path | str,
    terrain_dir: Path | str,
    output_dir: Path | str,
    weather_audit: Path | str | None = None,
    pilot_scenario: Path | str | None = None,
    station_wind_factor: float = 1.0,
    moisture_mode: str = "emc_lag",
    synthetic: bool = False,
) -> dict[str, Any]:
    """Build and save an enriched copy of all verified scenario splits."""
    scenarios_dir, terrain_dir, output_dir = map(Path, (scenarios_dir, terrain_dir, output_dir))
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to mix with existing data in {output_dir}; choose a new empty directory")
    if not math.isfinite(float(station_wind_factor)) or station_wind_factor <= 0:
        raise ValueError("station_wind_factor must be finite and positive")
    if moisture_mode not in {"emc_lag", "fixed"}:
        raise ValueError("moisture_mode must be 'emc_lag' or 'fixed'")
    if (weather_audit is None) != (pilot_scenario is None):
        raise ValueError("--weather-audit and --pilot-scenario must be supplied together")

    split_scenarios = {split: load_geographic_split(split, scenarios_dir, synthetic=synthetic)
                       for split in ("development", "validation", "reserved")}
    manifest = json.loads((scenarios_dir / "manifest.json").read_text(encoding="utf-8"))
    grouped: dict[str, list[dict[str, Any]]] = {activation: [] for activation in TERRAIN_FILES}
    for scenarios in split_scenarios.values():
        for scenario in scenarios:
            activation = scenario.get("activation_id")
            if activation not in CRS_BY_ACTIVATION:
                raise ValueError(f"Unsupported incident in input family: {activation!r}")
            grouped.setdefault(activation, []).append(scenario)

    grids = {activation: _assert_family_grids(grouped[activation], activation) for activation in grouped}
    sources_path = terrain_dir / "sources.json"
    source_manifest = None
    if sources_path.is_file():
        source_manifest = json.loads(sources_path.read_text(encoding="utf-8"))
        if not isinstance(source_manifest, dict):
            raise ValueError(f"Terrain sources manifest must map activation IDs to metadata: {sources_path}")
        missing_sources = set(grouped) - source_manifest.keys()
        if missing_sources:
            raise ValueError(f"Terrain sources manifest is missing: {', '.join(sorted(missing_sources))}")
    dem = {
        activation: _read_dem(
            terrain_dir / terrain_filename(activation), grids[activation], activation,
            source_manifest[activation] if source_manifest is not None else None,
        )
        for activation in grouped
    }
    audit = None
    pilot = None
    if weather_audit is not None:
        audit = json.loads(Path(weather_audit).read_text(encoding="utf-8"))
        pilot = json.loads(Path(pilot_scenario).read_text(encoding="utf-8"))
        from .pilot_terrain import scenario_sha256
        if audit.get("scenario_sha256") != scenario_sha256(pilot):
            raise ValueError("weather audit scenario_sha256 does not match --pilot-scenario")
        if audit.get("scenario_id") != pilot.get("scenario_id"):
            raise ValueError("weather audit scenario_id does not match --pilot-scenario")

    enriched: dict[str, list[dict[str, Any]]] = {}
    for split, scenarios in split_scenarios.items():
        enriched[split] = []
        for scenario in scenarios:
            elevation, source = dem[scenario["activation_id"]]
            enriched[split].append(_enrich_scenario(
                scenario, elevation, source, audit=audit, pilot_scenario=pilot,
                station_wind_factor=float(station_wind_factor), moisture_mode=moisture_mode,
            ))

    output_manifest = deepcopy(manifest)
    output_manifest["schema_version"] = SCHEMA_VERSION
    output_manifest["family_id"] = "geographic-environment-family-0.1"
    output_manifest["description"] = "Geographic scenarios with source-bound aligned terrain and optional EMSR578 station-derived forcing."
    output_manifest["created_at_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    output_manifest["environment"] = {
        **_summarize_sources({activation: dem[activation][1] for activation in grouped}),
        "weather_mode": "station_derived" if audit is not None else "constant_synthetic",
        "weather_audit_sha256": None,
        "station_wind_factor": float(station_wind_factor), "moisture_mode": moisture_mode,
    }
    if source_manifest is not None:
        from hashlib import sha256
        output_manifest["environment"]["terrain_sources_manifest_sha256"] = sha256(sources_path.read_bytes()).hexdigest()
    if audit is not None:
        from hashlib import sha256
        output_manifest["environment"]["weather_audit_sha256"] = sha256(Path(weather_audit).read_bytes()).hexdigest()
    output_manifest["scenarios"] = {
        split: [{"scenario_id": item["scenario_id"], "hash": item["scenario_hash"],
                 "source_kind": item["source_kind"], "activation_id": item["activation_id"],
                 "source_zip_sha256": item["provenance"].get("source_zip_sha256")}
                for item in scenarios]
        for split, scenarios in enriched.items()
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(json.dumps(output_manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for split, scenarios in enriched.items():
        split_dir = output_dir / split
        split_dir.mkdir()
        for scenario in scenarios:
            (split_dir / f"{scenario['scenario_id']}.json").write_text(
                json.dumps(scenario, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return {"manifest": output_manifest, "scenarios": enriched}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenarios-dir", type=Path, required=True)
    parser.add_argument("--terrain-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--weather-audit", type=Path)
    parser.add_argument("--pilot-scenario", type=Path)
    parser.add_argument("--station-wind-factor", type=float, default=1.0)
    parser.add_argument("--moisture-mode", choices=("emc_lag", "fixed"), default="emc_lag")
    parser.add_argument("--synthetic", action="store_true", help="accept an explicitly generated synthetic input family")
    args = parser.parse_args()
    try:
        result = build_environment_family(
            args.scenarios_dir, args.terrain_dir, args.output_dir,
            weather_audit=args.weather_audit, pilot_scenario=args.pilot_scenario,
            station_wind_factor=args.station_wind_factor, moisture_mode=args.moisture_mode,
            synthetic=args.synthetic,
        )
    except (ValueError, FileNotFoundError, FileExistsError, OSError) as exc:
        parser.error(str(exc))
    print(f"Wrote {result['manifest']['counts']['total']} enriched scenarios to {args.output_dir}")


if __name__ == "__main__":
    main()
