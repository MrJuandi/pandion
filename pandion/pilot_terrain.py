"""Strict ESRI ASCII elevation ingestion for the EMSR578 pilot grid."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "nightjar-pilot-terrain-0.1"
SUPPORTED_CRS = "EPSG:25831"
ALIGNMENT_TOLERANCE_M = 1e-5


def scenario_sha256(scenario: dict[str, Any]) -> str:
    """Hash scenario JSON using sorted keys and compact separators."""
    encoded = json.dumps(scenario, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return sha256(encoded).hexdigest()


def _finite_number(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def _scenario_grid(scenario: dict[str, Any]) -> tuple[int, int, float, float, float]:
    spatial = scenario.get("spatial_reference", {})
    if spatial.get("crs") != SUPPORTED_CRS:
        raise ValueError(f"scenario CRS must be {SUPPORTED_CRS}")
    grid = scenario.get("grid", {})
    bounds = spatial.get("bounds_utm31n", {})
    try:
        rows, cols = int(grid["rows"]), int(grid["cols"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("scenario grid must include integer rows and cols") from exc
    if rows <= 0 or cols <= 0:
        raise ValueError("scenario dimensions must be positive")
    if rows != grid["rows"] or cols != grid["cols"] or isinstance(grid["rows"], bool) or isinstance(grid["cols"], bool):
        raise ValueError("scenario dimensions must be integers")
    cell_size = _finite_number(grid.get("cell_size_m"), "scenario cell_size_m")
    x0 = _finite_number(bounds.get("min_x"), "scenario bounds min_x")
    y0 = _finite_number(bounds.get("min_y"), "scenario bounds min_y")
    if cell_size <= 0:
        raise ValueError("scenario cell_size_m must be positive")
    return rows, cols, cell_size, x0, y0


def _parse_ascii(path: Path) -> tuple[dict[str, float | int], list[list[float | None]]]:
    lines = path.read_text(encoding="ascii").splitlines()
    header: dict[str, float | int] = {}
    data_start = 0
    allowed = {"ncols", "nrows", "xllcorner", "yllcorner", "cellsize", "nodata_value"}
    for index, raw_line in enumerate(lines):
        parts = raw_line.split()
        if not parts:
            raise ValueError("blank lines are not allowed in ESRI ASCII grid")
        key = parts[0].lower()
        if key not in allowed:
            try:
                float(parts[0])
            except ValueError as exc:
                raise ValueError(f"unknown ESRI ASCII header key: {key}") from exc
            data_start = index
            break
        if len(parts) != 2:
            raise ValueError(f"header line {index + 1} must contain one key and one value")
        if key in header:
            raise ValueError(f"duplicate ESRI ASCII header key: {key}")
        if key not in allowed:
            raise ValueError(f"unknown ESRI ASCII header key: {key}")
        try:
            number = float(parts[1])
        except ValueError as exc:
            raise ValueError(f"invalid value for ESRI ASCII header key {key}") from exc
        if not math.isfinite(number):
            raise ValueError(f"header value for {key} must be finite")
        if key in ("ncols", "nrows"):
            if not number.is_integer():
                raise ValueError(f"{key} must be an integer")
            header[key] = int(number)
        else:
            header[key] = number
        data_start = index + 1
    required = {"ncols", "nrows", "xllcorner", "yllcorner", "cellsize"}
    missing = required - header.keys()
    if missing:
        raise ValueError(f"missing ESRI ASCII header keys: {', '.join(sorted(missing))}")
    if set(header) - required - {"nodata_value"}:
        raise ValueError("unknown ESRI ASCII header key")
    ncols, nrows = int(header["ncols"]), int(header["nrows"])
    if ncols <= 0 or nrows <= 0:
        raise ValueError("ESRI ASCII dimensions must be positive")
    if float(header["cellsize"]) <= 0:
        raise ValueError("ESRI ASCII cellsize must be positive")

    raster_lines = lines[data_start:]
    if len(raster_lines) != nrows:
        raise ValueError(f"raster has {len(raster_lines)} rows; expected {nrows}")
    rows_north_first: list[list[float | None]] = []
    nodata = header.get("nodata_value")
    for row_index, raw_line in enumerate(raster_lines):
        fields = raw_line.split()
        if len(fields) != ncols:
            raise ValueError(f"raster row {row_index + 1} has {len(fields)} values; expected {ncols}")
        row: list[float | None] = []
        for field in fields:
            try:
                value = float(field)
            except ValueError as exc:
                raise ValueError(f"invalid raster number: {field!r}") from exc
            if not math.isfinite(value):
                raise ValueError("raster values must be finite")
            row.append(None if nodata is not None and value == nodata else value)
        rows_north_first.append(row)
    header["ncols"] = ncols
    header["nrows"] = nrows
    return header, list(reversed(rows_north_first))


def _check_alignment(header: dict[str, float | int], scenario: dict[str, Any], crs: str) -> tuple[int, int, float, float, float]:
    if crs != SUPPORTED_CRS:
        raise ValueError(f"only caller-asserted {SUPPORTED_CRS} is supported")
    rows, cols, cell_size, x0, y0 = _scenario_grid(scenario)
    if (int(header["nrows"]), int(header["ncols"])) != (rows, cols):
        raise ValueError("terrain dimensions do not match scenario grid")
    checks = ((float(header["cellsize"]), cell_size, "cellsize"),
              (float(header["xllcorner"]), x0, "xllcorner"),
              (float(header["yllcorner"]), y0, "yllcorner"))
    for actual, expected, name in checks:
        if abs(actual - expected) > ALIGNMENT_TOLERANCE_M:
            raise ValueError(f"terrain {name} does not align with scenario grid")
    return rows, cols, cell_size, x0, y0


def build_terrain(
    scenario: dict[str, Any],
    ascii_grid_path: Path,
    *,
    crs: str,
    source_url: str | None = None,
    provider: str | None = None,
    dataset: str | None = None,
    native_resolution_m: float | None = None,
    vertical_reference: str | None = None,
    acquisition_period: str | None = None,
    resampling: str | None = None,
) -> dict[str, Any]:
    """Build a terrain artifact aligned exactly to the pilot scenario grid."""
    path = Path(ascii_grid_path)
    header, elevation = _parse_ascii(path)
    rows, cols, cell_size, x0, y0 = _check_alignment(header, scenario, crs)
    if native_resolution_m is not None:
        native_resolution_m = _finite_number(native_resolution_m, "native_resolution_m")
        if native_resolution_m <= 0:
            raise ValueError("native_resolution_m must be positive")
    valid = [[value is not None for value in row] for row in elevation]
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "seed": None,
        "scenario_sha256": scenario_sha256(scenario),
        "units": {"horizontal": "m", "elevation": "m"},
        "spatial_reference": {
            "crs": SUPPORTED_CRS,
            "crs_assertion": "caller_asserted_from_source_request; ESRI ASCII has no CRS field",
        },
        "grid": {
            "rows": rows,
            "cols": cols,
            "cell_size_m": cell_size,
            "lower_left_origin_m": {"x": x0, "y": y0},
            "row_order": "south_to_north",
            "column_order": "west_to_east",
        },
        "elevation_m": elevation,
        "valid_mask": valid,
        "coverage": {
            "status": "nodata_encoded" if "nodata_value" in header else "source_footprint_unknown",
            "nodata_value": header.get("nodata_value"),
        },
        "source": {
            "raw_sha256": sha256(path.read_bytes()).hexdigest(),
            "url": source_url,
            "provider": provider,
            "dataset": dataset,
            "native_resolution_m": native_resolution_m,
            "vertical_reference": vertical_reference,
            "acquisition_period": acquisition_period,
            "resampling": resampling,
            "unknown_fields": [
                name for name, value in (
                    ("url", source_url), ("provider", provider), ("dataset", dataset),
                    ("native_resolution_m", native_resolution_m),
                    ("vertical_reference", vertical_reference),
                    ("acquisition_period", acquisition_period), ("resampling", resampling),
                ) if value is None
            ],
        },
    }


def load_terrain(artifact_path: Path, scenario: dict[str, Any]) -> dict[str, Any]:
    """Load and validate a terrain artifact against its pilot scenario."""
    artifact = json.loads(Path(artifact_path).read_text(encoding="utf-8"))
    if artifact.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"terrain schema_version must be {SCHEMA_VERSION}")
    if artifact.get("scenario_sha256") != scenario_sha256(scenario):
        raise ValueError("terrain scenario_sha256 does not match scenario")
    timestamp = artifact.get("created_at")
    try:
        parsed_timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except (AttributeError, ValueError) as exc:
        raise ValueError("terrain created_at must be a UTC timestamp") from exc
    if parsed_timestamp.tzinfo is None or parsed_timestamp.utcoffset() != timezone.utc.utcoffset(parsed_timestamp):
        raise ValueError("terrain created_at must be UTC")
    if artifact.get("seed", "not-null") is not None:
        raise ValueError("terrain seed must be null for deterministic source ingestion")
    if artifact.get("units") != {"horizontal": "m", "elevation": "m"}:
        raise ValueError("terrain units are missing or unsupported")
    spatial = artifact.get("spatial_reference", {})
    if spatial.get("crs") != SUPPORTED_CRS:
        raise ValueError(f"terrain CRS must be {SUPPORTED_CRS}")
    source = artifact.get("source", {})
    digest = source.get("raw_sha256", "")
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("terrain source raw_sha256 must be a SHA-256 digest")
    coverage = artifact.get("coverage", {})
    if coverage.get("status") not in ("nodata_encoded", "source_footprint_unknown"):
        raise ValueError("terrain coverage status must be explicit")
    grid = artifact.get("grid", {})
    rows, cols, cell_size, x0, y0 = _scenario_grid(scenario)
    if grid.get("rows") != rows or grid.get("cols") != cols:
        raise ValueError("terrain dimensions do not match scenario grid")
    if grid.get("row_order") != "south_to_north" or grid.get("column_order") != "west_to_east":
        raise ValueError("terrain grid orientation is unsupported")
    origin = grid.get("lower_left_origin_m", {})
    for actual, expected, name in ((grid.get("cell_size_m"), cell_size, "cell_size_m"),
                                   (origin.get("x"), x0, "origin x"),
                                   (origin.get("y"), y0, "origin y")):
        if abs(_finite_number(actual, f"terrain {name}") - expected) > ALIGNMENT_TOLERANCE_M:
            raise ValueError(f"terrain {name} does not match scenario grid")
    elevations = artifact.get("elevation_m")
    mask = artifact.get("valid_mask")
    if not isinstance(elevations, list) or len(elevations) != rows or not isinstance(mask, list) or len(mask) != rows:
        raise ValueError("terrain elevation and validity arrays have incorrect row count")
    for r, (elevation_row, mask_row) in enumerate(zip(elevations, mask)):
        if not isinstance(elevation_row, list) or len(elevation_row) != cols or not isinstance(mask_row, list) or len(mask_row) != cols:
            raise ValueError(f"terrain row {r} has incorrect column count")
        for c, (value, valid) in enumerate(zip(elevation_row, mask_row)):
            if not isinstance(valid, bool):
                raise ValueError(f"valid_mask[{r}][{c}] must be boolean")
            if value is None:
                if valid:
                    raise ValueError(f"valid_mask[{r}][{c}] marks missing elevation valid")
            elif not math.isfinite(_finite_number(value, f"elevation_m[{r}][{c}]")):
                raise ValueError(f"elevation_m[{r}][{c}] must be finite or null")
            elif not valid:
                raise ValueError(f"valid_mask[{r}][{c}] disagrees with elevation")
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", type=Path, required=True)
    parser.add_argument("--ascii-grid", type=Path, required=True)
    parser.add_argument("--crs", required=True, help="Caller assertion; supported value: EPSG:25831")
    parser.add_argument("--source-url")
    parser.add_argument("--provider")
    parser.add_argument("--dataset")
    parser.add_argument("--native-resolution-m", type=float)
    parser.add_argument("--vertical-reference")
    parser.add_argument("--acquisition-period")
    parser.add_argument("--resampling")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    scenario = json.loads(args.scenario.read_text(encoding="utf-8"))
    artifact = build_terrain(
        scenario, args.ascii_grid, crs=args.crs, source_url=args.source_url,
        provider=args.provider, dataset=args.dataset,
        native_resolution_m=args.native_resolution_m,
        vertical_reference=args.vertical_reference,
        acquisition_period=args.acquisition_period, resampling=args.resampling,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
