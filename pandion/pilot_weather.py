"""Audit local Meteocat XEMA measurements for the EMSR578 observation window.

This produces station observations and coverage evidence, not fire-engine forcing.
No wind-at-flame-height or vegetation moisture is inferred here.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "nightjar-pilot-weather-audit-0.1"
DATA_BASE = "https://analisi.transparenciacatalunya.cat/resource/"
VARIABLES = {
    "32": ("air_temperature", "°C"),
    "33": ("relative_humidity", "%"),
    "35": ("precipitation", "mm"),
    "46": ("wind_speed_at_2m", "m/s"),
    "47": ("wind_direction_at_2m", "°"),
}


def _utc(value: str) -> datetime:
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    # Meteocat documents all XEMA date values as UTC, including offsetless exports.
    return result.replace(tzinfo=timezone.utc) if result.tzinfo is None else result.astimezone(timezone.utc)


def _stamp(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


def audit_weather(
    scenario: dict[str, Any], stations_path: Path, variables_path: Path,
    measurements_path: Path, station_code: str = "X6",
) -> dict[str, Any]:
    paths = {"stations": stations_path, "variables": variables_path, "measurements": measurements_path}
    raw = {name: path.read_bytes() for name, path in paths.items()}
    data = {name: json.loads(content) for name, content in raw.items()}
    if any(not isinstance(value, list) for value in data.values()):
        raise ValueError("source exports must be JSON lists")
    stations = [s for s in data["stations"] if s["codi_estacio"] == station_code]
    if len(stations) != 1:
        raise ValueError("expected exactly one station metadata record")
    station = stations[0]
    longitude, latitude = float(station["longitud"]), float(station["latitud"])
    if not (isfinite(longitude) and isfinite(latitude) and -180 <= longitude <= 180 and -90 <= latitude <= 90):
        raise ValueError("invalid station coordinates")
    variable_metadata = {}
    for code, (_, unit) in VARIABLES.items():
        matches = [v for v in data["variables"] if v["codi_variable"] == code]
        if len(matches) != 1 or matches[0]["unitat"] != unit:
            raise ValueError(f"variable {code}: missing/duplicate metadata or unexpected unit")
        variable_metadata[code] = matches[0]
    times = scenario["observation_timestamps"]
    t0, t1, t2 = [_utc(times[key]) for key in ("t0_initial_utc", "t1_monitoring_utc", "t2_monitoring_utc")]
    if not t0 < t1 < t2:
        raise ValueError("observation timestamps must increase")
    cadence = timedelta(minutes=30)
    first = t0.replace(minute=(t0.minute // 30) * 30, second=0, microsecond=0)
    slots = []
    current = first
    while current < t2:
        slots.append(current)
        current += cadence
    if _utc(station["data_inici"]) > first or (station.get("data_fi") and _utc(station["data_fi"]) < t2):
        raise ValueError("station metadata does not cover observation window")
    records: dict[str, dict[datetime, dict[str, Any]]] = {code: {} for code in VARIABLES}
    for row in data["measurements"]:
        if row["codi_estacio"] != station_code:
            raise ValueError("measurement export mixes station identities")
        code = row["codi_variable"]
        if code not in VARIABLES:
            continue
        instant = _utc(row["data_lectura"])
        if row["codi_base"] != "SH" or instant.minute not in (0, 30) or instant.second or instant.microsecond:
            raise ValueError("expected measurements on the XEMA half-hour grid")
        if instant in records[code]:
            raise ValueError("duplicate station/variable/timestamp measurement")
        value = float(row["valor_lectura"])
        if not isfinite(value):
            raise ValueError("nonfinite station measurement")
        records[code][instant] = {
            "interval_start_utc": _stamp(instant), "interval_end_utc": _stamp(instant + cadence),
            "value": value, "quality_code": row["codi_estat"],
            "usable": row["codi_estat"] == "V",
        }
    audited = {}
    for code, (name, unit) in VARIABLES.items():
        selected = [records[code][slot] for slot in slots if slot in records[code]]
        usable = [row for row in selected if row["usable"]]
        missing = [slot for slot in slots if slot not in records[code]]
        invalid = [slot for slot in slots if slot in records[code] and not records[code][slot]["usable"]]
        audited[name] = {
            "source_variable_code": code, "source_name": variable_metadata[code]["nom_variable"],
            "unit": unit, "expected_intervals": len(slots), "present_intervals": len(selected),
            "usable_intervals": len(usable), "missing_intervals_utc": [_stamp(s) for s in missing],
            "unusable_intervals_utc": [_stamp(s) for s in invalid],
            "min_valid": min((r["value"] for r in usable), default=None),
            "max_valid": max((r["value"] for r in usable), default=None), "records": selected,
        }
    return {
        "schema_version": SCHEMA_VERSION, "created_at_utc": _stamp(datetime.now(timezone.utc)),
        "seed": None, "deterministic": True, "scenario_id": scenario["scenario_id"],
        "scenario_sha256": sha256(json.dumps(scenario, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest(),
        "station": {
            "code": station_code, "name": station["nom_estacio"], "crs": "EPSG:4326",
            "longitude": longitude, "latitude": latitude, "altitude_m": float(station["altitud"]),
        },
        "units": {"time": "s", "station_altitude": "m", "coordinates": "degrees"},
        "window": {
            "start_utc": _stamp(t0), "end_utc": _stamp(t2), "reading_interval_s": 1800,
            "timestamp_semantics": "start of the following measurement interval, not instantaneous",
            "t0_to_t1_s": (t1-t0).total_seconds(), "t0_to_t2_s": (t2-t0).total_seconds(),
        },
        "sources": {
            "provider": "Servei Meteorologic de Catalunya (Meteocat)",
            "exports": {name: {"sha256": sha256(content).hexdigest(), "records": len(data[name])} for name, content in raw.items()},
            "stations_url": DATA_BASE + "yqwd-vj5e.json?$limit=1000",
            "variables_url": DATA_BASE + "4fb2-n3yi.json?$limit=1000",
            "measurement_dataset_url": DATA_BASE + "nzvn-apee.json",
            "measurement_filter": {"station_code": station_code, "time_field": "data_lectura"},
            "time_semantics_url": "https://apidocs.meteocat.gencat.cat/documentacio/dades-de-la-xema/",
        },
        "variables": audited,
        "complete_valid_coverage": all(v["usable_intervals"] == len(slots) for v in audited.values()),
        "limitations": [
            "Station observations do not establish weather over the entire fire landscape.",
            "The 2 m station wind is not a measured midflame wind over every vegetation type.",
            "Air relative humidity is not fuel moisture; no fuel-moisture conversion is implemented.",
            "Interval summaries are historical evidence, not instantaneous or pre-delivered controller observations.",
            "No weather forcing, interpolation, gap filling, or physical validation is performed.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", type=Path, required=True)
    parser.add_argument("--stations", type=Path, required=True)
    parser.add_argument("--variables", type=Path, required=True)
    parser.add_argument("--measurements", type=Path, required=True)
    parser.add_argument("--station-code", default="X6")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit_weather(json.loads(args.scenario.read_text()), args.stations, args.variables, args.measurements, args.station_code)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"Weather audit saved to {args.output}; complete valid coverage: {result['complete_valid_coverage']}")


if __name__ == "__main__":
    main()
