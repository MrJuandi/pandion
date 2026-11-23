"""Timestamped, coverage-limited fire observations from a synthetic replay.

This is an information-flow test component; it does not model a real sensor's
detection accuracy, acquisition schedule, or probability of cloud/smoke cover.
"""

from __future__ import annotations

import argparse
import json
from math import isfinite
from pathlib import Path


def build_reports(arrival_time_s: list[list[float | None]], schedule: list[dict]) -> list[dict]:
    """Capture fire-presence booleans at observation times for specified cells."""
    if not arrival_time_s or not arrival_time_s[0] or any(len(row) != len(arrival_time_s[0]) for row in arrival_time_s):
        raise ValueError("arrival map must be a nonempty rectangular grid")
    rows, cols = len(arrival_time_s), len(arrival_time_s[0])
    reports = []
    for item in schedule:
        observed = float(item["observed_at_s"])
        available = float(item["available_at_s"])
        if not isfinite(observed) or not isfinite(available) or observed < 0 or available < observed:
            raise ValueError("each report needs nonnegative observation time and delivery no earlier than observation")
        if not isinstance(item.get("source_id"), str) or not item["source_id"]:
            raise ValueError("source_id must be a nonempty string")
        cells = item["coverage_cells"]
        if not isinstance(cells, list):
            raise ValueError("coverage_cells must be a list")
        captured = []
        seen = set()
        for coordinate in cells:
            if not isinstance(coordinate, list) or len(coordinate) != 2 or any(not isinstance(v, int) or isinstance(v, bool) for v in coordinate):
                raise ValueError("coverage cell must be [row, col]")
            r, c = coordinate
            if not (0 <= r < rows and 0 <= c < cols) or (r, c) in seen:
                raise ValueError("coverage cell is out of bounds or duplicated")
            seen.add((r, c))
            arrival = arrival_time_s[r][c]
            captured.append({"cell": [r, c], "fire_detected": arrival is not None and arrival <= observed})
        reports.append({
            "source_id": item["source_id"], "observed_at_s": observed,
            "available_at_s": available, "cells": captured,
        })
    return reports


def observe(reports: list[dict], decision_time_s: float, rows: int, cols: int) -> dict:
    """Return latest *delivered* observation per cell, with age and missingness."""
    if not isfinite(decision_time_s) or decision_time_s < 0:
        raise ValueError("decision_time_s must be nonnegative and finite")
    if rows < 1 or cols < 1:
        raise ValueError("rows and cols must be positive")
    cells = [[None for _ in range(cols)] for _ in range(rows)]
    for report in reports:
        if report["available_at_s"] > decision_time_s:
            continue
        for captured in report["cells"]:
            r, c = captured["cell"]
            previous = cells[r][c]
            if previous is None or report["observed_at_s"] > previous["observed_at_s"]:
                cells[r][c] = {
                    "fire_detected": captured["fire_detected"],
                    "observed_at_s": report["observed_at_s"],
                    "age_s": decision_time_s - report["observed_at_s"],
                    "source_id": report["source_id"],
                }
    return {"time_s": decision_time_s, "cells": cells}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fire_replay", type=Path)
    parser.add_argument("schedule", type=Path)
    parser.add_argument("--time-s", type=float, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    fire = json.loads(args.fire_replay.read_text())
    schedule = json.loads(args.schedule.read_text())
    arrivals = fire["arrival_time_s"]
    reports = build_reports(arrivals, schedule)
    result = observe(reports, args.time_s, len(arrivals), len(arrivals[0]))
    result.update({
        "observation_schema_version": "synthetic-fire-observation-0.1",
        "scenario_id": fire["scenario_id"], "scenario_sha256": fire["scenario_sha256"],
        "seed": fire["seed"], "crs": fire["crs"], "provenance": "synthetic",
        "units": {"time": "s"},
    })
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered)
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
