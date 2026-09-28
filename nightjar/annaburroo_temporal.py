"""No-fit FM1 head-distance diagnostic against Annaburroo burn-period records."""

from __future__ import annotations

import argparse
import csv
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
from statistics import mean, median

from .rothermel import FM1_MOISTURE_EXTINCTION, fm1_components


PLOT_SHA256 = "4425d11fe970e263d400b7a9d089b8eea70bda4de4a1e29cbeedea28bc727934"
PERIOD_SHA256 = "bdd5bc1354b7b9e72e17616aa733b5bc869fa3b86686ea102dcb73846761ca0d"


def _audited_rows(path: Path, expected_hash: str) -> tuple[list[dict[str, str]], str]:
    raw = path.read_bytes()
    digest = sha256(raw).hexdigest()
    if digest != expected_hash:
        raise ValueError(f"{path.name} differs from the audited CSIRO version")
    return list(csv.DictReader(raw.decode("utf-8-sig").splitlines())), digest


def compare(plot_table: Path, period_table: Path) -> dict:
    plots, plot_hash = _audited_rows(plot_table, PLOT_SHA256)
    periods, period_hash = _audited_rows(period_table, PERIOD_SHA256)
    plot_by_id = {row["FireID"]: row for row in plots}
    records_by_fire: dict[str, list[dict]] = {}
    excluded = []
    for row in periods:
        fire_id = row["FireID"]
        plot = plot_by_id.get(fire_id)
        if plot is None:
            ignition_length = row["IgnLngth.m"]
            ignition_type = "point" if ignition_length == "0" else "line_or_other"
            excluded.append({"fire_id": fire_id, "reason": "no matching plot-average wind record",
                             "ignition_type": ignition_type})
            continue
        try:
            wind = float(plot["MidFlwind.ms"])
            moisture = float(plot["DFMC.%"]) / 100
            elapsed = float(row["TimeSinceIgn"])
            observed = float(row["CumDist.m"])
            observed_ros = float(row["ROS.ms"])
        except ValueError:
            excluded.append({"fire_id": fire_id, "reason": "missing or nonnumeric wind, moisture, time, distance or ROS"})
            continue
        if (not all(isfinite(value) for value in (wind, moisture, elapsed, observed, observed_ros))
                or wind < 0 or not 0 <= moisture <= 1 or elapsed <= 0 or observed < 0 or observed_ros < 0):
            excluded.append({"fire_id": fire_id, "reason": "input outside declared domain"})
            continue
        if moisture >= FM1_MOISTURE_EXTINCTION:
            rate = 0.0
        else:
            components = fm1_components(moisture, wind)
            rate = components["r0_m_s"] * (1 + components["phi_w"])
        predicted = rate * elapsed
        records_by_fire.setdefault(fire_id, []).append({
            "time_since_ignition_s": elapsed, "observed_head_distance_m": observed,
            "predicted_fm1_head_distance_m": predicted,
            "error_m": predicted - observed, "predicted_head_rate_m_s": rate,
            "observed_period_ros_m_s": observed_ros, "rate_error_m_s": rate - observed_ros,
        })
    fires = []
    for fire_id, records in sorted(records_by_fire.items()):
        records.sort(key=lambda record: record["time_since_ignition_s"])
        if any(records[i]["time_since_ignition_s"] <= records[i - 1]["time_since_ignition_s"] or
               records[i]["observed_head_distance_m"] < records[i - 1]["observed_head_distance_m"]
               for i in range(1, len(records))):
            raise ValueError(f"{fire_id}: temporal head-distance series is not strictly ordered and nondecreasing")
        plot = plot_by_id[fire_id]
        fires.append({"fire_id": fire_id, "fuel_type": plot["FuelType"], "fuel_treatment": plot["FuelTreat"],
                      "midflame_wind_m_s": float(plot["MidFlwind.ms"]),
                      "dead_fuel_moisture_fraction": float(plot["DFMC.%"]) / 100,
                      "period_count": len(records), "final": records[-1], "periods": records})
    if not fires:
        raise ValueError("no comparable burn-period records")
    final_errors = [fire["final"]["error_m"] for fire in fires]
    all_errors = [record["error_m"] for fire in fires for record in fire["periods"]]
    rate_errors = [record["rate_error_m_s"] for fire in fires for record in fire["periods"]]
    by_type = {}
    for fuel_type in sorted({fire["fuel_type"] for fire in fires}):
        values = [fire["final"]["error_m"] for fire in fires if fire["fuel_type"] == fuel_type]
        by_type[fuel_type] = {"fires": len(values), "final_distance_mae_m": mean(abs(value) for value in values),
                              "final_distance_bias_m": mean(values)}
    return {
        "schema_version": "annaburroo-temporal-fm1-0.1",
        "source": {"collection": "csiro:58746", "version": 1, "plot_sha256": plot_hash,
                   "period_sha256": period_hash, "plot_rows": len(plots), "period_rows": len(periods)},
        "selection": "all burn periods with matching plot-average midflame wind and moisture, positive elapsed time, nonnegative cumulative head distance and period ROS",
        "prediction": "unmodified FM1 flat-terrain head rate from plot-average midflame wind and dead-fuel moisture multiplied by time since ignition; zero at or above 12% extinction moisture; no fitted parameters or acceleration",
        "target": "CumDist.m: observed head-fire distance from ignition at TimeSinceIgn",
        "included_fires": len(fires), "included_periods": len(all_errors),
        "excluded_periods": len(excluded), "exclusion_counts": {
            reason: sum(item["reason"] == reason for item in excluded) for reason in sorted({item["reason"] for item in excluded})},
        "unmatched_plot_ignition_types": {
            kind: sum(item.get("ignition_type") == kind for item in excluded)
            for kind in ("point", "line_or_other")},
        "final_distance_mae_m": mean(abs(value) for value in final_errors),
        "final_distance_median_absolute_error_m": median(abs(value) for value in final_errors),
        "final_distance_bias_m": mean(final_errors),
        "mean_observed_final_distance_m": mean(fire["final"]["observed_head_distance_m"] for fire in fires),
        "mean_predicted_final_distance_m": mean(fire["final"]["predicted_fm1_head_distance_m"] for fire in fires),
        "period_point_mae_m": mean(abs(value) for value in all_errors),
        "period_ros_mae_m_s": mean(abs(value) for value in rate_errors),
        "mean_fire_period_ros_mae_m_s": mean(mean(abs(record["rate_error_m_s"]) for record in fire["periods"]) for fire in fires),
        "by_fuel_type": by_type,
        "fires_at_or_above_fm1_extinction": [fire["fire_id"] for fire in fires if fire["dead_fuel_moisture_fraction"] >= FM1_MOISTURE_EXTINCTION],
        "fires": fires,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path("data/raw/annaburroo"))
    parser.add_argument("--output", type=Path, default=Path("runs/annaburroo-temporal-fm1.json"))
    args = parser.parse_args()
    result = compare(args.directory / "Plot_Average_Data.csv", args.directory / "Burn_Period_Data.csv")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "fires"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
