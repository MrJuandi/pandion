"""Declared Rothermel fuel-bed and ignition-growth sensitivity on Annaburroo."""

from __future__ import annotations

import argparse
import csv
from math import exp, isfinite
import json
from pathlib import Path
from statistics import mean, median

from .annaburroo_temporal import compare as temporal_compare
from .rothermel import one_class_components


TAU_SECONDS = (0, 30, 60, 120)


def growth_distance(rate_m_s: float, time_s: float, tau_s: float) -> float:
    """Integral of rate*(1-exp(-t/tau)); tau=0 is constant rate."""
    if tau_s < 0 or rate_m_s < 0 or time_s < 0:
        raise ValueError("rate, time and growth timescale must be nonnegative")
    return rate_m_s * (time_s - tau_s * (1 - exp(-time_s / tau_s))) if tau_s else rate_m_s * time_s


def compare(plot_table: Path, period_table: Path) -> dict:
    baseline = temporal_compare(plot_table, period_table)  # Audits both exact source hashes.
    with plot_table.open(encoding="utf-8-sig", newline="") as source:
        plots = {row["FireID"]: row for row in csv.DictReader(source)}
    with period_table.open(encoding="utf-8-sig", newline="") as source:
        periods = list(csv.DictReader(source))
    sav_by_fire: dict[str, set[float]] = {}
    for row in periods:
        if row["FireID"] in plots:
            try:
                sav = float(row["SAVR.cm"])
            except ValueError:
                continue
            if isfinite(sav) and sav > 0:
                sav_by_fire.setdefault(row["FireID"], set()).add(sav)
    rate_by_fire = {}
    omitted = []
    for fire in baseline["fires"]:
        fire_id = fire["fire_id"]
        plot = plots[fire_id]
        try:
            load = float(plot["FuelLoad.kgm2"])
            depth = float(plot["FuelHt.m"])
        except ValueError:
            omitted.append(fire_id)
            continue
        sav_values = sav_by_fire.get(fire_id, set())
        if len(sav_values) != 1 or not all(isfinite(v) and v > 0 for v in (load, depth)):
            omitted.append(fire_id)
            continue
        pieces = one_class_components(fire["dead_fuel_moisture_fraction"], fire["midflame_wind_m_s"],
                                      load, depth, next(iter(sav_values)))
        rate_by_fire[fire_id] = pieces["r0_m_s"] * (1 + pieces["phi_w"])
    comparable = [fire for fire in baseline["fires"] if fire["fire_id"] in rate_by_fire]
    if not comparable:
        raise ValueError("no comparable fires with measured fuel-bed inputs")
    variants = {}
    for fuel_bed in ("FM1", "measured_bed_all_dead"):
        for tau_s in TAU_SECONDS:
            errors = []
            period_errors = []
            rows = []
            for fire in comparable:
                rate = (fire["final"]["predicted_head_rate_m_s"] if fuel_bed == "FM1"
                        else rate_by_fire[fire["fire_id"]])
                final = fire["final"]
                predicted = growth_distance(rate, final["time_since_ignition_s"], tau_s)
                error = predicted - final["observed_head_distance_m"]
                errors.append(error)
                period_errors.extend(growth_distance(rate, p["time_since_ignition_s"], tau_s)
                                     - p["observed_head_distance_m"] for p in fire["periods"])
                rows.append({"fire_id": fire["fire_id"], "predicted_rate_m_s": rate,
                             "observed_final_distance_m": final["observed_head_distance_m"],
                             "predicted_final_distance_m": predicted, "final_error_m": error})
            variants[f"{fuel_bed}_tau_{tau_s}s"] = {
                "fuel_bed": fuel_bed, "tau_s": tau_s, "fires": len(rows),
                "final_distance_mae_m": mean(map(abs, errors)),
                "final_distance_median_absolute_error_m": median(map(abs, errors)),
                "final_distance_bias_m": mean(errors),
                "period_point_mae_m": mean(map(abs, period_errors)),
                "per_fire": rows,
            }
    return {"schema_version": "annaburroo-rothermel-sensitivity-0.1",
            "source": baseline["source"], "included_fires": len(comparable),
            "omitted_fires_without_fuel_bed": omitted,
            "assumptions": ["Measured plot-average dry load and height are used as one homogeneous dead fuel bed.",
                            "Species SAV is taken from the burn-period table; wind and moisture stay at plot-average values.",
                            "Particle properties and 12% moisture extinction remain FM1 defaults; live fuel and curing are omitted.",
                            "Growth timescales are prespecified sensitivity cases, not fitted or validated parameters."],
            "variants": variants}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path("data/raw/annaburroo"))
    parser.add_argument("--output", type=Path, default=Path("runs/annaburroo-rothermel-sensitivity.json"))
    args = parser.parse_args()
    result = compare(args.directory / "Plot_Average_Data.csv", args.directory / "Burn_Period_Data.csv")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({**result, "variants": {key: {k: v for k, v in value.items() if k != "per_fire"}
                                                for key, value in result["variants"].items()}}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
