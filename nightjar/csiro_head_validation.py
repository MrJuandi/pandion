"""Unfitted retrospective head-advance diagnostic for CSIRO C064 and F19.

This runs local spread laws against timed measurements. It does not simulate
closed perimeters, validate a grid propagator, or modify production dynamics.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import version
import json
from math import exp, isfinite
from pathlib import Path
import platform
import subprocess

from . import rothermel

EXP_REVISION = "e5de6811036d252b4f9085866f08da66efdda7d3"
SOURCE_HASHES = {
    "Plot_Average_Data.csv": "4425d11fe970e263d400b7a9d089b8eea70bda4de4a1e29cbeedea28bc727934",
    "Burn_Period_Data.csv": "bdd5bc1354b7b9e72e17616aa733b5bc869fa3b86686ea102dcb73846761ca0d",
    "Case_C064_exp.csv": "11626c452c5a9b695392a02b32bacd3138e5bbc0094b33312e01a2053c90154a",
    "Case_F19_exp.csv": "a24ae185d89810d029b248b8d8e462925c69b4f9cd431528460a896e49465df4",
    "Case_C064_flame_position_data.csv": "651989688986077ce633ef6699dbf65b81289a39da841f8b1aa088b002bd54f0",
    "Case_F19_flame_position_data.csv": "c7c52b85302622bbceeea2a8113c7b2bb801f7c64025a6101ef1ac5b5c191ce7",
}
PAPER_INPUTS = {
    "C064": {"load_kg_m2": 0.283, "depth_m": 0.21, "sav_m_inv": 9770.0,
             "moisture_fraction": 0.063, "midflame_wind_m_s": 2.8},
    "F19": {"load_kg_m2": 0.313, "depth_m": 0.51, "sav_m_inv": 12240.0,
            "moisture_fraction": 0.058, "midflame_wind_m_s": 4.4},
}
MODELS = ("fm1_original_inputs", "fm1_paper_inputs", "anderson_1_3_original_inputs",
          "measured_dead_original_inputs", "measured_dead_paper_inputs",
          "cheney_mell_reference")


def canonical_hash(value: dict) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _verified_sources(annaburroo: Path, firemodels: Path) -> tuple[dict, list[dict]]:
    paths, evidence = {}, []
    for name, expected in SOURCE_HASHES.items():
        path = (firemodels if name.startswith("Case_") else annaburroo) / name
        actual = sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"{name}: source hash changed; review source version before running")
        paths[name] = path
        evidence.append({"filename": name, "sha256": actual,
                         "source": f"firemodels/exp@{EXP_REVISION}" if name.startswith("Case_") else "csiro:58746 version 1 archive 64873323"})
    return paths, evidence


def fetch_sources(annaburroo: Path, firemodels: Path) -> None:
    """Explicit acquisition; default diagnostic execution never downloads."""
    from .annaburroo import fetch_tables

    manifest = fetch_tables(annaburroo)
    (annaburroo / "source-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    firemodels.mkdir(parents=True, exist_ok=True)
    for name in SOURCE_HASHES:
        if not name.startswith("Case_"):
            continue
        url = f"https://raw.githubusercontent.com/firemodels/exp/{EXP_REVISION}/CSIRO_Grassland_Fires/{name}"
        result = subprocess.run(["curl", "--proto", "=https", "--location", "--fail", "--silent", "--show-error", "--output", str(firemodels / name), url], capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(f"{name}: source download failed")
    _, evidence = _verified_sources(annaburroo, firemodels)
    (firemodels / "diagnostic-source-manifest.json").write_text(json.dumps({
        "schema_version": "nightjar-csiro-head-sources-0.1",
        "acquired_at_utc": datetime.now(timezone.utc).isoformat(),
        "files": evidence, "artifact_kind": "raw source inventory, not a scenario or replay",
    }, indent=2) + "\n")


def load_forcing(paths: dict) -> dict:
    """Whitelist measured input columns; never use ROS/RSS to set predictions."""
    with paths["Plot_Average_Data.csv"].open(newline="") as handle:
        plots = {row["FireID"]: row for row in csv.DictReader(handle) if row["FireID"] in PAPER_INPUTS}
    with paths["Burn_Period_Data.csv"].open(newline="") as handle:
        periods = [row for row in csv.DictReader(handle) if row["FireID"] in PAPER_INPUTS]
    forcing = {}
    for burn in PAPER_INPUTS:
        plot = plots[burn]
        savs = {float(row["SAVR.cm"]) * 100 for row in periods if row["FireID"] == burn}
        if len(savs) != 1:
            raise ValueError(f"{burn}: inconsistent measured SAV")
        forcing[burn] = {
            "load_kg_m2": float(plot["FuelLoad.kgm2"]),
            "depth_m": float(plot["FuelHt.m"]), "sav_m_inv": savs.pop(),
            "moisture_fraction": float(plot["DFMC.%"]) / 100,
            "midflame_wind_m_s": float(plot["MidFlwind.ms"]),
            "wind_2m_m_s": float(plot["2mWind.ms"]),
            "effective_head_width_m": float(plot["EffHFWidth.m"]),
            "cure_fraction": float(plot["Cure.%"]) / 100,
        }
    return forcing


def cheney_mell_rate(wind_2m_m_s: float, moisture_fraction: float, width_m: float) -> float:
    """Mell et al. (2007), Eq. 2; moisture M is percent, W is metres.

    An in-campaign empirical reference, not an independently validated model.
    The equation assumes cured grass; no invented curing correction is applied.
    """
    if not all(isfinite(v) for v in (wind_2m_m_s, moisture_fraction, width_m)) or wind_2m_m_s < 0 or not 0 <= moisture_fraction <= 1 or width_m <= 0:
        raise ValueError("invalid grass-reference inputs")
    return (0.165 + 0.534 * wind_2m_m_s) * exp((-0.859 - 2.036 * wind_2m_m_s) / width_m) * exp(-0.108 * moisture_fraction * 100)


def predict_rates(forcing: dict) -> dict:
    """Receive only fuel/forcing, no subsequent front observations."""
    from pyretechnics import fuel_models, surface_fire

    def measured_rate(inputs: dict) -> float:
        pieces = rothermel.dead_fuel_components(
            inputs["moisture_fraction"], inputs["midflame_wind_m_s"],
            load_kg_m2=inputs["load_kg_m2"], depth_m=inputs["depth_m"],
            sav_m_inv=inputs["sav_m_inv"], moisture_extinction_fraction=0.22,
            heat_kj_kg=18608.0)
        return pieces["r0_m_s"] * (1 + pieces["phi_w"])

    predictions = {}
    for burn, inputs in forcing.items():
        moisture, wind = inputs["moisture_fraction"], inputs["midflame_wind_m_s"]
        pieces = rothermel.fm1_components(moisture, wind)
        paper_pieces = rothermel.fm1_components(PAPER_INPUTS[burn]["moisture_fraction"], PAPER_INPUTS[burn]["midflame_wind_m_s"])
        fuel = fuel_models.moisturize(fuel_models.get_fuel_model(1 if burn == "C064" else 3),
                                     (moisture, moisture, moisture, moisture, 0.3, 0.6))
        base = surface_fire.calc_surface_fire_behavior_no_wind_no_slope(fuel)
        behavior = surface_fire.calc_surface_fire_behavior_max(base, wind * 60, 270.0, 0.0, 0.0)
        predictions[burn] = {
            "fm1_original_inputs": pieces["r0_m_s"] * (1 + pieces["phi_w"]),
            "fm1_paper_inputs": paper_pieces["r0_m_s"] * (1 + paper_pieces["phi_w"]),
            "anderson_1_3_original_inputs": behavior["max_spread_rate"] / 60,
            "measured_dead_original_inputs": measured_rate(inputs),
            "measured_dead_paper_inputs": measured_rate(PAPER_INPUTS[burn]),
            "cheney_mell_reference": cheney_mell_rate(inputs["wind_2m_m_s"], moisture, inputs["effective_head_width_m"]),
        }
    return predictions


def forecast_head(initial_time_s: float, initial_position_m: float, rate_m_s: float, time_s: float) -> float:
    """Advance the first observed head without subsequent observation resets."""
    if not all(isfinite(v) for v in (initial_time_s, initial_position_m, rate_m_s, time_s)) or rate_m_s <= 0 or time_s <= initial_time_s:
        raise ValueError("forecast requires finite inputs, positive rate and later time")
    return initial_position_m + rate_m_s * (time_s - initial_time_s)


def parse_front_curves(path: Path) -> dict[float, list[tuple[float, float]]]:
    """Read paired elapsed-time x/y columns, including ragged empty pairs."""
    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        if len(header) % 2:
            raise ValueError("front header requires paired coordinates")
        times = []
        for i in range(0, len(header), 2):
            tx, xaxis = header[i].split()
            ty, yaxis = header[i + 1].split()
            if tx != ty or (xaxis, yaxis) != ("x", "y") or float(tx) in times:
                raise ValueError("front header requires unique matching time x/y pairs")
            times.append(float(tx))
        curves = {time: [] for time in times}
        for row in reader:
            if len(row) != len(header):
                raise ValueError("front row has wrong number of columns")
            for i, time in enumerate(times):
                x, y = row[2 * i:2 * i + 2]
                if not x.strip() and not y.strip():
                    continue
                if not x.strip() or not y.strip():
                    raise ValueError("unpaired front coordinate")
                point = float(x), float(y)
                if not all(isfinite(v) for v in point):
                    raise ValueError("nonfinite front coordinate")
                curves[time].append(point)
        if any(not points for points in curves.values()):
            raise ValueError("empty front curve")
        return curves


def load_observations(paths: dict) -> tuple[dict, dict, dict]:
    observations, curves, audit = {}, {}, {}
    with paths["Burn_Period_Data.csv"].open(newline="") as handle:
        periods = [row for row in csv.DictReader(handle) if row["FireID"] in PAPER_INPUTS]
    for burn in PAPER_INPUTS:
        curves[burn] = parse_front_curves(paths[f"Case_{burn}_flame_position_data.csv"])
        projected = [(time, max(x for x, _ in points)) for time, points in curves[burn].items()]
        with paths[f"Case_{burn}_exp.csv"].open(newline="") as handle:
            reader = csv.reader(handle)
            if next(reader) != ["s", "m"] or next(reader) != ["Time", "Front"]:
                raise ValueError("unexpected prepared head CSV units/header")
            prepared = [(float(row[0]), float(row[1])) for row in reader if row]
        rows = [row for row in periods if row["FireID"] == burn]
        original = [(float(row["TimeSinceIgn"]), float(row["CumDist.m"])) for row in rows]
        observations[burn] = {"digitized_max_x": projected, "prepared_head": prepared,
                              "original_cumulative_distance": original}
        audit[burn] = [{"time_s": float(row["TimeSinceIgn"]), "cumulative_distance_m": float(row["CumDist.m"]),
                        "reported_increment_m": float(row["IncDist.m"]), "reported_rate_m_s": float(row["ROS.ms"]),
                        "head_travel_bearing_deg": float(row["FireDir.deg"])} for row in rows]
    return observations, curves, audit


def score_series(burn: str, samples: list, rate_m_s: float) -> dict:
    """Score primary windows separately from the known F19 wind-shift tail."""
    if burn not in PAPER_INPUTS or not isfinite(rate_m_s) or rate_m_s <= 0:
        raise ValueError("invalid burn or predicted rate")
    samples = sorted(samples)
    if len(samples) < 2 or any(not isfinite(v) for pair in samples for v in pair) or any(t1 <= t0 or x1 <= x0 for (t0, x0), (t1, x1) in zip(samples, samples[1:])):
        raise ValueError("head samples require finite, strictly advancing distinct times")
    first_t, first_x = samples[0]
    records = []
    for (t0, x0), (t1, x1) in zip(samples, samples[1:]):
        predicted = forecast_head(first_t, first_x, rate_m_s, t1)
        observed_rate = (x1 - x0) / (t1 - t0)
        records.append({"window_start_s": t0, "window_end_s": t1,
                        "eligible_primary_window": burn != "F19" or t1 <= 86.0,
                        "observed_rate_m_s": observed_rate, "predicted_rate_m_s": rate_m_s,
                        "rate_residual_m_s": rate_m_s - observed_rate,
                        "observed_position_m": x1, "predicted_position_m": predicted,
                        "position_residual_m": predicted - x1,
                        "predicted_arrival_s": first_t + (x1 - first_x) / rate_m_s,
                        "arrival_residual_s": first_t + (x1 - first_x) / rate_m_s - t1})
    primary = [row for row in records if row["eligible_primary_window"]]
    if not primary:
        raise ValueError("no eligible primary interval")
    mean_rate = sum(row["observed_rate_m_s"] for row in primary) / len(primary)
    nmae = sum(abs(row["rate_residual_m_s"]) for row in primary) / len(primary) / mean_rate
    bias = sum(row["rate_residual_m_s"] for row in primary) / len(primary) / mean_rate
    return {"initial_observed_time_s": first_t, "initial_observed_position_m": first_x,
            "primary_window_count": len(primary), "mean_observed_rate_m_s": mean_rate,
            "predicted_rate_m_s": rate_m_s, "rate_nmae": nmae, "rate_normalized_bias": bias,
            "position_mae_m": sum(abs(row["position_residual_m"]) for row in primary) / len(primary),
            "arrival_mae_s": sum(abs(row["arrival_residual_s"]) for row in primary) / len(primary),
            "restricted_rate_gate_pass": nmae < 0.30 and abs(bias) < 0.20,
            "records": records}


def run_diagnostic(annaburroo: Path, firemodels: Path) -> dict:
    paths, sources = _verified_sources(annaburroo, firemodels)
    forcing = load_forcing(paths)
    configuration = {
        "burns": ["C064", "F19"], "forcing": forcing, "paper_inputs": PAPER_INPUTS,
        "models": MODELS, "slope": 0.0, "measured_dead_extinction_fraction": 0.22,
        "measured_dead_heat_kj_kg": 18608.0, "measured_fuel_policy": "one dead particle class, all measured load dead; no fitted correction",
        "standard_fuel_policy": "Anderson 1 for C064, 3 for F19 per FDS experiment chapter; default Pyretechnics wind limit; only dead 1h class has load in these models",
        "standard_fuel_mapping_source": "https://github.com/firemodels/fds/blob/bb1fb3a6f90a9a7de16ecf436c0092bfbfe852f2/Manuals/FDS_Validation_Guide/Experiment_Chapter.tex",
        "initialization": "first observed head per source; excluded from scores; no later resets",
        "primary_observable": "max x of the observed leading-edge samples in the local plot frame",
        "primary_windows": {"C064": [[27, 53], [53, 100]], "F19": [[56, 86]]},
        "secondary_sources": "prepared head positions and original CumDist are distinct observables; never merged",
        "secondary_window_policy": "all source intervals ending at/before86s in F19; all available intervals in C064; original source includes extra85s sample",
        "wind_shift_rule": "F19 intervals ending after 86s secondary only; constant forcing is unsupported there",
        "rate_gate": {"nmae_lt": 0.30, "absolute_normalized_bias_lt": 0.20},
        "window_weighting": "equal primary intervals within each source and burn; never pool burns",
        "seed": 0,
    }
    # Mechanically set rates before reading outcome series. This is explicitly
    # retrospective: source observations were already inspected during design.
    frozen_at = datetime.now(timezone.utc).isoformat()
    predictions = predict_rates(forcing)
    observations, curves, audit = load_observations(paths)
    scores = {burn: {model: {source: score_series(burn, samples, rates[model])
                             for source, samples in observations[burn].items()}
                     for model in MODELS} for burn, rates in predictions.items()}
    for burn in scores.values():
        for row in burn["cheney_mell_reference"].values():
            row["restricted_rate_gate_pass"] = None
            row["acceptance_exclusion"] = "reference derived from same experimental campaign; no independent validation claim"
    return {
        "schema_version": "nightjar-csiro-head-diagnostic-0.1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "configuration_set_at_utc": frozen_at,
        "configuration_sha256": canonical_hash(configuration), "configuration": configuration,
        "observation_access_status": "already inspected; unfitted retrospective diagnostic, no blind evaluation split",
        "crs": "LOCAL_PLOT_METERS", "coordinate_definition": "digitized x/y as Fig16 in Mell2007, x along initial spread; no geographic georeferencing",
        "time_origin": "elapsed seconds from experimental ignition; absolute timezone unresolved",
        "experiment_local_ignition": {"C064": "1986-08-18T12:56:00", "F19": "1986-08-20T12:56:00"},
        "experiment_time_origin_utc": None, "seed": 0, "seed_policy": "deterministic equations; metadata only",
        "units": {"position": "m", "time": "s", "rate": "m/s", "wind": "m/s", "fuel_load": "kg/m2", "fuel_depth": "m", "sav": "m^-1", "moisture": "fraction"},
        "sources": sources, "literature": ["https://doi.org/10.1071/WF06002", "https://doi.org/10.1071/WF23100"],
        "runtime": {"python": platform.python_version(), "pyretechnics": version("pyretechnics")},
        "code_sha256": {"nightjar/csiro_head_validation.py": sha256(Path(__file__).read_bytes()).hexdigest(),
                        "nightjar/rothermel.py": sha256(Path(rothermel.__file__).read_bytes()).hexdigest()},
        "predicted_rates_m_s": predictions, "scores": scores,
        "observations": observations, "observed_leading_edge_xy_m": curves, "original_period_audit": audit,
        "physical_validation_pass": None,
        "limitations": [
            "Three primary time intervals in two burns, from an already inspected campaign; no independent held-out validation.",
            "Only longitudinal head advance is tested; no flank/back spread, complete perimeter, spotting, terrain, suppression or forest transfer.",
            "Whole-burn mean wind and effective head width are retrospective forcing, not causally available forecast/controller inputs.",
            "No measured error bounds on digitization, timing or input uncertainty; source differences are not statistical confidence intervals.",
            "Prepared head, digitized max-x and original cumulative travel are not interchangeable, especially when head direction changes.",
            "Later boundary-touching positions can be censored by firebreaks; predictions are not clipped to conceal overrun; primary results include these endpoints transparently.",
            "Measured-dead fuel approximates 95-100 percent cured grass as entirely dead; particle density/minerals are standard assumptions. No fitted fuel/wind multipliers.",
            "Changing the spatial propagator cannot resolve errors in this no-grid local head-rate comparison.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annaburroo", type=Path, default=Path("data/raw/annaburroo"))
    parser.add_argument("--firemodels", type=Path, default=Path("data/raw/csiro-firemodels"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fetch-sources", action="store_true", help="explicitly fetch and verify the six pinned raw tables before running")
    args = parser.parse_args()
    if args.fetch_sources:
        fetch_sources(args.annaburroo, args.firemodels)
    report = run_diagnostic(args.annaburroo, args.firemodels)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
