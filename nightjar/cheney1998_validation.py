"""Isolated retrospective grass-rate comparison; freeze before new errors.

No controller, production propagation, filtered FireFlux arrays or fitted
parameters enter this experiment. See docs/CHENEY_1998_PROTOCOL.md.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
import platform
import subprocess
import tempfile

from . import cheney1998, csiro_head_validation as csiro, fireflux_validation as fireflux

SCHEMA_VERSION = "nightjar-cheney1998-comparison-0.1"
PROTOCOL_VERSION = "nightjar-cheney1998-protocol-0.1"
ROOT = Path(__file__).resolve().parent.parent
LITERATURE_URL = "https://mdpi-res.com/d_attachment/fire/fire-05-00055/article_deploy/fire-05-00055.pdf"
LITERATURE_HASH = "3ecfa6cd2329412c898a3e61bc57dd895c9e605310254c9c80c6a9edaf3ef08d"
TREATMENT_MAPPING = {"E1": "undisturbed", "E2": "cut_grazed", "E3": "cut_grazed",
                     "T1": "undisturbed", "T2": "cut_grazed", "T3": "cut_grazed"}
SOURCES = [
    {"doi": "10.1071/WF9980001", "access": "abstract only; original PDF not acquired"},
    {"doi": "10.1080/00049158.2015.1055063", "locator": "corrected coauthor restatement pp126-128 Eqs6-11; original curing10a",
     "corrigendum": "10.1080/00049158.2015.1080789", "access": "full author-uploaded text read; no downloaded-byte hash",
     "url": "https://www.researchgate.net/publication/281801393_Empirical-based_models_for_predicting_head-fire_rate_of_spread_in_Australian_fuel_types"},
    {"doi": "10.3390/fire5020055", "locator": "Appendix A3-A7 pp13-14; A4 0.705 and A7 updated curing are excluded", "url": LITERATURE_URL},
    {"url": "https://publications.gc.ca/collections/collection_2010/nrcan/Fo123-2-10-2009-eng.pdf",
     "locator": "GLC-X-10 section3.2.2 p9 original Australian curing curve; read PDF, not locally acquired"},
    {"doi": "10.1071/WF14146", "locator": "primary curing experiment pp838-848; later model excluded"},
    {"doi": "10.1071/WF23100", "locator": "Table1 wind profile extrapolation and treatment definitions"},
    {"doi": "10.13140/RG.2.2.28866.68804", "locator": "CSIRO2017 section3.1.4 Australian U2/U10 near0.8; not FireFlux conversion evidence"},
    {"doi": "10.1071/WF18082", "locator": "Appendix U2/U10=0.81; sensitivity assumption only, not validated in Texas"},
]


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def write_new_json(path: Path, value: dict) -> None:
    """Never overwrite a previous protocol/report."""
    encoded = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        stream.write(encoded)


def fetch_literature(directory: Path) -> None:
    """Acquire only the new pinned 2022 PDF; never overwrite a source."""
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / "cruz2022.pdf"
    if target.exists():
        if digest(target) != LITERATURE_HASH:
            raise ValueError("existing literature hash differs; refusing overwrite")
        return
    with tempfile.NamedTemporaryFile(dir=directory, delete=False) as f:
        temporary = Path(f.name)
    try:
        subprocess.run(["curl", "--proto", "=https", "--location", "--fail", "--silent", "--show-error",
                        "--output", str(temporary), LITERATURE_URL], check=True)
        if digest(temporary) != LITERATURE_HASH:
            raise ValueError("downloaded literature hash differs; review source version")
        # Exclusive creation even if another process acquired it meanwhile.
        with target.open("xb") as f:
            f.write(temporary.read_bytes())
    finally:
        temporary.unlink(missing_ok=True)


def load_candidate_inputs(paths: dict) -> tuple[dict, dict]:
    """Whitelist physical columns; retain discrepant source variants explicitly."""
    with paths["Plot_Average_Data.csv"].open(newline="") as f:
        plots = {r["FireID"]: r for r in csv.DictReader(f) if r["FireID"] in csiro.PAPER_INPUTS}
    with paths["Burn_Period_Data.csv"].open(newline="") as f:
        periods = [r for r in csv.DictReader(f) if r["FireID"] in csiro.PAPER_INPUTS]
    inputs, audit = {}, {}
    for burn in csiro.PAPER_INPUTS:
        plot = plots[burn]
        base = {"open_wind_10m_km_h": float(plot["10mWind.ms"]) * 3.6,
                "dead_moisture_percent": float(plot["DFMC.%"]),
                "curing_percent": float(plot["Cure.%"]),
                "pasture_condition": TREATMENT_MAPPING[plot["FuelTreat"]]}
        # Validate without calculating a speed or looking at outcome columns.
        cheney1998.GrassInputs(**base)
        variants = {"plot_table": base}
        alternatives = {(r["FuelTreat"], float(r["Cure.%"]), float(r["DFMC.%"]))
                        for r in periods if r["FireID"] == burn}
        if len(alternatives) != 1:
            raise ValueError(f"{burn}: period forcing needs a separately reviewed protocol")
        treatment, curing, moisture = alternatives.pop()
        period = {**base, "dead_moisture_percent": moisture, "curing_percent": curing,
                  "pasture_condition": TREATMENT_MAPPING[treatment]}
        cheney1998.GrassInputs(**period)
        if period != base or treatment != plot["FuelTreat"]:
            variants["period_fuel_variant"] = period
        inputs[burn] = variants
        audit[burn] = {"plot_treatment": plot["FuelTreat"], "period_treatment": treatment,
                       "plot_curing_percent": base["curing_percent"], "period_curing_percent": curing,
                       "wind_source_column": "10mWind.ms", "wind_source_m_s": float(plot["10mWind.ms"]),
                       "wind_provenance": "source-provided open10m extrapolation of2m wind using measured profile; WF23100 Table1",
                       "retrospective_whole_burn_mean": True,
                       "head_width_m_for_eligibility_only": float(plot["EffHFWidth.m"])}
    return inputs, audit


def experiment_definition(annaburroo: Path, firemodels: Path, fireflux_sources: Path, literature: Path) -> dict:
    paths, csiro_sources = csiro._verified_sources(annaburroo, firemodels)
    fireflux_evidence = fireflux._verified_sources(fireflux_sources)
    pdf = literature / "cruz2022.pdf"
    if digest(pdf) != LITERATURE_HASH:
        raise ValueError("2022 literature source changed")
    inputs, audit = load_candidate_inputs(paths)
    engine_version = version("pyretechnics")
    if engine_version != "2026.8.10":
        raise ValueError("baseline engine requires pinned pyretechnics2026.8.10")
    files = ["nightjar/cheney1998.py", "nightjar/cheney1998_validation.py", "nightjar/csiro_head_validation.py",
             "nightjar/fireflux_validation.py", "nightjar/rothermel.py", "docs/CHENEY_1998_PROTOCOL.md"]
    return {
        "model_version": cheney1998.MODULE_VERSION, "inputs": inputs, "input_audit": audit,
        "runtime": {"python": platform.python_version(), "pyretechnics": engine_version},
        "code_and_protocol_sha256": {name: digest(ROOT / name) for name in files},
        "sources": csiro_sources + fireflux_evidence + [{"filename": "cruz2022.pdf", "sha256": LITERATURE_HASH, "url": LITERATURE_URL}],
        "literature": SOURCES, "source_access_date": "2026-09-30",
        "prior_models": list(csiro.MODELS), "production_replacement": False,
        "metrics": ["equal-interval per-burn/source rate nMAE", "normalized signed bias", "head position MAE m", "conditional arrival MAE s"],
        "primary_windows_s": {"C064": [[27, 53], [53, 100]], "F19": [[56, 86]]},
        "source_series": ["digitized_max_x", "prepared_head", "original_cumulative_distance"],
        "initialization": "first observed head only; later head resets and clipping forbidden",
        "secondary": "all F19 intervals ending after86s retained; period C064 fuel variant separate",
        "gates": {"rate_nmae_lt": 0.3, "absolute_normalized_bias_lt": 0.2,
                  "restricted_improvement": "both burns/all three source definitions pass and lower nMAE than original-input FM1; variants retained separately",
                  "physical_validation_pass": None},
        "acceptance_exclusion": "CSIRO development-campaign overlap; potential wide-head law versus finite developing fronts; retrospectively observed forcing",
        "fireflux": {"eligibility": "not_evaluable", "missing": ["reconciled open10m wind", "reconciled curing"],
                     "prior_conditional_mell_inputs": fireflux.FORCING, "published_rate_m_s": fireflux.OBSERVED_RATE_M_S,
                     "sensitivity_only": {"wind_2m_m_s": [3.0, 4.0], "dead_moisture_percent": [9.0, 18.5], "curing_percent": [95.0, 100.0],
                                          "pasture_condition": ["undisturbed", "cut_grazed", "eaten_out"], "assumed_u2_over_u10": 0.81},
                     "filtered_arrays_used": False},
        "other_ineligible": {"RxCADRE": "existing authentication and input eligibility blocker",
                             "CEMS": "not matched continuous grass inputs; mixed fuels/forcing/interventions"},
        "observation_access": "all earlier observed outcomes already known; retrospective, unfitted evaluation, not blind",
        "units": {"wind_input": "km/h at10m open", "dead_moisture": "% oven-dry mass", "curing": "%", "rate": "m/s", "position": "m", "elapsed_time": "s"},
        "crs": "LOCAL_PLOT_METERS for CSIRO; LOCAL_TRANSECT_METERS for FireFlux",
        "time_origin": "existing elapsed ignition clocks; CSIRO absolute timezone unresolved; FireFlux published CST clocks unreconciled",
        "seed": 0, "seed_policy": "deterministic arithmetic only, no stochastic tests or simulations",
    }


def freeze_protocol(protocol: Path, **roots) -> dict:
    definition = experiment_definition(**roots)
    value = {"schema_version": PROTOCOL_VERSION, "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
             "definition": definition, "definition_sha256": csiro.canonical_hash(definition)}
    write_new_json(protocol, value)
    return value


def checked_protocol(protocol: Path, **roots) -> dict:
    value = json.loads(protocol.read_text())
    definition = experiment_definition(**roots)
    if value.get("schema_version") != PROTOCOL_VERSION or value.get("definition") != definition or value.get("definition_sha256") != csiro.canonical_hash(definition):
        raise ValueError("protocol/code/source/configuration mismatch; do not calculate errors under a changed freeze")
    frozen = datetime.fromisoformat(value["frozen_at_utc"])
    if frozen.tzinfo is None or frozen.utcoffset().total_seconds() != 0 or frozen > datetime.now(timezone.utc):
        raise ValueError("freeze requires a valid past UTC timestamp")
    return value


def run_experiment(protocol: Path, **roots) -> dict:
    frozen = checked_protocol(protocol, **roots)
    definition = frozen["definition"]
    # All input-only predictions before loading outcomes; historical blindness is not claimed.
    components = {burn: {variant: cheney1998.grass_rate(cheney1998.GrassInputs(**values))
                         for variant, values in variants.items()}
                  for burn, variants in definition["inputs"].items()}
    previous = csiro.run_diagnostic(roots["annaburroo"], roots["firemodels"])
    scores = {}
    for burn, variants in components.items():
        scores[burn] = {}
        for variant, result in variants.items():
            rate = result["rate_m_s"]
            if rate == 0:
                # Zero is valid model output; retain observations and rate errors.
                scores[burn][variant] = {source: zero_rate_score(burn, samples)
                                        for source, samples in previous["observations"][burn].items()}
            else:
                scores[burn][variant] = {source: csiro.score_series(burn, samples, rate)
                                        for source, samples in previous["observations"][burn].items()}
            for score in scores[burn][variant].values():
                score["physical_validation_pass"] = None
                score["acceptance_exclusion"] = definition["acceptance_exclusion"]
    improvement = all(s["restricted_rate_gate_pass"] and s["rate_nmae"] < previous["scores"][burn]["fm1_original_inputs"][source]["rate_nmae"]
                      for burn in scores for source, s in scores[burn]["plot_table"].items())
    sensitivities = []
    grid = definition["fireflux"]["sensitivity_only"]
    for wind in grid["wind_2m_m_s"]:
        for moisture in grid["dead_moisture_percent"]:
            for curing in grid["curing_percent"]:
                for pasture in grid["pasture_condition"]:
                    values = {"open_wind_10m_km_h": wind / grid["assumed_u2_over_u10"] * 3.6,
                              "dead_moisture_percent": moisture, "curing_percent": curing, "pasture_condition": pasture}
                    result = cheney1998.grass_rate(cheney1998.GrassInputs(**values))
                    residual = result["rate_m_s"] - fireflux.OBSERVED_RATE_M_S
                    sensitivities.append({"assumed_inputs": values, "original_u2_m_s": wind, "components": result,
                                          "normalized_absolute_rate_error": abs(residual) / fireflux.OBSERVED_RATE_M_S,
                                          "normalized_signed_bias": residual / fireflux.OBSERVED_RATE_M_S,
                                          "rate_residual_m_s": residual, "eligible_main_comparison": False,
                                          "physical_validation_pass": None})
    prior_fireflux = fireflux.run_diagnostic(roots["fireflux_sources"])
    return {"schema_version": SCHEMA_VERSION, "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "protocol": frozen, "protocol_file_sha256": digest(protocol), "seed": 0,
            "units": definition["units"], "crs": definition["crs"], "time_origin": definition["time_origin"],
            "runtime": {"python": platform.python_version(), **previous["runtime"]},
            "candidate_components": components, "candidate_scores": scores,
            "restricted_improvement_across_burns_and_sources": improvement,
            "conclusion": "restricted_improvement" if improvement else "fixed_CSIRO_diagnostic_fails_improvement_criterion",
            "physical_conclusion": "insufficient_evidence", "physical_validation_pass": None,
            "previous_csiro_experiment": previous, "previous_fireflux_experiment": prior_fireflux,
            "fireflux_main_comparison": {"status": "not_evaluable", "reasons": definition["fireflux"]["missing"], "candidate_rate_m_s": None},
            "fireflux_sensitivity_only": sensitivities,
            "other_ineligible_cases": definition["other_ineligible"],
            "limitations": ["Potential wide-head rate is not measured finite-head acceleration; no head-width correction is fitted.",
                            "Original1998 PDF unavailable; reconstructed from coauthor restatement with explicitly resolved version discrepancies.",
                            "Source measurement/profile/timing/digitization errors unquantified; sensitivities are not confidence intervals.",
                            "No grid propagator was run or replaced; these residuals contain no spatial numerical error.",
                            "FireFlux conversion/curing are sensitivity assumptions only, not new physical inputs."]}


def zero_rate_score(burn: str, samples: list) -> dict:
    """Retain extinguished predictions without inventing finite arrival times."""
    result = csiro.score_series(burn, samples, 1.0)
    primary_rates = []
    first_x = result["initial_observed_position_m"]
    for row in result["records"]:
        row.update(predicted_rate_m_s=0.0, rate_residual_m_s=-row["observed_rate_m_s"],
                   predicted_position_m=first_x, position_residual_m=first_x - row["observed_position_m"],
                   predicted_arrival_s=None, arrival_residual_s=None)
        if row["eligible_primary_window"]:
            primary_rates.append(row)
    result.update(predicted_rate_m_s=0.0, rate_nmae=1.0, rate_normalized_bias=-1.0,
                  position_mae_m=sum(abs(r["position_residual_m"]) for r in primary_rates) / len(primary_rates),
                  arrival_mae_s=None, restricted_rate_gate_pass=False)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("freeze", "run"))
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--annaburroo", type=Path, default=Path("data/raw/annaburroo"))
    parser.add_argument("--firemodels", type=Path, default=Path("data/raw/csiro-firemodels"))
    parser.add_argument("--fireflux-sources", type=Path, default=Path("data/raw/fireflux"))
    parser.add_argument("--literature", type=Path, default=Path("data/raw/cheney1998"))
    parser.add_argument("--fetch-literature", action="store_true", help="explicitly acquire the new pinned2022 PDF only")
    args = parser.parse_args()
    if args.mode == "run" and args.output is None:
        parser.error("run requires --output")
    if args.mode == "freeze" and args.output is not None:
        parser.error("freeze writes --protocol only")
    # Check existing output before doing arithmetic or network work.
    destination = args.protocol if args.mode == "freeze" else args.output
    if destination.exists():
        parser.error(f"{destination}: refusing to overwrite previous artifact")
    if args.fetch_literature:
        fetch_literature(args.literature)
    roots = {"annaburroo": args.annaburroo, "firemodels": args.firemodels,
             "fireflux_sources": args.fireflux_sources, "literature": args.literature}
    if args.mode == "freeze":
        freeze_protocol(args.protocol, **roots)
    else:
        write_new_json(args.output, run_experiment(args.protocol, **roots))


if __name__ == "__main__":
    main()
