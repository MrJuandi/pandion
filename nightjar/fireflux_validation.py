"""Offline, unfitted upper-bound diagnostic for the FireFlux I grass-fire rate.

The primary quantity is the analytic W -> infinity upper bound of the
unchanged Mell et al. (2007) Eq. 2 law. It is not a finite-width forecast.
This module never downloads sources unless ``--fetch-sources`` is supplied.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from math import exp, isfinite, log
from pathlib import Path
import platform
import subprocess
import tempfile

from .csiro_head_validation import cheney_mell_rate

SCHEMA_VERSION = "nightjar-fireflux-upper-bound-0.1"
SOURCE_URLS = {
    "clements2007.pdf": "https://research.fs.usda.gov/download/treesearch/14829.pdf",
    "kochanski2013.pdf": "https://gmd.copernicus.org/articles/6/1109/2013/gmd-6-1109-2013.pdf",
}
SOURCE_HASHES = {
    "clements2007.pdf": "861a38a211887d7d06115506b557be39975f0bc6f0ce83584a4ddbde184a3053",
    "kochanski2013.pdf": "7317593e559aea86030c6b2568656d69086db211c88cea3e08f12fa84bb604ae",
}
FORCING = {"wind_2m_m_s": 3.0, "dead_moisture_fraction": 0.09}
SENSITIVITY_WINDS_M_S = (3.0, 4.0)
SENSITIVITY_MOISTURES = (0.09, 0.185)
OBSERVED_RATE_M_S = 1.61
OBSERVED_RATE_ROUNDING_M_S = (1.605, 1.615)
TOWER_SEPARATION_M = 300.0
INITIAL_HEAD_TIME_S = 152.0
INITIAL_HEAD_POSITION_M = -45.0
FRONT_PASSAGE_TIME_S = 190.0
PLUME_PEAK_TIME_S = 195.0
KOCHANSKI_MT_PEAK_TIME_S = 219.0
IGNITION_WIDTH_SURROGATE_M = 385.0
RATE_GATES = {"nmae_lt": 0.30, "absolute_bias_lt": 0.20}


def canonical_hash(value: dict) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _verified_sources(sources: Path) -> list[dict]:
    evidence = []
    for filename, expected in SOURCE_HASHES.items():
        path = sources / filename
        if not path.is_file():
            raise FileNotFoundError(f"{path}: pinned FireFlux source is missing")
        actual = sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"{filename}: source hash changed; review source version before running")
        evidence.append({"filename": filename, "sha256": actual, "url": SOURCE_URLS[filename],
                         "kind": "pinned published source PDF"})
    return evidence


def fetch_sources(sources: Path) -> None:
    """Explicitly acquire PDFs; refuse to overwrite any mismatched existing file."""
    sources.mkdir(parents=True, exist_ok=True)
    downloaded = []
    for filename, url in SOURCE_URLS.items():
        target = sources / filename
        if target.exists():
            actual = sha256(target.read_bytes()).hexdigest()
            if actual != SOURCE_HASHES[filename]:
                raise ValueError(f"{target}: existing source does not match pinned hash; refusing to overwrite")
            continue
        with tempfile.NamedTemporaryFile(dir=sources, prefix=f".{filename}.", delete=False) as handle:
            temporary = Path(handle.name)
        try:
            result = subprocess.run(["curl", "--proto", "=https", "--location", "--fail", "--silent", "--show-error",
                                     "--output", str(temporary), url], capture_output=True, text=True)
            if result.returncode:
                raise RuntimeError(f"{filename}: source download failed")
            actual = sha256(temporary.read_bytes()).hexdigest()
            if actual != SOURCE_HASHES[filename]:
                raise ValueError(f"{filename}: downloaded source does not match pinned hash")
            temporary.replace(target)
            downloaded.append(filename)
        finally:
            temporary.unlink(missing_ok=True)
    evidence = _verified_sources(sources)
    manifest = {"schema_version": "nightjar-fireflux-source-manifest-0.1",
                "verified_at_utc": datetime.now(timezone.utc).isoformat(),
                "downloaded_filenames_in_this_call": downloaded, "files": evidence,
                "artifact_kind": "raw source inventory, not a scenario or replay"}
    (sources / "source-manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def mell_rate_upper_bound(wind_2m_m_s: float, moisture_fraction: float) -> float:
    """Mell et al. (2007) Eq. 2 at W -> infinity (analytic upper bound)."""
    if not all(isfinite(x) for x in (wind_2m_m_s, moisture_fraction)) or wind_2m_m_s < 0 or not 0 <= moisture_fraction <= 1:
        raise ValueError("invalid grass-reference inputs")
    return (0.165 + 0.534 * wind_2m_m_s) * exp(-0.108 * moisture_fraction * 100)


def minimum_wind_for_rate(rate_m_s: float, moisture_fraction: float) -> float:
    """Invert the W -> infinity law for the minimum matching 2 m wind."""
    if not isfinite(rate_m_s) or rate_m_s <= 0 or not isfinite(moisture_fraction) or not 0 <= moisture_fraction <= 1:
        raise ValueError("invalid inverse-rate inputs")
    return max(0.0, (rate_m_s / exp(-0.108 * moisture_fraction * 100) - 0.165) / 0.534)


def maximum_moisture_for_rate(rate_m_s: float, wind_2m_m_s: float) -> float | None:
    """Invert the W -> infinity law for moisture matching a target rate."""
    if not all(isfinite(x) for x in (rate_m_s, wind_2m_m_s)) or rate_m_s <= 0 or wind_2m_m_s < 0:
        raise ValueError("invalid inverse-rate inputs")
    dry_rate = 0.165 + 0.534 * wind_2m_m_s
    if rate_m_s > dry_rate:
        return None
    return -log(rate_m_s / dry_rate) / 0.108 / 100


def _protocol_check(protocol_path: Path | None) -> dict | None:
    if protocol_path is None:
        return None
    protocol = json.loads(protocol_path.read_text())
    required = {
        "schema_version": "nightjar-fireflux-protocol-0.1",
        "crs": "LOCAL_TRANSECT_METERS",
        "model": "Unchanged Mell et al. 2007 Eq.2 from CSIRO diagnostic",
        "forcing": {"wind_2m_m_s": 3.0, "dead_moisture_fraction": 0.09},
        "gates": {"nmae_lt": 0.3, "absolute_bias_lt": 0.2, "physical_validation_pass": None},
        "primary_metrics": ["single interval normalized rate error and signed bias", "distance below observed reported rate", "conditional minimum transit time300/R_upper"],
        "primary_observable": "Kochanski et al. 2013 section 5.1 observed between-tower rate, 1.61 m/s",
        "primary_geometry": "300m downwind tower separation, from section 2; no closed perimeter",
        "primary_test": "Upper bound at W approaching infinity; not a finite-width forecast",
        "initialization": "First observed photo head at152s,-45m for Clements trajectories; Kochanski rate scored directly and transit300/R conditional, without inventing ST timestamp",
        "secondary_metrics": ["head residual at190s and separate195s plume endpoint", "conditional minimum tower arrival152+45/R_upper"],
        "secondary_observable": "Clements et al. 2007 p1375 head about 45m upstream at12:46:02 CST to front passage at12:46:40 CST on p1377; elapsed152 to190s from12:43:30 CST",
        "secondary_source_variants": [
            "4.5m plume temperature peak at12:46:45 CST is a different endpoint, retain separately",
            "Kochanski2013 MT thermocouple peak219s is not substituted for Clements2007 front190s",
        ],
        "sensitivity": {"finite_width_surrogate_m": 385.0, "moisture_fraction": [0.09, 0.185], "wind_2m_m_s": [3.0, 4.0]},
        "sensitivity_meaning": "3-4m/s comes from preliminary Clements2007 AMS overview;18.5% is measured mixed live+dead morning sample not dead fuel at burn;385m is ignition length not measured head width. No confidence bounds or eligible alternative fits.",
        "uncertainty_policy": "Retain unknown instrument/input/distance/time-convention errors as unknown. Quantify two-decimal rate rounding and1Hz sample discretization only as separate arithmetic sensitivities, never a confidence interval. Report required wind/moisture for matching rates instead of invented uncertainty distributions.",
        "seed": 0,
    }
    for key, value in required.items():
        if protocol.get(key) != value:
            raise ValueError(f"protocol does not match supported FireFlux diagnostic at {key}")
    if protocol.get("sources") != SOURCE_HASHES:
        raise ValueError("protocol source hashes do not match supported pinned sources")
    candidate_hash = sha256(Path(__file__).with_name("csiro_head_validation.py").read_bytes()).hexdigest()
    if protocol.get("candidate_code_sha256") != candidate_hash:
        raise ValueError("candidate code differs from protocol freeze; review required")
    return {"path_name": protocol_path.name, "sha256": sha256(protocol_path.read_bytes()).hexdigest(),
            "schema_version": protocol.get("schema_version"),
            "protocol_frozen_at_utc": protocol.get("frozen_at_utc"),
            "historical_observation_access_before_freeze": protocol.get("observation_access_before_freeze"),
            "historical_independence_note": protocol.get("independence"),
            "candidate_code_sha256_in_protocol": protocol.get("candidate_code_sha256"),
            "history_note": "A current protocol hash verifies this file only; it does not establish historical observation blindness."}


def run_diagnostic(sources: Path, protocol_path: Path | None = None) -> dict:
    source_evidence = _verified_sources(sources)
    protocol_evidence = _protocol_check(protocol_path)
    primary = mell_rate_upper_bound(**{
        "wind_2m_m_s": FORCING["wind_2m_m_s"],
        "moisture_fraction": FORCING["dead_moisture_fraction"],
    })
    sensitivity = []
    for wind in SENSITIVITY_WINDS_M_S:
        for moisture in SENSITIVITY_MOISTURES:
            upper = mell_rate_upper_bound(wind, moisture)
            finite_surrogate = cheney_mell_rate(wind, moisture, IGNITION_WIDTH_SURROGATE_M)
            sensitivity.append({"wind_2m_m_s": wind, "dead_moisture_fraction": moisture,
                                "upper_bound_m_s": upper,
                                "upper_bound_residuals_vs_reported_rate": {
                                    str(rate): {"residual_m_s": upper - rate, "nmae": abs(upper - rate) / rate,
                                                "normalized_signed_bias": (upper - rate) / rate}
                                    for rate in (OBSERVED_RATE_M_S, *OBSERVED_RATE_ROUNDING_M_S)
                                },
                                "finite_385m_ignition_length_surrogate_m_s": finite_surrogate,
                                "finite_surrogate_residuals_vs_reported_rate": {
                                    str(rate): {"residual_m_s": finite_surrogate - rate,
                                                "nmae": abs(finite_surrogate - rate) / rate,
                                                "normalized_signed_bias": (finite_surrogate - rate) / rate}
                                    for rate in (OBSERVED_RATE_M_S, *OBSERVED_RATE_ROUNDING_M_S)
                                },
                                "surrogate_interpretation": "secondary mathematical scenario only; ignition length is not observed head width"})

    residual = primary - OBSERVED_RATE_M_S
    nmae = abs(residual) / OBSERVED_RATE_M_S
    normalized_bias = residual / OBSERVED_RATE_M_S
    rounding = []
    for observed in OBSERVED_RATE_ROUNDING_M_S:
        rate_residual = primary - observed
        rounding.append({"observed_rate_m_s": observed, "rate_residual_m_s": rate_residual,
                         "nmae": abs(rate_residual) / observed, "normalized_signed_bias": rate_residual / observed,
                         "underpredicts": primary < observed})

    min_arrival = INITIAL_HEAD_TIME_S + (-INITIAL_HEAD_POSITION_M) / primary
    predicted_front_x = INITIAL_HEAD_POSITION_M + primary * (FRONT_PASSAGE_TIME_S - INITIAL_HEAD_TIME_S)
    predicted_plume_endpoint_x = INITIAL_HEAD_POSITION_M + primary * (PLUME_PEAK_TIME_S - INITIAL_HEAD_TIME_S)
    transit = TOWER_SEPARATION_M / OBSERVED_RATE_M_S
    rounded_transits = [TOWER_SEPARATION_M / rate for rate in OBSERVED_RATE_ROUNDING_M_S]
    minimum_possible_nmae = max(0.0, (OBSERVED_RATE_M_S - primary) / OBSERVED_RATE_M_S)
    observation_facts = [
        {"fact": "observed between-tower spread rate is 1.61 m/s", "kind": "published-derived observation", "source": "Kochanski et al. (2013), section 5.1, p. 1114"},
        {"fact": "Fig. 4 plotted temperature peaks appear near 219 s at MT and approximately 460 s at ST", "kind": "visual audit after protocol freeze; approximate only, not separately scored", "source": "Kochanski et al. (2013), Fig. 4, p. 1115", "interpretation": "about 240 s peak separation across 300 m implies about 1.25 m/s; this appears inconsistent with the paper's stated 1.61 m/s and needs raw-source reconciliation"},
        {"fact": "main-to-short tower separation is 300 m", "kind": "published geometry", "source": "Kochanski et al. (2013), section 2, p. 1111"},
        {"fact": "ambient wind at 2 m is 3 m/s", "kind": "published ambient forcing", "source": "Clements et al. (2007), preliminary results, p. 1375"},
        {"fact": "estimated fuel moisture at burn time is approximately 9%", "kind": "published estimate", "source": "Clements et al. (2007), preliminary results, p. 1375"},
        {"fact": "photo head is approximately 45 m from tower at 12:46:02 CST", "kind": "published approximate photo observation", "source": "Clements et al. (2007), Fig. 5 discussion, p. 1375"},
        {"fact": "front passage at 12:46:40 CST; 4.5 m plume peak at 12:46:45 CST", "kind": "published distinct endpoints", "source": "Clements et al. (2007), Figs. 5-6 discussion, pp. 1376-1377"},
        {"fact": "FireFlux MT temperature peak at 219 s", "kind": "published thermocouple peak", "source": "Kochanski et al. (2013), section 5.1, p. 1114"},
        {"fact": "3-5% of total fuel load was live", "kind": "published fuel estimate", "source": "Clements et al. (2007), fuel conditions, p. 1375"},
        {"fact": "preliminary AMS overview reports ambient wind of roughly 3-4 m/s", "kind": "published campaign overview; not per-window uncertainty", "source": "Clements (2007), AMS overview abstract, https://ams.confex.com/ams/7firenortheast/techprogram/paper_127373.htm"},
        {"fact": "mixed live+dead morning sample moisture was 18.5%", "kind": "published mixed-sample measurement", "source": "Clements et al. (2007), fuel conditions, p. 1375"},
        {"fact": "head-fire horizontal winds rose from about 3 to over 10 m/s", "kind": "published fire-affected observation", "source": "Clements et al. (2007), flow and turbulence, p. 1377; Kochanski et al. (2013), section 5.3, p. 1119"},
        {"fact": "tallgrass fuel height estimated at 1.5 m", "kind": "published fuel estimate", "source": "Kochanski et al. (2013), section 4, p. 1113"},
        {"fact": "the ignition procedure used a 385 m line", "kind": "published ignition geometry; not observed head width", "source": "Kochanski et al. (2013), section 4, p. 1113"},
    ]
    configuration = {
        "law": "Mell et al. (2007) Eq. 2 unchanged; W -> infinity analytical limit",
        "equation_constants": {"intercept_m_s": 0.165, "wind_coefficient": 0.534,
                               "width_intercept_m": -0.859, "width_wind_coefficient_s": -2.036,
                               "moisture_coefficient_per_percent": -0.108},
        "primary_forcing": FORCING,
        "sensitivity_winds_m_s": SENSITIVITY_WINDS_M_S,
        "sensitivity_moisture_fractions": SENSITIVITY_MOISTURES,
        "finite_width_secondary": {"surrogate_m": IGNITION_WIDTH_SURROGATE_M, "source": "Kochanski et al. (2013), section 4, p. 1113, ignition line length"},
        "rate_gates": RATE_GATES,
        "seed": 0,
        "source_sha256": {row["filename"]: row["sha256"] for row in source_evidence},
    }
    code_hashes = {
        "nightjar/fireflux_validation.py": sha256(Path(__file__).read_bytes()).hexdigest(),
        "nightjar/csiro_head_validation.py": sha256(Path(__file__).with_name("csiro_head_validation.py").read_bytes()).hexdigest(),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_time_origin_utc": "2006-02-23T18:43:30Z",
        "time_origin_basis": "Clements et al. (2007) ignition at 12:43:30 CST; CST interpreted explicitly as UTC-6",
        "crs": "LOCAL_TRANSECT_METERS", "units": {"position": "m", "time": "s", "rate": "m/s", "wind": "m/s", "moisture": "fraction"},
        "seed": 0, "seed_policy": "deterministic analytic equations; metadata only",
        "configuration_sha256": canonical_hash(configuration), "configuration_hash_policy": "scientific forcing/constants and source hashes only; observations and timestamps excluded",
        "observation_facts_sha256": canonical_hash({"observation_facts": observation_facts}),
        "configuration": configuration,
        "protocol": protocol_evidence,
        "sources": source_evidence,
        "runtime": {"python": platform.python_version()},
        "code_sha256": code_hashes,
        "primary": {
            "rate_m_s": primary, "observed_rate_m_s": OBSERVED_RATE_M_S,
            "bound_kind": "analytic W -> infinity upper bound; no finite-width forecast",
            "forcing": FORCING, "signed_rate_residual_m_s": residual, "nmae": nmae,
            "normalized_signed_bias": normalized_bias,
            "minimum_possible_nmae_given_upper_bound": minimum_possible_nmae,
            "signed_bias_bound_given_upper_bound": {"lower_infimum": -1.0, "upper": normalized_bias,
                                                        "definition": "(predicted-observed)/observed; finite-width rates lie between zero and the analytic upper bound"},
            "rate_rounding_sensitivity": rounding,
            "reported_rate_implied_300m_transit_s": transit,
            "reported_rate_implied_300m_transit_rate_rounding_s": {"min": min(rounded_transits), "max": max(rounded_transits)},
            "reported_rate_implied_300m_transit_sampling_only_s": {"min": transit - 2.0, "max": transit + 2.0,
                                                                         "assumption": "each of two 1 Hz endpoint times is within +/-1 s; arithmetic sensitivity only"},
            "conditional_minimum_predicted_transit_s": TOWER_SEPARATION_M / primary,
            "nmae_gate_rejected_by_upper_bound": minimum_possible_nmae >= RATE_GATES["nmae_lt"],
            "absolute_bias_gate_rejected_by_upper_bound": minimum_possible_nmae >= RATE_GATES["absolute_bias_lt"],
            "configuration_rejection_scope": "A failing upper bound rejects only the published ambient-static forcing configuration; no intrinsic rejection under actual fire-front wind forcing.",
        },
        "inverse_matching_without_tuning": {
            "minimum_wind_2m_m_s_at_9_percent_for_rate_m_s": {
                str(rate): minimum_wind_for_rate(rate, 0.09)
                for rate in (OBSERVED_RATE_M_S, *OBSERVED_RATE_ROUNDING_M_S)
            },
            "maximum_dead_moisture_fraction_matching_rate": {
                str(rate): {str(wind): maximum_moisture_for_rate(rate, wind) for wind in SENSITIVITY_WINDS_M_S}
                for rate in (OBSERVED_RATE_M_S, *OBSERVED_RATE_ROUNDING_M_S)
            },
        },
        "sensitivity_variants": sensitivity,
        "secondary_photo_trajectory": {
            "anchor": {"time_s": INITIAL_HEAD_TIME_S, "head_position_m": INITIAL_HEAD_POSITION_M,
                       "observation_kind": "published approximate head position from photo"},
            "front_passage_endpoint": {"time_s": FRONT_PASSAGE_TIME_S, "predicted_head_max_x_m": predicted_front_x,
                                       "observed_tower_coordinate_m": 0.0, "position_residual_m": predicted_front_x,
                                       "arrival_time_minimum_s": min_arrival,
            "interpretation": "unreset constant-rate trajectory from first observed head; interval overlaps final 11 s of eastern walking ignition, so not an independent free-spread test"},
            "plume_peak_endpoint_variant": {"time_s": PLUME_PEAK_TIME_S, "predicted_head_max_x_m": predicted_plume_endpoint_x,
                                             "endpoint_kind": "4.5 m plume temperature peak, not ground arrival"},
            "kochanski_mt_peak_time_s": KOCHANSKI_MT_PEAK_TIME_S,
            "source_time_discrepancies_s": {"kochanski_mt_peak_minus_clements_plume_peak": KOCHANSKI_MT_PEAK_TIME_S - PLUME_PEAK_TIME_S,
                                             "kochanski_mt_peak_minus_clements_front_passage": KOCHANSKI_MT_PEAK_TIME_S - FRONT_PASSAGE_TIME_S},
        },
        "observation_facts": observation_facts,
        "limitations": [
            "This retrospective upper-bound calculation is not a blind validation and does not establish physical_validation_pass.",
            "The reported 2 m wind is an ambient tower value and is not established to match the law's open 2 m wind convention. Fire-induced horizontal winds exceeded 10 m/s; this result cannot intrinsically reject the law under actual fire-front wind.",
            "The analytic upper bound assumes W tends to infinity. No observed head width is available; the 385 m ignition-line length used in the secondary scenario is not a head width.",
            "The 18.5% value is a mixed live+dead morning sample, not dead fuel moisture at burn; no moisture tuning or cure correction is applied. Clements et al. report 3-5% live fuel, so cured grass is an approximation.",
            "The published 3-4 m/s preliminary AMS overview is not per-window uncertainty. No confidence intervals are calculated.",
            "Two-decimal rate rounding and +/-1 s endpoint sampling effects are arithmetic sensitivities only. Instrument response, observation mapping, distance errors, and time conventions remain unknown.",
            "No midflame conversion, FM1, or Anderson mapping is supported here. The geometry transit is conditional and does not create an observed ST timestamp.",
            "Clements photo head, ground front passage, 4.5 m plume peak, and Kochanski MT temperature peak remain distinct observations and endpoints.",
            "Kochanski's published 1.61 m/s statement conflicts with the apparent Fig. 4 MT/ST temperature-peak timing (about 219/460 s). The reported-rate-implied 300 m transit is arithmetic, not an independently observed arrival; raw tower records must reconcile the discrepancy before any acceptance claim.",
            "This is one longitudinal rate diagnostic, not a perimeter, flank, spotting, suppression, or transfer validation.",
        ],
        "gates": {**RATE_GATES, "physical_validation_pass": None},
        "physical_validation_pass": None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, default=Path("data/raw/fireflux"))
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fetch-sources", action="store_true", help="explicitly download and verify pinned FireFlux PDFs")
    args = parser.parse_args()
    if args.fetch_sources:
        fetch_sources(args.sources)
    report = run_diagnostic(args.sources, args.protocol)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
