"""Score matched RxCADRE predictions; no data reconstruction or model fitting."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
import platform


def _number(value, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("measurements must be numeric")
    if not isfinite(value) or value < 0 or (positive and value == 0):
        raise ValueError("measurements must be finite and nonnegative; observed rates must be positive")
    return float(value)


def score(observations: dict, predictions: dict) -> dict:
    """Evaluate every preselected sample, without pooling independent burns.

    These are metric checks, not a physical-validity certificate. Input coverage,
    frozen selection, numerical verification and uncertainty need separate review.
    """
    if observations.get("schema_version") != "rxcadre-observations-0.1":
        raise ValueError("unsupported observation schema")
    if predictions.get("schema_version") != "rxcadre-predictions-0.1":
        raise ValueError("unsupported prediction schema")
    for field in ("crs", "time_origin_utc", "seed", "units"):
        if field not in observations or predictions.get(field) != observations[field]:
            raise ValueError(f"matching {field} metadata required")
    if observations["units"] != {"rate": "m/s", "arrival": "s", "position": "m"}:
        raise ValueError("units must be m/s, s, m")
    origin = datetime.fromisoformat(observations["time_origin_utc"].replace("Z", "+00:00"))
    if origin.utcoffset() is None or origin.utcoffset().total_seconds() != 0:
        raise ValueError("time origin must be UTC")
    if type(observations["seed"]) is not int or not observations["crs"]:
        raise ValueError("integer seed and explicit CRS required")
    config_hash = predictions.get("frozen_configuration_sha256", "")
    if len(config_hash) != 64 or any(c not in "0123456789abcdef" for c in config_hash):
        raise ValueError("frozen configuration SHA-256 required")
    obs_hash = sha256(json.dumps(observations, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if predictions.get("observations_sha256") != obs_hash:
        raise ValueError("predictions must identify the exact observation contract")

    def keyed(records):
        result = {}
        for record in records:
            key = (record["burn_id"], record["sample_id"])
            if not all(isinstance(v, str) and v for v in key) or key in result:
                raise ValueError("sample keys must be unique nonempty strings")
            result[key] = record
        if not result:
            raise ValueError("empty sample set")
        return result

    observed, predicted = keyed(observations["records"]), keyed(predictions["records"])
    if observed.keys() != predicted.keys():
        raise ValueError("prediction keys must exactly match all observed samples; no omissions or extras")
    burns = {}
    paired = []
    for key, obs in observed.items():
        pred = predicted[key]
        # A wind-aligned head-rate estimate cannot be substituted for a flank
        # rate or an ignition-to-sensor average without declaring that mismatch.
        for field in ("window_start_s", "window_end_s", "direction_deg", "rate_definition"):
            if field not in obs or pred.get(field) != obs[field]:
                raise ValueError(f"sample {key}: matching {field} required")
        start, end = _number(obs["window_start_s"]), _number(obs["window_end_s"])
        if end <= start or not obs["rate_definition"]:
            raise ValueError("positive observation window and rate definition required")
        direction = _number(obs["direction_deg"])
        if direction >= 360:
            raise ValueError("direction must be in [0, 360)")
        if not obs.get("source") or obs.get("split") not in ("development", "evaluation"):
            raise ValueError("source and explicit development/evaluation split required")
        rate = _number(obs["rate_m_s"], positive=True)
        estimate = _number(pred["rate_m_s"])
        lo, hi = (_number(v) for v in obs["rate_interval_m_s"])
        if not lo <= rate <= hi:
            raise ValueError("observed rate must lie in its stated uncertainty interval")
        arrival = _number(obs["arrival_s"])
        if not start <= arrival <= end:
            raise ValueError("observed arrival must be in the declared measurement window")
        position = _number(obs["front_position_m"])
        pa, pp = _number(pred["arrival_s"]), _number(pred["front_position_m"])
        row = {"burn_id": key[0], "sample_id": key[1], "observed_rate_m_s": rate,
               "predicted_rate_m_s": estimate, "rate_error_m_s": estimate - rate,
               "distance_to_rate_interval_m_s": max(lo - estimate, estimate - hi, 0.0),
               "arrival_error_s": pa - arrival, "front_position_error_m": pp - position}
        paired.append(row)
        group = burns.setdefault(key[0], {"split": obs["split"], "rows": []})
        if group["split"] != obs["split"]:
            raise ValueError("a burn cannot occur in both splits")
        group["rows"].append(row)
    summaries = {}
    for burn, group in burns.items():
        rows = group["rows"]
        n = len(rows)
        mean_rate = sum(r["observed_rate_m_s"] for r in rows) / n
        nmae = sum(abs(r["rate_error_m_s"]) for r in rows) / n / mean_rate
        bias = sum(r["rate_error_m_s"] for r in rows) / n / mean_rate
        summaries[burn] = {"split": group["split"], "samples": n, "normalized_rate_mae": nmae,
                           "normalized_rate_bias": bias, "rate_criteria_pass": nmae < 0.30 and abs(bias) < 0.20,
                           "arrival_mae_s": sum(abs(r["arrival_error_s"]) for r in rows) / n,
                           "front_position_mae_m": sum(abs(r["front_position_error_m"]) for r in rows) / n,
                           "rate_interval_distance_mean_m_s": sum(r["distance_to_rate_interval_m_s"] for r in rows) / n}
    evaluation = [r for r in summaries.values() if r["split"] == "evaluation"]
    development = [r for r in summaries.values() if r["split"] == "development"]
    return {"schema_version": "rxcadre-metrics-0.1", "created_at_utc": datetime.now(timezone.utc).isoformat(),
            **{k: observations[k] for k in ("crs", "units", "seed", "time_origin_utc")},
            "observations_sha256": obs_hash, "frozen_configuration_sha256": config_hash,
            "scorer_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
            "python_version": platform.python_version(),
            "thresholds": {"normalized_rate_mae_lt": 0.30, "absolute_normalized_rate_bias_lt": 0.20},
            "minimum_split_coverage": len(development) >= 1 and len(evaluation) >= 2,
            "all_evaluation_rate_criteria_pass": bool(evaluation) and all(r["rate_criteria_pass"] for r in evaluation),
            "physical_validation_status": "not established by metrics alone",
            "by_burn": summaries, "paired_records": paired}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = score(json.loads(args.observations.read_text()), json.loads(args.predictions.read_text()))
    report["input_file_sha256"] = {name: sha256(path.read_bytes()).hexdigest() for name, path in
                                 (("observations", args.observations), ("predictions", args.predictions))}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "paired_records"}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
