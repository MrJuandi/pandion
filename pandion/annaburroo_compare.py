"""No-fit external diagnostic of fixed Anderson FM1 against Annaburroo plots."""

from __future__ import annotations

import argparse
import csv
from hashlib import sha256
import json
from pathlib import Path
from statistics import median

from pandion.rothermel import fm1_components


EXPECTED_SHA256 = "4425d11fe970e263d400b7a9d089b8eea70bda4de4a1e29cbeedea28bc727934"


def compare(table: Path) -> dict:
    raw = table.read_bytes()
    digest = sha256(raw).hexdigest()
    if digest != EXPECTED_SHA256:
        raise ValueError("Annaburroo plot table differs from the audited version")
    rows = list(csv.DictReader(raw.decode("utf-8-sig").splitlines()))
    records = []
    excluded = []
    for row in rows:
        fire_id = row["FireID"]
        try:
            wind = float(row["MidFlwind.ms"])
            moisture = float(row["DFMC.%"]) / 100
            observed = float(row["RSS.ms"])
        except ValueError:
            excluded.append({"fire_id": fire_id, "reason": "missing or nonnumeric wind, moisture or RSS"})
            continue
        if wind < 0 or not 0 <= moisture <= 1 or observed <= 0:
            excluded.append({"fire_id": fire_id, "reason": "input outside declared domain"})
            continue
        pieces = fm1_components(moisture, wind)
        predicted = pieces["r0_m_s"] * (1 + pieces["phi_w"])
        records.append({"fire_id": fire_id, "fuel_type": row["FuelType"],
                        "curing_percent": float(row["Cure.%"]),
                        "wind_m_s": wind, "dead_moisture_fraction": moisture,
                        "observed_rss_m_s": observed, "predicted_fm1_m_s": predicted,
                        "error_m_s": predicted - observed})
    if not records:
        raise ValueError("no comparable rows")
    errors = [r["error_m_s"] for r in records]
    absolute_errors = [abs(e) for e in errors]
    by_type = {}
    for fuel_type in sorted({r["fuel_type"] for r in records}):
        subset = [r for r in records if r["fuel_type"] == fuel_type]
        by_type[fuel_type] = {"n": len(subset),
                              "mae_m_s": sum(abs(r["error_m_s"]) for r in subset) / len(subset),
                              "bias_m_s": sum(r["error_m_s"] for r in subset) / len(subset)}
    return {"schema_version": "annaburroo-fm1-diagnostic-0.1",
            "source_table_sha256": digest, "selection": "all line-ignition plot averages with numeric midflame wind, dead fuel moisture and positive RSS",
            "prediction": "unmodified Anderson FM1, flat terrain, reported midflame wind and dead fuel moisture; no fitted parameters",
            "target": "RSS.ms, reported width-adjusted steady-state spread rate",
            "source_rows": len(rows), "included": len(records), "excluded": excluded,
            "mae_m_s": sum(absolute_errors) / len(records),
            "median_absolute_error_m_s": median(absolute_errors),
            "bias_m_s": sum(errors) / len(records),
            "mean_observed_m_s": sum(r["observed_rss_m_s"] for r in records) / len(records),
            "mean_predicted_m_s": sum(r["predicted_fm1_m_s"] for r in records) / len(records),
            "by_fuel_type": by_type, "records": records}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", type=Path, default=Path("data/raw/annaburroo/Plot_Average_Data.csv"))
    parser.add_argument("--output", type=Path, default=Path("runs/annaburroo-fm1-diagnostic.json"))
    args = parser.parse_args()
    result = compare(args.table)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "records"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
