"""Audit a published filtered FireFlux I observation export, without fitting.

Simulation arrays in the same archive never enter this report or the spread
diagnostic. Column-to-height mapping and absolute clocks remain unresolved.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import tempfile

SOURCE_FILES = {
    "MainTower_Anel_25m.npz": {
        "url": "https://zenodo.org/api/records/6372268/files/MainTower_Anel_25m.npz/content",
        "sha256": "2c576a05c2c15d1b3fb322455b411038466ce04540ce1b6989798903b2443321",
        "publisher_md5": "e95a3d64ea2759e85ebdcd5eaae78e5a",
    },
    "costes2022.pdf": {
        "url": "https://www.umr-cnrm.fr/old/IMG/pdf/costes_etal_2022_james.pdf",
        "sha256": "080dc7859c7341829c01671811e24cfb31f1e96efff0b7a930c2ec860370706f",
    },
}
OBSERVATION_COLUMNS = {"ObsTInterp": 6, "ObsHWSInterp": 4, "ObsWInterp": 4}


def verify_sources(root: Path) -> list[dict]:
    sources = []
    for name, source in SOURCE_FILES.items():
        actual = sha256((root / name).read_bytes()).hexdigest()
        if actual != source["sha256"]:
            raise ValueError(f"{name}: source hash changed; review required")
        sources.append({"filename": name, **source})
    return sources


def fetch_sources(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    downloaded = []
    for name, source in SOURCE_FILES.items():
        target = root / name
        if target.exists():
            if sha256(target.read_bytes()).hexdigest() != source["sha256"]:
                raise ValueError(f"{name}: refusing to overwrite changed source")
            continue
        with tempfile.NamedTemporaryFile(dir=root, delete=False) as handle:
            temporary = Path(handle.name)
        try:
            subprocess.run(["curl", "--proto", "=https", "--location", "--fail", "--silent", "--show-error",
                            "--output", str(temporary), source["url"]], check=True)
            if sha256(temporary.read_bytes()).hexdigest() != source["sha256"]:
                raise ValueError(f"{name}: downloaded source hash changed")
            temporary.replace(target)
            downloaded.append(name)
        finally:
            temporary.unlink(missing_ok=True)
    (root / "filtered-source-manifest.json").write_text(json.dumps({
        "schema_version": "nightjar-fireflux-filtered-sources-0.1",
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "downloaded_filenames_in_this_call": downloaded, "files": verify_sources(root),
    }, indent=2) + "\n")


def audit(root: Path) -> dict:
    import numpy as np

    sources = verify_sources(root)
    statistics = {}
    with np.load(root / "MainTower_Anel_25m.npz", allow_pickle=False) as archive:
        time = archive["TimeMean"]
        if time.ndim != 1 or len(time) < 2 or not np.isfinite(time).all() or not np.all(np.diff(time) > 0):
            raise ValueError("invalid centered observation time axis")
        selected = {"TimeMean"}
        for key, columns in OBSERVATION_COLUMNS.items():
            values, variability = archive[key], archive[key + "Std"]
            selected.update((key, key + "Std"))
            if values.shape != (len(time), columns) or variability.shape != values.shape:
                raise ValueError(f"{key}: unexpected observation shape")
            if not np.isfinite(values).all() or not np.isfinite(variability).all() or np.any(variability < 0):
                raise ValueError(f"{key}: invalid observations or temporal variability")
            statistics[key] = {
                "shape": list(values.shape), "column_height_mapping": None,
                "minimum_by_column": values.min(axis=0).tolist(),
                "maximum_by_column": values.max(axis=0).tolist(),
                "maximum_5s_temporal_standard_deviation_by_column": variability.max(axis=0).tolist(),
            }
        excluded = sorted(set(archive.files) - selected)
    return {
        "schema_version": "nightjar-fireflux-filtered-audit-0.1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "seed": 0,
        "crs": "UNRESOLVED_TOWER_COLUMN_COORDINATES", "time_origin_utc": None,
        "units": {"time": "s relative to processed main-tower passage", "temperature": "degC", "wind": "m/s"},
        "source_publication_date": "2022-03-21", "source_doi": "10.5281/zenodo.6372268",
        "source_license": "CC-BY-4.0", "sources": sources,
        "method_source": "Costes et al. 2022, DOI 10.1029/2021MS002955, section 5.1",
        "time_axis": {"count": len(time), "first_s": float(time[0]), "last_s": float(time[-1]),
                      "minimum_step_s": float(np.diff(time).min()), "maximum_step_s": float(np.diff(time).max())},
        "observation_statistics": statistics, "excluded_arrays": excluded,
        "code_sha256": sha256(Path(__file__).read_bytes()).hexdigest(), "numpy_version": np.__version__,
        "physical_validation_pass": None,
        "limitations": [
            "These are Hodrick-Prescott filtered observations, not raw instrument samples. Std arrays are 5-second temporal variability, not instrumental error or confidence intervals.",
            "Only numeric observation arrays and the centered time axis are loaded; simulated ROS, ArrivalTime and ensemble arrays are excluded. Pickled failed-sensor index arrays are never loaded.",
            "Failed-sensor masks remain unresolved. Finite interpolated/filtered values do not establish valid measurements at every time; archive statistics are not eligible forcing or a completeness claim.",
            "The export contains main-tower records only. Its centered time axis does not resolve absolute clocks or short-tower arrivals, and its array columns do not explicitly identify measurement heights.",
            "No statistic here is used as candidate forcing or as a new evaluation outcome. Post-freeze source inspection does not change the frozen ambient-wind diagnostic.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, default=Path("data/raw/fireflux"))
    parser.add_argument("--fetch-sources", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.fetch_sources:
        fetch_sources(args.sources)
    report = audit(args.sources)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
