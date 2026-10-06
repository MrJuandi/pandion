"""Content identity and incident grouping, independent of generation clocks.

Legacy scenario_hash remains a digest of the complete original JSON payload.
Content digests are additional identities; they never replace legacy validation.
"""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
from typing import Any

IDENTITY_VERSION = "nightjar-content-0.1"


def without_generation_times(value: Any) -> Any:
    """Remove generation clocks, retaining acquisition/delivery timestamps."""
    if isinstance(value, dict):
        return {key: without_generation_times(item) for key, item in value.items()
                if key not in ("created_at_utc", "generated_at_utc")}
    if isinstance(value, list):
        return [without_generation_times(item) for item in value]
    return value


def content_hash(value: Any) -> str:
    payload = without_generation_times(value)
    if isinstance(payload, dict):
        payload.pop("scenario_hash", None)
    return sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def incident_group(activation: str) -> str:
    """Group AOIs from the same CEMS activation into one independence unit."""
    return activation.split("_AOI", 1)[0]


def partition_holdout(scenarios: list[dict], activation: str) -> tuple[list[dict], list[dict]]:
    group = incident_group(activation)
    held = [s for s in scenarios if incident_group(s["activation_id"]) == group]
    training = [s for s in scenarios if incident_group(s["activation_id"]) != group]
    if not held or not training:
        raise ValueError(f"holdout activation {activation} must split the development set by incident group")
    return training, held


def scenario_entry(scenario: dict) -> dict:
    """Report content, lineage and spatial extent without exposing fire rasters."""
    grid, provenance = scenario["grid"], scenario.get("provenance", {})
    entry = {"scenario_id": scenario["scenario_id"], "scenario_hash": scenario["scenario_hash"],
             "content_sha256": content_hash(scenario), "source_kind": scenario["source_kind"],
             "activation_id": scenario["activation_id"],
             "incident_group": incident_group(scenario["activation_id"]),
             "source_zip_sha256": provenance.get("source_zip_sha256"),
             "source_product_code": provenance.get("source_product_code"),
             "crs": deepcopy(scenario.get("crs")), "grid": deepcopy(grid),
             "lineage": {key: value for key, value in provenance.items()
                         if "parent" in key and "hash" in key}}
    return entry


def audit_independence(splits: dict[str, list[dict]]) -> list[str]:
    """Find incident, content and lineage sharing across experiment partitions."""
    issues = []
    owners: dict[tuple[str, str], str] = {}
    for split, entries in splits.items():
        for entry in entries:
            tokens = [("scenario", entry.get("scenario_hash")), ("content", entry.get("content_sha256")),
                      ("incident", entry.get("incident_group")), ("source_product", entry.get("source_product_code"))]
            tokens += [("lineage", v) for v in entry.get("lineage", {}).values() if isinstance(v, str)]
            for kind, value in tokens:
                if not value:
                    continue
                key = (kind, value)
                previous = owners.setdefault(key, split)
                if previous != split:
                    issues.append(f"{kind} {value} shared by {previous} and {split}")
    return sorted(set(issues))
