"""Deterministic, synthetic one-aircraft movement and water accounting.

This module does not simulate fire, aviation safety, or water effectiveness.
"""

from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
from math import hypot, isfinite


CAPACITY_L = 6137.0  # CL-415 reference capacity; see docs/PHYSICAL_MODEL_V0.md


def tank_capacity_l(aircraft: dict) -> float:
    """Tank capacity of an observed aircraft; observations without one are CL-415s."""
    return float(aircraft.get("capacity_l", CAPACITY_L))


def _point(value: object, name: str) -> list[float]:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{name} must be [x_m, y_m]")
    point = [float(v) for v in value]
    if not all(isfinite(v) for v in point):
        raise ValueError(f"{name} must contain finite coordinates")
    return point


def _positive(value: object, name: str, *, allow_zero: bool = False) -> float:
    number = float(value)
    if not isfinite(number) or number < 0 or (number == 0 and not allow_zero):
        raise ValueError(f"{name} must be {'nonnegative' if allow_zero else 'positive'} and finite")
    return number


def validate_scenario(scenario: dict) -> dict:
    """Return a normalized copy of the versioned synthetic scenario."""
    if scenario.get("schema_version") != "0.1":
        raise ValueError("scenario schema_version must be '0.1'")
    if scenario.get("provenance") != "synthetic":
        raise ValueError("v0.1 requires provenance='synthetic'")
    if not isinstance(scenario.get("scenario_id"), str) or not scenario["scenario_id"]:
        raise ValueError("scenario_id must be a nonempty string")
    if not isinstance(scenario.get("seed"), int) or isinstance(scenario["seed"], bool):
        raise ValueError("seed must be an integer")
    if scenario.get("crs") != "LOCAL_METERS":
        raise ValueError("v0.1 requires crs='LOCAL_METERS'")
    out = deepcopy(scenario)
    out["horizon_s"] = _positive(out["horizon_s"], "horizon_s")
    out["decision_interval_s"] = _positive(out["decision_interval_s"], "decision_interval_s")
    out["aircraft"]["initial_position_m"] = _point(out["aircraft"]["initial_position_m"], "initial_position_m")
    out["water_source_m"] = _point(out["water_source_m"], "water_source_m")
    for key in ("loaded_ground_speed_m_s", "empty_ground_speed_m_s", "refill_time_s"):
        out["aircraft"][key] = _positive(out["aircraft"][key], key)
    out["aircraft"]["initial_water_l"] = _positive(out["aircraft"]["initial_water_l"], "initial_water_l", allow_zero=True)
    if out["aircraft"]["initial_water_l"] > CAPACITY_L:
        raise ValueError("initial_water_l exceeds CL-415 reference capacity")
    return out


def run_episode(scenario: dict, orders: list[dict]) -> dict:
    """Run one episode; orders are indexed by decision number.

    A busy aircraft rejects new missions. HOLD never interrupts a mission.
    At an arrival, the current phase completes before remaining interval time advances.
    """
    config = validate_scenario(scenario)
    aircraft = config["aircraft"]
    state = {
        "position_m": aircraft["initial_position_m"].copy(),
        "water_l": aircraft["initial_water_l"],
        "phase": "idle",
        "target_m": None,
        "refill_remaining_s": 0.0,
    }
    events: list[dict] = []
    decisions: list[dict] = []
    time_s = 0.0
    flight_s = 0.0
    dropped_l = 0.0
    rejected = 0

    def event(kind: str, at_s: float, **details: object) -> None:
        events.append({"time_s": round(at_s, 6), "kind": kind, **details})

    index = 0
    while time_s < config["horizon_s"] - 1e-9:
        observation = {
            "time_s": round(time_s, 6),
            "aircraft": deepcopy(state),
            "water_source_m": config["water_source_m"].copy(),
        }
        order = orders[index] if index < len(orders) else {"type": "HOLD"}
        if not isinstance(order, dict):
            raise ValueError("each order must be an object")
        kind = order.get("type")
        accepted = True
        reason = None
        if kind == "HOLD":
            pass
        elif kind not in ("DROP", "RETURN_TO_REFILL"):
            accepted, reason = False, "unknown_order"
        elif state["phase"] != "idle":
            accepted, reason = False, "aircraft_busy"
        elif kind == "DROP":
            if state["water_l"] <= 0:
                accepted, reason = False, "no_water"
            else:
                target = _point(order.get("target_m"), "target_m")
                state["target_m"] = target
                state["phase"] = "to_drop"
        elif state["water_l"] >= CAPACITY_L:
            accepted, reason = False, "tank_full"
        else:
            state["target_m"] = config["water_source_m"].copy()
            state["phase"] = "to_water"
        if not accepted:
            rejected += 1
            event("order_rejected", time_s, reason=reason)
        decisions.append({"observation": observation, "order": deepcopy(order), "accepted": accepted, "reason": reason})

        remaining = min(config["decision_interval_s"], config["horizon_s"] - time_s)
        while remaining > 1e-9:
            phase = state["phase"]
            if phase == "idle":
                time_s += remaining
                remaining = 0.0
            elif phase == "refilling":
                elapsed = min(remaining, state["refill_remaining_s"])
                time_s += elapsed
                remaining -= elapsed
                state["refill_remaining_s"] -= elapsed
                if state["refill_remaining_s"] <= 1e-9:
                    state["water_l"] = CAPACITY_L
                    state["refill_remaining_s"] = 0.0
                    state["phase"] = "idle"
                    event("refill_complete", time_s, water_l=CAPACITY_L)
            else:
                target = state["target_m"]
                distance = hypot(target[0] - state["position_m"][0], target[1] - state["position_m"][1])
                speed_key = "loaded_ground_speed_m_s" if state["water_l"] > 0 else "empty_ground_speed_m_s"
                speed = aircraft[speed_key]
                elapsed = min(remaining, distance / speed)
                if distance > 0:
                    fraction = min(1.0, speed * elapsed / distance)
                    state["position_m"] = [p + fraction * (t - p) for p, t in zip(state["position_m"], target)]
                time_s += elapsed
                remaining -= elapsed
                flight_s += elapsed
                if elapsed >= distance / speed - 1e-9:
                    state["position_m"] = target.copy()
                    state["target_m"] = None
                    if phase == "to_drop":
                        amount = state["water_l"]
                        state["water_l"] = 0.0
                        dropped_l += amount
                        state["phase"] = "idle"
                        event("drop", time_s, position_m=target.copy(), water_l=amount)
                    else:
                        state["phase"] = "refilling"
                        state["refill_remaining_s"] = aircraft["refill_time_s"]
                        event("refill_started", time_s)
        index += 1
    return {
        "replay_schema_version": "0.1",
        "scenario_id": config["scenario_id"],
        "seed": config["seed"],
        "scenario_sha256": sha256(json.dumps(scenario, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "provenance": "synthetic",
        "crs": config["crs"],
        "units": {"position": "m", "time": "s", "water": "L"},
        "decisions": decisions,
        "events": events,
        "final_state": state,
        "metrics": {"flight_s": round(flight_s, 6), "water_dropped_l": dropped_l, "rejected_orders": rejected},
    }
