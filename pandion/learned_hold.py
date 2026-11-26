"""Learned effect of water on a cell, fitted to outcomes of flights (no fire formula, no hold limit; D-98).

A small network predicts the probability that a cell bordering the fire has burned by the end of the
horizon from observable values only: brightness of the bordering fire, terrain, wind, fuel, moisture,
time left and the drops already on the cell. Weights are read from an ``.npz`` in a run directory.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

from .environment import MOISTURE_KEYS

FEATURES = ("heat_max", "heat_mean", "burning_share", "slope_to", "wind_to", "wind_speed", "time_left", "drops") + \
    tuple(MOISTURE_KEYS) + tuple(f"fuel_{k}" for k in range(7))
_DROPS = FEATURES.index("drops")


@lru_cache(maxsize=4)
def load_hold(path: str) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {k: data[k] for k in data.files}


def burn_probability(rows: np.ndarray, path: str | Path) -> np.ndarray:
    """[N] probability of burning for feature rows [N, len(FEATURES)]."""
    net = load_hold(str(path))
    x = ((rows - net["mu"]) / net["sd"]).astype(np.float32)
    for layer in (0, 1):
        x = x @ net[f"{layer}/w"].T + net[f"{layer}/b"]
        x = x / (1.0 + np.exp(-x))  # SiLU
    logit = (x @ net["2/w"].T + net["2/b"])[:, 0]
    return 1.0 / (1.0 + np.exp(-np.clip(logit, -30, 30)))


def risk_before_and_after(rows: np.ndarray, path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Burn probability with the drops now on each cell and with one more (at most three)."""
    after = rows.copy()
    after[:, _DROPS] = np.minimum(rows[:, _DROPS] + 1.0 / 3.0, 1.0)
    return burn_probability(rows, path), burn_probability(after, path)
