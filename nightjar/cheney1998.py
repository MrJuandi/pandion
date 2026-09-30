"""Reconstruction of the Cheney et al. (1998) grass head-fire rate model.

The equations are reconstructed from the corrected full coauthor restatement
in Cruz et al. (2015), section 2.1.2, Eqs. 6–11, DOI
10.1080/00049158.2015.1055063 (corrigendum DOI 10.1080/00049158.2015.1080789).
The original 1998 PDF was not acquired, so this module does not claim direct
verification against that article. Cruz et al. (2022), Appendix A.3, later
prints 0.705 for the cut/grazed wind coefficient (rather than the corrected
2015 value 0.715), and explicitly uses the later 2015 curing function; neither
later change is included here.

This module is a pure local potential-rate calculation. It does not import
production propagation or controller code. Inputs are open 10 m wind in km/h,
dead fuel moisture as percent of oven-dry mass, and curing as percent.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real


MODULE_VERSION = "cheney1998-reconstruction-0.1"

_PASTURE_CONDITIONS = frozenset({"undisturbed", "cut_grazed", "eaten_out"})


@dataclass(frozen=True)
class GrassInputs:
    """Explicit inputs for the grass potential head-rate equation.

    ``open_wind_10m_km_h`` is open wind at 10 m (km/h); moisture and curing
    are percentages, with moisture expressed on an oven-dry-mass basis.
    """

    open_wind_10m_km_h: float
    dead_moisture_percent: float
    curing_percent: float
    pasture_condition: str

    def __post_init__(self) -> None:
        values = (
            ("open_wind_10m_km_h", self.open_wind_10m_km_h, 0.0, 80.0),
            ("dead_moisture_percent", self.dead_moisture_percent, 2.0, 24.0),
            ("curing_percent", self.curing_percent, 50.0, 100.0),
        )
        for name, value, lower, upper in values:
            if isinstance(value, bool) or not isinstance(value, Real):
                raise ValueError(f"{name} must be a finite real number")
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            if not lower <= value <= upper:
                raise ValueError(f"{name} must be in [{lower:g}, {upper:g}]")
        if self.pasture_condition not in _PASTURE_CONDITIONS:
            raise ValueError(
                "pasture_condition must be 'undisturbed', 'cut_grazed', or 'eaten_out'"
            )
        if self.pasture_condition == "eaten_out" and self.open_wind_10m_km_h <= 5:
            raise ValueError("eaten_out has no supported low-wind branch (U must be > 5)")


def grass_rate(inputs: GrassInputs) -> dict[str, float | bool]:
    """Calculate the reconstructed potential grass head-fire rate.

    The result contains the unmodified moisture factor for auditability and
    the nonnegative factor actually used. ``rate_m_s`` is in metres/second.
    """
    if not isinstance(inputs, GrassInputs):
        raise TypeError("inputs must be a GrassInputs instance")

    wind = float(inputs.open_wind_10m_km_h)
    moisture = float(inputs.dead_moisture_percent)
    curing = float(inputs.curing_percent)
    pasture = inputs.pasture_condition

    if wind <= 5:
        wind_coefficient = 0.269 if pasture == "undisturbed" else 0.209
        base_rate = 0.054 + wind_coefficient * wind
    else:
        coefficient, exponent_coefficient = {
            "undisturbed": (1.4, 0.838),
            "cut_grazed": (1.1, 0.715),
            "eaten_out": (0.55, 0.357),
        }[pasture]
        base_rate = coefficient + exponent_coefficient * (wind - 5) ** 0.844

    if moisture < 12:
        raw_moisture_factor = math.exp(-0.108 * moisture)
    elif wind < 10:
        raw_moisture_factor = 0.684 - 0.0342 * moisture
    else:
        raw_moisture_factor = 0.547 - 0.0228 * moisture
    moisture_factor = max(raw_moisture_factor, 0.0)
    curing_factor = 1.120 / (1.0 + 59.2 * math.exp(-0.124 * (curing - 50.0)))
    rate_m_s = base_rate * moisture_factor * curing_factor / 3.6

    return {
        "base_rate_km_h": base_rate,
        "raw_moisture_factor": raw_moisture_factor,
        "moisture_factor": moisture_factor,
        "curing_factor": curing_factor,
        "rate_m_s": rate_m_s,
        "moisture_floor_applied": raw_moisture_factor < 0.0,
    }


def estimate_dead_moisture_percent(temp_c: float, rh_percent: float) -> float:
    """Estimate dead moisture from temperature and relative humidity.

    This optional published estimate is separate from :func:`grass_rate`; it
    is neither clamped nor automatically fed to the model.
    """
    for name, value in (("temp_c", temp_c), ("rh_percent", rh_percent)):
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError(f"{name} must be a finite real number")
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
    if not 0 <= rh_percent <= 100:
        raise ValueError("rh_percent must be in [0, 100]")
    return 9.58 - 0.205 * temp_c + 0.138 * rh_percent
