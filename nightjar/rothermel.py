"""Rothermel/Albini one-size dead-fuel surface spread calculations.

Equations: Andrews (2018), USDA Forest Service RMRS-GTR-371, table 3 and
section 6. ``fm1_components`` uses Anderson (1982), INT-GTR-122 FM1 properties;
``dead_fuel_components`` accepts SI properties for one dead fuel class. Both
retain the standard particle-density and mineral-content assumptions below.
This is a numerical implementation of published component equations, not field
validation.
"""

from __future__ import annotations

from math import exp, hypot, isfinite, sqrt


FT_PER_M = 3.280839895013123
FT_MIN_PER_M_S = FT_PER_M * 60
M_S_PER_FT_MIN = 1 / FT_MIN_PER_M_S

# Unit factors used when mapping the public SI interface onto the source
# equations' imperial units. Heat uses the international BTU (1055.05585262 J)
# and avoirdupois pound (0.45359237 kg): 1 kJ/kg = 0.4299226139294927 BTU/lb.
KG_M2_TO_LB_FT2 = 0.204816143622
KJ_KG_TO_BTU_LB = 0.4299226139294927

# Anderson fuel model 1: a single dead 1-hour fuel class.
FM1_LOAD_LB_FT2 = 0.74 * 2000 / 43560
FM1_DEPTH_FT = 1.0
FM1_SAV_FT_INV = 3500.0
FM1_MOISTURE_EXTINCTION = 0.12
HEAT_BTU_LB = 8000.0
PARTICLE_DENSITY_LB_FT3 = 32.0
TOTAL_MINERAL_FRACTION = 0.0555
EFFECTIVE_MINERAL_FRACTION = 0.01


def _dead_fuel_components_imperial(
    moisture_fraction: float,
    midflame_wind_m_s: float,
    slope: float,
    *,
    load_lb_ft2: float,
    depth_ft: float,
    sav_ft_inv: float,
    moisture_extinction_fraction: float,
    heat_btu_lb: float,
) -> dict[str, float]:
    if not isfinite(moisture_fraction) or not 0 <= moisture_fraction <= 1:
        raise ValueError("dead-fuel moisture must be a fraction in [0, 1]")
    if not isfinite(midflame_wind_m_s) or not isfinite(slope) or midflame_wind_m_s < 0 or slope < 0:
        raise ValueError("wind speed and slope magnitude must be nonnegative")
    sigma = sav_ft_inv
    bulk_density = load_lb_ft2 / depth_ft
    beta = bulk_density / PARTICLE_DENSITY_LB_FT3
    beta_opt = 3.348 * sigma ** -0.8189
    beta_ratio = beta / beta_opt
    gamma_max = sigma ** 1.5 / (495 + 0.0594 * sigma ** 1.5)
    a = 133 * sigma ** -0.7913  # Albini adjustment cited in Andrews table 3.
    gamma = gamma_max * beta_ratio ** a * exp(a * (1 - beta_ratio))
    moisture_ratio = min(moisture_fraction / moisture_extinction_fraction, 1.0)
    eta_m = max(0.0, 1 - 2.59 * moisture_ratio + 5.11 * moisture_ratio**2 - 3.52 * moisture_ratio**3)
    eta_s = min(0.174 * EFFECTIVE_MINERAL_FRACTION ** -0.19, 1.0)
    net_load = load_lb_ft2 * (1 - TOTAL_MINERAL_FRACTION)
    reaction_intensity = gamma * net_load * heat_btu_lb * eta_m * eta_s
    flux_ratio = exp((0.792 + 0.681 * sqrt(sigma)) * (beta + 0.1)) / (192 + 0.2595 * sigma)
    effective_heating = exp(-138 / sigma)
    preignition_heat = 250 + 1116 * moisture_fraction
    r0_ft_min = reaction_intensity * flux_ratio / (bulk_density * effective_heating * preignition_heat)
    c = 7.47 * exp(-0.133 * sigma**0.55)
    b = 0.02526 * sigma**0.54
    e = 0.715 * exp(-3.59e-4 * sigma)
    wind_ft_min = midflame_wind_m_s * FT_MIN_PER_M_S
    phi_w = c * wind_ft_min**b * beta_ratio**-e if wind_ft_min else 0.0
    phi_s = 5.275 * beta**-0.3 * slope**2
    return {
        "r0_m_s": r0_ft_min * M_S_PER_FT_MIN,
        "phi_w": phi_w,
        "phi_s": phi_s,
        "wind_c": c,
        "wind_b": b,
        "wind_e": e,
        "beta_ratio": beta_ratio,
    }


def dead_fuel_components(
    moisture_fraction: float,
    midflame_wind_m_s: float = 0.0,
    slope: float = 0.0,
    *,
    load_kg_m2: float,
    depth_m: float,
    sav_m_inv: float,
    moisture_extinction_fraction: float,
    heat_kj_kg: float = 18608.0,
) -> dict[str, float]:
    """Return spread components for one dead fuel particle class.

    Fuel load (kg/m²), bed depth (m), surface-area-to-volume ratio (m⁻¹),
    heat content (kJ/kg), moisture, midflame wind (m/s), and slope (rise/run)
    use SI units. Rates are returned in m/s. Fuel properties and heat must be
    finite and positive; extinction moisture must be in (0, 1]. The equation
    is evaluated in its published imperial units after explicit conversion.

    Particle density (32 lb/ft³), total mineral fraction (0.0555), and effective
    mineral fraction (0.01) retain the standard Rothermel assumptions used by
    the FM1 implementation; they are not inferred from these inputs. There is
    no wind-speed cap.
    """
    for name, value in (
        ("load_kg_m2", load_kg_m2),
        ("depth_m", depth_m),
        ("sav_m_inv", sav_m_inv),
        ("heat_kj_kg", heat_kj_kg),
    ):
        if not isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if not isfinite(moisture_extinction_fraction) or not 0 < moisture_extinction_fraction <= 1:
        raise ValueError("moisture extinction must be in (0, 1]")
    return _dead_fuel_components_imperial(
        moisture_fraction,
        midflame_wind_m_s,
        slope,
        load_lb_ft2=load_kg_m2 * KG_M2_TO_LB_FT2,
        depth_ft=depth_m * FT_PER_M,
        sav_ft_inv=sav_m_inv / FT_PER_M,
        moisture_extinction_fraction=moisture_extinction_fraction,
        heat_btu_lb=heat_kj_kg * KJ_KG_TO_BTU_LB,
    )


def fm1_components(moisture_fraction: float, midflame_wind_m_s: float = 0.0, slope: float = 0.0) -> dict[str, float]:
    """Return no-wind rate and directional factor magnitudes for FM1.

    Wind speed is *midflame* speed; slope is rise/run. All output rates are m/s.
    The legacy FM1 constants are passed directly in imperial units to preserve
    the existing numerical behavior exactly.
    """
    return _dead_fuel_components_imperial(
        moisture_fraction,
        midflame_wind_m_s,
        slope,
        load_lb_ft2=FM1_LOAD_LB_FT2,
        depth_ft=FM1_DEPTH_FT,
        sav_ft_inv=FM1_SAV_FT_INV,
        moisture_extinction_fraction=FM1_MOISTURE_EXTINCTION,
        heat_btu_lb=HEAT_BTU_LB,
    )


def fm1_directional_rate(moisture_fraction: float, wind_to_m_s: tuple[float, float], upslope_gradient: tuple[float, float], direction: tuple[float, float]) -> float:
    """Rate in a ray direction from an ignition point, using Andrews sec. 6.

    The 2-D grid applies this local elliptical-ignition rate to edges. That
    coupling is a Nightjar numerical approximation, not part of Rothermel's
    original one-dimensional equation.
    """
    wind_speed = hypot(*wind_to_m_s)
    slope = hypot(*upslope_gradient)
    pieces = fm1_components(moisture_fraction, wind_speed, slope)
    r0 = pieces["r0_m_s"]
    if r0 == 0:
        return 0.0
    wind_x = pieces["phi_w"] * wind_to_m_s[0] / wind_speed if wind_speed else 0.0
    wind_y = pieces["phi_w"] * wind_to_m_s[1] / wind_speed if wind_speed else 0.0
    slope_x = pieces["phi_s"] * upslope_gradient[0] / slope if slope else 0.0
    slope_y = pieces["phi_s"] * upslope_gradient[1] / slope if slope else 0.0
    resultant_x, resultant_y = wind_x + slope_x, wind_y + slope_y
    phi_effective = hypot(resultant_x, resultant_y)
    if phi_effective == 0:
        return r0
    head_rate = r0 * (1 + phi_effective)
    # Invert the table-3 wind factor, then convert effective wind to mi/h.
    effective_wind_ft_min = (phi_effective * pieces["beta_ratio"] ** pieces["wind_e"] / pieces["wind_c"]) ** (1 / pieces["wind_b"])
    length_width = 1 + 0.25 * (effective_wind_ft_min / 88)
    eccentricity = sqrt(length_width**2 - 1) / length_width
    direction_length = hypot(*direction)
    if direction_length == 0:
        raise ValueError("direction must be nonzero")
    cosine = (resultant_x * direction[0] + resultant_y * direction[1]) / (phi_effective * direction_length)
    cosine = max(-1.0, min(1.0, cosine))
    return head_rate * (1 - eccentricity) / (1 - eccentricity * cosine)
