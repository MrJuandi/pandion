"""Universal geodesy module for ETRS89 / WGS84 and Transverse Mercator (UTM).

Implements exact Krüger series for transverse Mercator projection on the
GRS80 / WGS84 ellipsoid for all UTM zones (Zones 1 to 60), with focus on
Iberian coordinate systems:
- UTM Zone 29N (EPSG:25829, Central Meridian 9°W, Portugal and Western Spain)
- UTM Zone 30N (EPSG:25830, Central Meridian 3°W, Central / Southern Spain)
- UTM Zone 31N (EPSG:25831, Central Meridian 3°E, Eastern Spain / Catalonia)

Provides sub-millimeter forward and inverse transformations without external GIS dependencies.
"""

from __future__ import annotations

import math
from typing import Tuple

# GRS80 / WGS84 ellipsoid parameters
_A = 6378137.0  # semi-major axis (meters)
_F = 1.0 / 298.257223563  # flattening
_B = _A * (1.0 - _F)  # semi-minor axis
_E2 = (_A**2 - _B**2) / (_A**2)  # first eccentricity squared
_E_PRIME2 = (_A**2 - _B**2) / (_B**2)  # second eccentricity squared
_K0 = 0.9996  # UTM scale factor
_FALSE_EASTING = 500000.0  # False Easting (meters)
_FALSE_NORTHING = 0.0  # False Northing (meters)


def utm_central_meridian_deg(zone: int) -> float:
    """Return the central meridian in degrees for a given UTM zone (1 to 60)."""
    if not (1 <= zone <= 60):
        raise ValueError(f"UTM zone must be between 1 and 60, got {zone}")
    return float(zone * 6 - 183)


def utm_zone_from_lon(lon_deg: float) -> int:
    """Return the default UTM zone (1 to 60) for a given longitude in degrees."""
    if not math.isfinite(lon_deg):
        raise ValueError("Longitude must be a finite number")
    lon_norm = (lon_deg + 180.0) % 360.0 - 180.0
    zone = int((lon_norm + 180.0) / 6.0) + 1
    return max(1, min(60, zone))


def epsg_for_utm_zone(zone: int, datum: str = "ETRS89") -> str:
    """Return standard EPSG code for a Northern Hemisphere UTM zone."""
    if not (1 <= zone <= 60):
        raise ValueError(f"UTM zone must be between 1 and 60, got {zone}")
    datum_upper = datum.upper()
    if datum_upper == "ETRS89":
        return f"EPSG:{25800 + zone}"
    if datum_upper == "WGS84":
        return f"EPSG:{32600 + zone}"
    raise ValueError(f"Unsupported datum: {datum} (expected 'ETRS89' or 'WGS84')")


def wgs84_to_utm(lon_deg: float, lat_deg: float, zone: int = 31) -> Tuple[float, float]:
    """Convert WGS84 geographic coordinates (degrees) to UTM Zone coordinates (meters).

    Uses Krüger's transverse Mercator series on the GRS80/WGS84 ellipsoid.
    Returns (easting_m, northing_m).
    """
    if not (math.isfinite(lon_deg) and math.isfinite(lat_deg)):
        raise ValueError("Coordinates must be finite numbers")
    if not (-180.0 <= lon_deg <= 180.0 and -80.0 <= lat_deg <= 84.0):
        raise ValueError(f"Coordinates ({lon_deg}, {lat_deg}) out of valid UTM range")
    if not (1 <= zone <= 60):
        raise ValueError(f"UTM zone must be between 1 and 60, got {zone}")

    lon0_rad = math.radians(utm_central_meridian_deg(zone))
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)

    sin_lat = math.sin(lat)
    cos_lat = math.cos(lat)
    tan_lat = math.tan(lat)

    n_val = _A / math.sqrt(1.0 - _E2 * sin_lat**2)
    t_val = tan_lat**2
    c_val = _E_PRIME2 * cos_lat**2
    a_val = (lon - lon0_rad) * cos_lat

    # Meridional distance M
    m_val = _A * (
        (1.0 - _E2 / 4.0 - 3.0 * _E2**2 / 64.0 - 5.0 * _E2**3 / 256.0) * lat
        - (3.0 * _E2 / 8.0 + 3.0 * _E2**2 / 32.0 + 45.0 * _E2**3 / 1024.0) * math.sin(2.0 * lat)
        + (15.0 * _E2**2 / 256.0 + 45.0 * _E2**3 / 1024.0) * math.sin(4.0 * lat)
        - (35.0 * _E2**3 / 3072.0) * math.sin(6.0 * lat)
    )

    x = _K0 * n_val * (
        a_val
        + (1.0 - t_val + c_val) * a_val**3 / 6.0
        + (5.0 - 18.0 * t_val + t_val**2 + 72.0 * c_val - 58.0 * _E_PRIME2) * a_val**5 / 120.0
    ) + _FALSE_EASTING

    y = _K0 * (
        m_val
        + n_val * tan_lat * (
            a_val**2 / 2.0
            + (5.0 - t_val + 9.0 * c_val + 4.0 * c_val**2) * a_val**4 / 24.0
            + (61.0 - 58.0 * t_val + t_val**2 + 600.0 * c_val - 330.0 * _E_PRIME2) * a_val**6 / 720.0
        )
    ) + _FALSE_NORTHING

    return x, y


def utm_to_wgs84(x_m: float, y_m: float, zone: int = 31) -> Tuple[float, float]:
    """Convert UTM Zone coordinates (meters) to WGS84 geographic (degrees).

    Returns (lon_deg, lat_deg).
    """
    if not (math.isfinite(x_m) and math.isfinite(y_m)):
        raise ValueError("Coordinates must be finite numbers")
    if not (1 <= zone <= 60):
        raise ValueError(f"UTM zone must be between 1 and 60, got {zone}")

    lon0_rad = math.radians(utm_central_meridian_deg(zone))
    x_adj = x_m - _FALSE_EASTING
    y_adj = y_m - _FALSE_NORTHING

    m_val = y_adj / _K0
    mu = m_val / (_A * (1.0 - _E2 / 4.0 - 3.0 * _E2**2 / 64.0 - 5.0 * _E2**3 / 256.0))
    e1 = (1.0 - math.sqrt(1.0 - _E2)) / (1.0 + math.sqrt(1.0 - _E2))

    phi1 = mu + (
        (3.0 * e1 / 2.0 - 27.0 * e1**3 / 32.0) * math.sin(2.0 * mu)
        + (21.0 * e1**2 / 16.0 - 55.0 * e1**4 / 32.0) * math.sin(4.0 * mu)
        + (151.0 * e1**3 / 96.0) * math.sin(6.0 * mu)
        + (1097.0 * e1**4 / 512.0) * math.sin(8.0 * mu)
    )

    sin_phi1 = math.sin(phi1)
    cos_phi1 = math.cos(phi1)
    tan_phi1 = math.tan(phi1)

    n1 = _A / math.sqrt(1.0 - _E2 * sin_phi1**2)
    t1 = tan_phi1**2
    c1 = _E_PRIME2 * cos_phi1**2
    r1 = _A * (1.0 - _E2) / ((1.0 - _E2 * sin_phi1**2)**1.5)
    d = x_adj / (n1 * _K0)

    lat = phi1 - (n1 * tan_phi1 / r1) * (
        d**2 / 2.0
        - (5.0 + 3.0 * t1 + 10.0 * c1 - 4.0 * c1**2 - 9.0 * _E_PRIME2) * d**4 / 24.0
        + (61.0 + 90.0 * t1 + 298.0 * c1 + 45.0 * t1**2 - 252.0 * _E_PRIME2 - 3.0 * c1**2) * d**6 / 720.0
    )

    lon = lon0_rad + (
        d
        - (1.0 + 2.0 * t1 + c1) * d**3 / 6.0
        + (5.0 - 2.0 * c1 + 28.0 * t1 - 3.0 * c1**2 + 8.0 * _E_PRIME2 + 24.0 * t1**2) * d**5 / 120.0
    ) / cos_phi1

    return math.degrees(lon), math.degrees(lat)


# Backward-compatible convenience functions
def wgs84_to_utm31n(lon_deg: float, lat_deg: float) -> Tuple[float, float]:
    """Convert WGS84 to ETRS89 / UTM Zone 31N (meters)."""
    return wgs84_to_utm(lon_deg, lat_deg, zone=31)


def utm31n_to_wgs84(x_m: float, y_m: float) -> Tuple[float, float]:
    """Convert ETRS89 / UTM Zone 31N (meters) to WGS84."""
    return utm_to_wgs84(x_m, y_m, zone=31)


def wgs84_to_utm30n(lon_deg: float, lat_deg: float) -> Tuple[float, float]:
    """Convert WGS84 to ETRS89 / UTM Zone 30N (meters)."""
    return wgs84_to_utm(lon_deg, lat_deg, zone=30)


def utm30n_to_wgs84(x_m: float, y_m: float) -> Tuple[float, float]:
    """Convert ETRS89 / UTM Zone 30N (meters) to WGS84."""
    return utm_to_wgs84(x_m, y_m, zone=30)


def wgs84_to_utm29n(lon_deg: float, lat_deg: float) -> Tuple[float, float]:
    """Convert WGS84 to ETRS89 / UTM Zone 29N (meters)."""
    return wgs84_to_utm(lon_deg, lat_deg, zone=29)


def utm29n_to_wgs84(x_m: float, y_m: float) -> Tuple[float, float]:
    """Convert ETRS89 / UTM Zone 29N (meters) to WGS84."""
    return utm_to_wgs84(x_m, y_m, zone=29)
