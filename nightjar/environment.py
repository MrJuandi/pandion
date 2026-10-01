"""Retrospective station forcing, causal delivery, and piecewise edge travel.

Station air temperature/RH drive a simple Simard equilibrium + exponential
lag estimate, not measured fuel moisture or a calibrated wetting model.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from math import exp, isfinite, radians, sin, cos, inf
from typing import Any, Callable

SCHEMA_VERSION = 'nightjar-environment-0.1'
MOISTURE_KEYS = ('dead_1h', 'dead_10h', 'dead_100h', 'dead_herbaceous', 'live_herbaceous', 'live_woody')
VARIABLES = {
    'air_temperature': ('air_temperature_c', '°C', -80, 70),
    'relative_humidity': ('relative_humidity_pct', '%', 0, 100),
    'precipitation': ('precipitation_mm', 'mm', 0, 1000),
    'wind_speed_at_2m': ('wind_speed_2m_m_s', 'm/s', 0, 100),
    'wind_direction_at_2m': ('wind_from_deg', '°', 0, 360),
}


def utc(value: str) -> datetime:
    try:
        result = datetime.fromisoformat(value.replace('Z', '+00:00'))
    except (ValueError, AttributeError) as exc:
        raise ValueError('environment timestamps must be UTC ISO-8601') from exc
    if result.tzinfo is None or result.utcoffset().total_seconds() != 0:
        raise ValueError('environment timestamps must include UTC timezone')
    return result.astimezone(timezone.utc)


def equilibrium_moisture(temperature_c: float, relative_humidity_pct: float) -> float:
    """Simard equilibrium moisture as dry-weight fraction (PSW-GTR-82)."""
    if not isfinite(temperature_c) or not isfinite(relative_humidity_pct) or not 0 <= relative_humidity_pct <= 100:
        raise ValueError('temperature must be finite and RH within 0–100 percent')
    t, h = temperature_c * 1.8 + 32.0, relative_humidity_pct
    if h < 10:
        percent = 0.03229 + 0.281073*h - 0.000578*h*t
    elif h < 50:
        percent = 2.22749 + 0.160107*h - 0.014784*t
    else:
        percent = 21.0606 + 0.005565*h*h - 0.00035*h*t - 0.483199*h
    return max(0.0, percent / 100.0)


def build_station_environment(scenario: dict, audit: dict, *, station_wind_factor: float = 1.0,
                              moisture_mode: str = 'emc_lag') -> dict:
    """Build bounded forcing from valid aligned intervals; delivery waits for end.

    The caller verifies audit-to-original-pilot SHA256 binding. This function
    also requires EMSR578 identity and the same initial historical timestamp.
    """
    if audit.get('schema_version') != 'nightjar-pilot-weather-audit-0.1':
        raise ValueError('unsupported station weather audit schema')
    if scenario.get('activation_id') != 'EMSR578' or audit.get('station', {}).get('code') != 'X6':
        raise ValueError('Baldomar station forcing is scoped to EMSR578')
    origin_text = scenario['provenance']['image_acquired_at_utc']
    origin = utc(origin_text)
    if utc(audit['window']['start_utc']) != origin:
        raise ValueError('weather audit time origin differs from scenario')
    if not isfinite(station_wind_factor) or station_wind_factor <= 0 or moisture_mode not in ('emc_lag', 'fixed'):
        raise ValueError('invalid station wind factor or moisture mode')
    horizon = float(scenario['fire']['horizon_s'])
    if not isfinite(horizon) or horizon <= 0:
        raise ValueError('environment horizon must be positive and finite')
    series: dict[tuple[datetime, datetime], dict] = {}
    keys_reference = None
    for key, (field, unit, low, high) in VARIABLES.items():
        variable = audit['variables'][key]
        if variable.get('unit') != unit:
            raise ValueError(f'{key}: unexpected measurement units')
        keys = set()
        for record in variable['records']:
            start, end = utc(record['interval_start_utc']), utc(record['interval_end_utc'])
            if (end-start).total_seconds() != 1800:
                raise ValueError('station weather requires half-hour intervals')
            if end <= origin or (start-origin).total_seconds() >= horizon:
                continue
            pair = (start, end)
            if pair in keys:
                raise ValueError(f'{key}: duplicate weather interval')
            keys.add(pair)
            value = record['value']
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value) or not low <= value <= high:
                raise ValueError(f'{key}: invalid measured weather value')
            if record.get('quality_code') != 'V' or record.get('usable') is not True:
                raise ValueError(f'{key}: weather interval is not validated')
            series.setdefault(pair, {})[field] = float(value)
        if keys_reference is not None and keys != keys_reference:
            raise ValueError('weather variables have inconsistent interval coverage')
        keys_reference = keys
    current_moisture = {k: float(scenario['fire']['moisture'][k]) for k in MOISTURE_KEYS}
    if any(not isfinite(v) or v < 0 for v in current_moisture.values()):
        raise ValueError('initial modeled fuel moisture must be finite and nonnegative')
    forcing, previous_end = [], 0.0
    for (start, end), weather in sorted(series.items()):
        start_s = max(0.0, (start-origin).total_seconds())
        end_s = min(horizon, (end-origin).total_seconds())
        if abs(start_s-previous_end) > 1e-6:
            raise ValueError('station weather has a gap in scenario horizon')
        if moisture_mode == 'emc_lag' and weather['precipitation_mm'] != 0:
            raise ValueError('emc_lag supports dry intervals only; rain wetting is not calibrated')
        direction = radians(weather['wind_from_deg'])
        speed = weather['wind_speed_2m_m_s'] * station_wind_factor
        forcing.append({
            'start_s': start_s, 'end_s': end_s,
            'available_at_s': (end-origin).total_seconds(),
            'interval_start_utc': start.isoformat().replace('+00:00','Z'),
            'interval_end_utc': end.isoformat().replace('+00:00','Z'),
            'station_weather': weather,
            'midflame_wind_to_m_s': [-speed*sin(direction), -speed*cos(direction)],
            'moisture': deepcopy(current_moisture),
        })
        if moisture_mode == 'emc_lag':
            equilibrium = equilibrium_moisture(weather['air_temperature_c'], weather['relative_humidity_pct'])
            elapsed = end_s-start_s
            for key, lag_h in [('dead_1h',1),('dead_10h',10),('dead_100h',100),('dead_herbaceous',1)]:
                current_moisture[key] = equilibrium + (current_moisture[key]-equilibrium)*exp(-elapsed/(lag_h*3600))
        previous_end = end_s
    if not forcing or abs(previous_end-horizon) > 1e-6:
        raise ValueError('station weather does not cover full scenario horizon')
    return {
        'schema_version': SCHEMA_VERSION, 'weather_mode': 'station_intervals',
        'time_origin_utc': origin_text, 'station': deepcopy(audit['station']),
        'sources': deepcopy(audit['sources']), 'forcing': forcing,
        'units': {'time':'s','temperature':'°C','relative_humidity':'%','precipitation':'mm','wind':'m/s','fuel_moisture':'dry_weight_fraction'},
        'assumptions': {'station_wind_factor':station_wind_factor,
                        'wind':'spatially uniform station 2m proxy; uncalibrated midflame conversion',
                        'moisture_mode':moisture_mode,
                        'dead_fuel_moisture':'Simard equilibrium with first-order 1/10/100h lag, piecewise held' if moisture_mode=='emc_lag' else 'fixed modeled scenario values',
                        'initial_fuel_moisture':'modeled, not measured',
                        'live_fuel_moisture':'fixed modeled scenario values',
                        'rain_wetting':'not calibrated; emc_lag rejects wet intervals',
                        'delivery':'interval end; additional source delivery latency unknown, assumed zero'},
    }


def validate_forcing(environment: dict | None, horizon_s: float) -> list[dict]:
    """Reject incomplete/noncausal forcing; constant legacy scenarios return []."""
    if not environment:
        return []
    if environment.get('schema_version') != SCHEMA_VERSION:
        raise ValueError('unsupported environment schema')
    if environment.get('weather_mode') == 'constant_synthetic':
        if environment.get('forcing'):
            raise ValueError('constant weather cannot include time-varying forcing')
        return []
    if environment.get('weather_mode') != 'station_intervals':
        raise ValueError('unsupported environment weather mode')
    utc(environment['time_origin_utc'])
    forcing = environment.get('forcing', [])
    previous = 0.0
    for segment in forcing:
        start, end, available = (float(segment[k]) for k in ('start_s','end_s','available_at_s'))
        if not all(isfinite(v) for v in (start,end,available)) or abs(start-previous)>1e-6 or end<=start or available<end:
            raise ValueError('forcing must cover continuous intervals and cannot arrive before interval end')
        actual_start, actual_end = utc(segment['interval_start_utc']), utc(segment['interval_end_utc'])
        origin = utc(environment['time_origin_utc'])
        if (abs(available-(actual_end-origin).total_seconds())>1e-6
                or abs(start-max(0.0,(actual_start-origin).total_seconds()))>1e-6
                or abs(end-min(horizon_s,(actual_end-origin).total_seconds()))>1e-6
                or (actual_end-actual_start).total_seconds()!=1800):
            raise ValueError('forcing interval timestamps and availability disagree')
        wind=segment['midflame_wind_to_m_s']
        if len(wind)!=2 or any(not isfinite(float(v)) for v in wind):
            raise ValueError('forcing wind must contain two finite components')
        if any(not isfinite(float(segment['moisture'][k])) or float(segment['moisture'][k])<0 for k in MOISTURE_KEYS):
            raise ValueError('forcing fuel moisture must be finite and nonnegative')
        previous=end
    if not forcing or abs(previous-horizon_s)>1e-6:
        raise ValueError('forcing does not cover simulation horizon')
    return forcing


def delivered_weather(forcing: list[dict], time_s: float) -> dict | None:
    """Expose only a completed interval; never return future forcing rows."""
    ready=[s for s in forcing if s['available_at_s']<=time_s]
    return deepcopy(ready[-1]) if ready else None


def edge_arrival(start_s: float, distance_m: float, forcing: list[dict], speed_at: Callable[[int], float]) -> float:
    """Integrate remaining edge distance across bounded piecewise constant speeds."""
    remaining, cursor = distance_m, start_s
    for index, segment in enumerate(forcing):
        if segment['end_s']<=cursor:
            continue
        cursor=max(cursor,segment['start_s'])
        speed=speed_at(index)
        if not isfinite(speed) or speed<0:
            raise ValueError('edge spread speed must be finite and nonnegative')
        duration=segment['end_s']-cursor
        if speed>0 and remaining<=speed*duration:
            return cursor+remaining/speed
        remaining-=speed*duration
        cursor=segment['end_s']
    return inf
