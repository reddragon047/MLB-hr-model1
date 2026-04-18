import numpy as np
import pandas as pd
import requests

def get_game_weather(lat: float, lon: float, date_str: str, game_hour_local: int | None = None) -> dict:
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        "&hourly=temperature_2m,windspeed_10m,winddirection_10m,precipitation,rain,snowfall"
        "&daily=temperature_2m_max"
        "&wind_speed_unit=mph"
        "&temperature_unit=fahrenheit&timezone=auto"
        f"&start_date={date_str}&end_date={date_str}"
    )
    try:
        r = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        j = r.json()
    except Exception:
        return {
            "temp_f": None,
            "wind_speed_mph": None,
            "wind_dir_deg": None,
            "precip_mm": None,
            "rain_mm": None,
            "snowfall_cm": None,
        }

    daily_temp = None
    try:
        temps = j.get("daily", {}).get("temperature_2m_max", [])
        if temps:
            daily_temp = float(temps[0])
    except Exception:
        daily_temp = None

    hourly = j.get("hourly", {}) or {}
    hours = hourly.get("time", []) or []
    temps = hourly.get("temperature_2m", []) or []
    speeds = hourly.get("windspeed_10m", []) or []
    dirs = hourly.get("winddirection_10m", []) or []
    precips = hourly.get("precipitation", []) or []
    rains = hourly.get("rain", []) or []
    snows = hourly.get("snowfall", []) or []

    if not hours:
        return {
            "temp_f": daily_temp,
            "wind_speed_mph": None,
            "wind_dir_deg": None,
            "precip_mm": None,
            "rain_mm": None,
            "snowfall_cm": None,
        }

    target_hour = 19 if game_hour_local is None else int(game_hour_local)
    best_idx = None
    best_dist = 999
    for i, ts in enumerate(hours):
        try:
            hour = pd.to_datetime(ts).hour
            dist = abs(hour - target_hour)
            if dist < best_dist:
                best_dist = dist
                best_idx = i
        except Exception:
            continue

    if best_idx is None:
        return {
            "temp_f": daily_temp,
            "wind_speed_mph": None,
            "wind_dir_deg": None,
            "precip_mm": None,
            "rain_mm": None,
            "snowfall_cm": None,
        }

    def _pick(arr):
        try:
            return float(arr[best_idx]) if len(arr) > best_idx else None
        except Exception:
            return None

    temp_f = _pick(temps)
    if temp_f is None:
        temp_f = daily_temp

    return {
        "temp_f": temp_f,
        "wind_speed_mph": _pick(speeds),
        "wind_dir_deg": _pick(dirs),
        "precip_mm": _pick(precips),
        "rain_mm": _pick(rains),
        "snowfall_cm": _pick(snows),
    }

def compute_weather_multiplier(
    loc: dict | None,
    home_team: str,
    game_datetime: str | None,
    date_str: str,
    parse_game_hour_local,
    default_out_to_cf_bearings,
    temp_multiplier_fn,
    wind_multiplier_fn,
) -> tuple[float, dict]:
    if not loc or "lat" not in loc or "lon" not in loc:
        return 1.0, {
            "temp_f": None,
            "wind_speed_mph": None,
            "wind_dir_deg": None,
            "precip_mm": None,
            "rain_mm": None,
            "snowfall_cm": None,
            "temp_mult": 1.0,
            "wind_mult": 1.0,
        }

    game_hour = parse_game_hour_local(game_datetime)
    weather = get_game_weather(float(loc["lat"]), float(loc["lon"]), date_str, game_hour_local=game_hour)
    out_to_cf_deg = loc.get("out_to_cf_deg")
    if out_to_cf_deg is None:
        out_to_cf_deg = default_out_to_cf_bearings.get(home_team)

    temp_f = weather.get("temp_f")
    wind_speed = weather.get("wind_speed_mph")
    wind_dir = weather.get("wind_dir_deg")
    precip_mm = weather.get("precip_mm")
    snowfall_cm = weather.get("snowfall_cm")

    t_mult = temp_multiplier_fn(temp_f)
    w_mult = wind_multiplier_fn(wind_speed, wind_dir, out_to_cf_deg)
    final_mult = float(np.clip(t_mult * w_mult, 0.72, 1.20))

    if temp_f is not None:
        if temp_f <= 38:
            final_mult *= 0.60
        elif temp_f <= 45:
            final_mult *= 0.80
        elif temp_f <= 50:
            final_mult *= 0.92

    if snowfall_cm is not None and snowfall_cm > 0:
        final_mult *= 0.60
    elif precip_mm is not None and precip_mm >= 2.5:
        final_mult *= 0.75
    elif precip_mm is not None and precip_mm >= 0.8:
        final_mult *= 0.90

    final_mult = float(np.clip(final_mult, 0.55, 1.20))
    weather.update({
        "temp_mult": round(t_mult, 3),
        "wind_mult": round(w_mult, 3),
        "out_to_cf_deg": out_to_cf_deg,
    })
    return final_mult, weather
