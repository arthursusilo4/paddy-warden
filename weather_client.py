# ============================================================
# weather_client.py — Open-Meteo Weather Fetcher
# ============================================================

import httpx
from datetime import date, timedelta
from typing import List, Dict

# Open-Meteo APIs
OPENMETEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
OPENMETEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# All 31 Jember locations
LOCATIONS = {
    1:  {"name": "ajung",        "lat": -8.24536,  "lon": 113.653218},
    2:  {"name": "ambulu",       "lat": -8.339242, "lon": 113.607208},
    3:  {"name": "arjasa",       "lat": -8.117073, "lon": 113.74904},
    4:  {"name": "balung",       "lat": -8.268611, "lon": 113.526667},
    5:  {"name": "bangsalsari",  "lat": -8.201327, "lon": 113.532373},
    6:  {"name": "gumukmas",     "lat": -8.323056, "lon": 113.406667},
    7:  {"name": "jelbuk",       "lat": -8.083994, "lon": 113.758512},
    8:  {"name": "jenggawah",    "lat": -8.26832,  "lon": 113.647703},
    9:  {"name": "jombang",      "lat": -8.244448, "lon": 113.365235},
    10: {"name": "kalisat",      "lat": -8.123056, "lon": 113.808333},
    11: {"name": "kaliwates",    "lat": -8.187778, "lon": 113.675556},
    12: {"name": "kencong",      "lat": -8.289722, "lon": 113.351111},
    13: {"name": "ledokombo",    "lat": -8.141667, "lon": 113.911944},
    14: {"name": "mayang",       "lat": -8.195833, "lon": 113.798333},
    15: {"name": "mumbulsari",   "lat": -8.261944, "lon": 113.740278},
    16: {"name": "pakusari",     "lat": -8.1525,   "lon": 113.769722},
    17: {"name": "panti",        "lat": -8.175472, "lon": 113.619047},
    18: {"name": "patrang",      "lat": -8.138889, "lon": 113.699722},
    19: {"name": "puger",        "lat": -8.330556, "lon": 113.468889},
    20: {"name": "rambipuji",    "lat": -8.223611, "lon": 113.598333},
    21: {"name": "sembero",      "lat": -8.203056, "lon": 113.445278},
    22: {"name": "silo",         "lat": -8.271111, "lon": 113.871111},
    23: {"name": "sukorambi",    "lat": -8.139444, "lon": 113.660833},
    24: {"name": "sukowono",     "lat": -8.059167, "lon": 113.836345},
    25: {"name": "sumberbaru",   "lat": -8.119229, "lon": 113.392973},
    26: {"name": "sumberjambe",  "lat": -8.0658,   "lon": 113.898801},
    27: {"name": "sumbersari",   "lat": -8.186669, "lon": 113.721453},
    28: {"name": "tanggul",      "lat": -8.161572, "lon": 113.451239},
    29: {"name": "tempurejo",    "lat": -8.338299, "lon": 113.712116},
    30: {"name": "umbulsari",    "lat": -8.264046, "lon": 113.448387},
    31: {"name": "wuluhan",      "lat": -8.342778, "lon": 113.547778},
}

# Open-Meteo daily fields to request
OPENMETEO_DAILY_FIELDS = [
    "temperature_2m_max",
    "temperature_2m_min",
    "temperature_2m_mean",
    "apparent_temperature_max",
    "apparent_temperature_min",
    "apparent_temperature_mean",
    "dew_point_2m_mean",
    "relative_humidity_2m_mean",
    "precipitation_sum",
    "wind_speed_10m_max",
    "wind_direction_10m_dominant",
    "wind_gusts_10m_max",
    "surface_pressure_mean",
    "cloud_cover_mean",
    "shortwave_radiation_sum",
    "uv_index_max",
    "snowfall_sum",
]


def compute_moonphase(d: date) -> float:
    """Approximate moon phase (0.0 = new moon, 0.5 = full moon)."""
    ref_new_moon = date(2000, 1, 6)
    days_since_ref = (d - ref_new_moon).days
    lunar_cycle = 29.53059
    phase = (days_since_ref % lunar_cycle) / lunar_cycle
    return round(phase, 4)


def celsius_to_fahrenheit(c: float) -> float:
    """Convert Celsius to Fahrenheit."""
    return (c * 9.0 / 5.0) + 32.0


def _parse_openmeteo_daily(daily: dict, i: int) -> dict:
    """Parse a single day's data from Open-Meteo response."""
    def get_val(field: str, default: float = 0.0) -> float:
        vals = daily.get(field, [])
        if i < len(vals) and vals[i] is not None:
            return float(vals[i])
        return default
    
    dt = date.fromisoformat(daily["time"][i])
    
    temp_c = get_val("temperature_2m_mean", 28.0)
    
    return {
        "datetime": dt,
        "tempmax": celsius_to_fahrenheit(get_val("temperature_2m_max", 32.0)),
        "tempmin": celsius_to_fahrenheit(get_val("temperature_2m_min", 24.0)),
        "temp": temp_c,
        "feelslikemax": celsius_to_fahrenheit(get_val("apparent_temperature_max", 35.0)),
        "feelslikemin": celsius_to_fahrenheit(get_val("apparent_temperature_min", 23.0)),
        "feelslike": celsius_to_fahrenheit(get_val("apparent_temperature_mean", 30.0)),
        "dew": get_val("dew_point_2m_mean", 22.0),
        "humidity": get_val("relative_humidity_2m_mean", 80.0),
        "precip": get_val("precipitation_sum", 0.0),
        "precipprob": 0.0 if get_val("precipitation_sum") == 0 else min(get_val("precipitation_sum") * 15, 100),
        "precipcover": 0.0 if get_val("precipitation_sum") == 0 else min(get_val("precipitation_sum") * 10, 100),
        "snow": get_val("snowfall_sum", 0.0),
        "snowdepth": 0.0,
        "windgust": get_val("wind_gusts_10m_max", 5.0),
        "windspeed": get_val("wind_speed_10m_max", 3.0),
        "winddir": get_val("wind_direction_10m_dominant", 180.0),
        "sealevelpressure": get_val("surface_pressure_mean", 1013.0),
        "cloudcover": get_val("cloud_cover_mean", 50.0),
        "visibility": 10.0,
        "solarradiation": get_val("shortwave_radiation_sum", 15.0),
        "solarenergy": get_val("shortwave_radiation_sum", 15.0) * 0.24,
        "uvindex": get_val("uv_index_max", 8.0),
        "severerisk": 0.0,
        "moonphase": compute_moonphase(dt),
    }


async def fetch_historical_weather(
    location_id: int, 
    days: int = 67
) -> List[Dict]:
    """
    Fetch historical weather from Open-Meteo Archive API.
    Returns data ending at yesterday.
    """
    if location_id not in LOCATIONS:
        raise ValueError(f"Invalid location_id: {location_id}. Must be 1-31.")
    
    loc = LOCATIONS[location_id]
    end_date = date.today() - timedelta(days=1)
    start_date = end_date - timedelta(days=days - 1)
    
    params = {
        "latitude": loc["lat"],
        "longitude": loc["lon"],
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "daily": ",".join(OPENMETEO_DAILY_FIELDS),
        "timezone": "Asia/Jakarta",
        "wind_speed_unit": "ms",
    }
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(OPENMETEO_ARCHIVE_URL, params=params)
        response.raise_for_status()
        data = response.json()
    
    daily = data.get("daily", {})
    if not daily or not daily.get("time"):
        raise ValueError(f"No historical data for location {location_id}")
    
    return [_parse_openmeteo_daily(daily, i) for i in range(len(daily["time"]))]


async def fetch_forecast_weather(
    location_id: int, 
    days: int = 16
) -> List[Dict]:
    """
    Fetch weather forecast from Open-Meteo Forecast API.
    Returns data starting from today.
    """
    if location_id not in LOCATIONS:
        raise ValueError(f"Invalid location_id: {location_id}. Must be 1-31.")
    
    loc = LOCATIONS[location_id]
    
    params = {
        "latitude": loc["lat"],
        "longitude": loc["lon"],
        "daily": ",".join(OPENMETEO_DAILY_FIELDS),
        "timezone": "Asia/Jakarta",
        "wind_speed_unit": "ms",
        "forecast_days": days,
    }
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(OPENMETEO_FORECAST_URL, params=params)
        response.raise_for_status()
        data = response.json()
    
    daily = data.get("daily", {})
    if not daily or not daily.get("time"):
        raise ValueError(f"No forecast data for location {location_id}")
    
    return [_parse_openmeteo_daily(daily, i) for i in range(len(daily["time"]))]


async def fetch_merged_weather(
    location_id: int,
    forecast_days: int = 14
) -> List[Dict]:
    """
    Fetch and merge historical + forecast weather.
    
    Returns continuous weather data suitable for multi-day predictions.
    
    Timeline:
    - Historical: Days [-13, -1] (13 days ending yesterday)
    - Forecast: Days [0, 15] (today + 15 days)
    - Total: ~29 days
    
    This allows predicting Days 1-14 with full 14-day lookback windows.
    """
    # Fetch 13 days historical (to get days -13 to -1)
    historical = await fetch_historical_weather(location_id, days=13)
    
    # Fetch 16 days forecast (to get days 0 to 15, giving buffer for Day 14)
    forecast = await fetch_forecast_weather(location_id, days=16)
    
    # Merge: historical ends at yesterday, forecast starts at today
    # No overlap, just concatenate
    merged = historical + forecast
    
    return merged