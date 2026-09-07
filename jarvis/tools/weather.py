"""Stage 4: weather, via Open-Meteo — free, no API key required.

Two Open-Meteo endpoints are chained: geocoding (name -> lat/lon) and
forecast (lat/lon -> current conditions + a short outlook). When the user
doesn't name a place ("what's the weather", "here", "my current location"),
a free IP-geolocation lookup guesses one from the machine's public IP.
"""

from __future__ import annotations

_GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
_IP_LOCATION_URL = "https://ipapi.co/json/"

_CURRENT_LOCATION_PHRASES = {
    "",
    "current",
    "current location",
    "here",
    "my location",
    "my current location",
    "where i am",
}

# WMO weather codes used by Open-Meteo -> plain-English description.
_WEATHER_CODES = {
    0: "clear sky",
    1: "mostly clear",
    2: "partly cloudy",
    3: "overcast",
    45: "foggy",
    48: "foggy with rime",
    51: "light drizzle",
    53: "moderate drizzle",
    55: "dense drizzle",
    56: "light freezing drizzle",
    57: "dense freezing drizzle",
    61: "light rain",
    63: "moderate rain",
    65: "heavy rain",
    66: "light freezing rain",
    67: "heavy freezing rain",
    71: "light snow",
    73: "moderate snow",
    75: "heavy snow",
    77: "snow grains",
    80: "light rain showers",
    81: "moderate rain showers",
    82: "violent rain showers",
    85: "light snow showers",
    86: "heavy snow showers",
    95: "thunderstorm",
    96: "thunderstorm with light hail",
    99: "thunderstorm with heavy hail",
}


class WeatherError(Exception):
    pass


def _require_requests():
    try:
        import requests
    except ImportError as exc:
        raise WeatherError("the weather tool requires the 'requests' package (pip install requests)") from exc
    return requests


def _describe_code(code: int) -> str:
    return _WEATHER_CODES.get(code, f"unrecognized conditions (code {code})")


def _resolve_current_location() -> tuple[float, float, str]:
    requests = _require_requests()
    try:
        response = requests.get(_IP_LOCATION_URL, timeout=8)
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as exc:
        raise WeatherError(f"couldn't determine your current location: {exc}") from exc

    if data.get("error") or "latitude" not in data or "longitude" not in data:
        raise WeatherError("couldn't determine your current location from your network connection")

    label = ", ".join(part for part in (data.get("city"), data.get("region"), data.get("country_name")) if part)
    return data["latitude"], data["longitude"], label or "your current location"


def _geocode(location: str) -> tuple[float, float, str]:
    requests = _require_requests()
    try:
        response = requests.get(_GEOCODE_URL, params={"name": location, "count": 1}, timeout=8)
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as exc:
        raise WeatherError(f"couldn't look up '{location}': {exc}") from exc

    results = data.get("results") or []
    if not results:
        raise WeatherError(f"couldn't find a place named '{location}'")

    match = results[0]
    label_parts = [match.get("name"), match.get("admin1"), match.get("country")]
    label = ", ".join(part for part in label_parts if part)
    return match["latitude"], match["longitude"], label or location


def get_weather(location: str = "current") -> str:
    """Current conditions + a short outlook for a named place, or the caller's
    apparent location (via IP geolocation) when no specific place is given."""
    if location.strip().lower() in _CURRENT_LOCATION_PHRASES:
        latitude, longitude, label = _resolve_current_location()
    else:
        latitude, longitude, label = _geocode(location)

    requests = _require_requests()
    try:
        response = requests.get(
            _FORECAST_URL,
            params={
                "latitude": latitude,
                "longitude": longitude,
                "current": "temperature_2m,apparent_temperature,weather_code,wind_speed_10m",
                "daily": "temperature_2m_max,temperature_2m_min,weather_code",
                "forecast_days": 2,
                "timezone": "auto",
            },
            timeout=8,
        )
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as exc:
        raise WeatherError(f"couldn't fetch weather for {label}: {exc}") from exc

    current = data.get("current", {})
    daily = data.get("daily", {})

    lines = [f"Weather for {label}:"]
    if current:
        lines.append(
            f"Right now: {_describe_code(current.get('weather_code'))}, "
            f"{current.get('temperature_2m')}°C (feels like {current.get('apparent_temperature')}°C), "
            f"wind {current.get('wind_speed_10m')} km/h."
        )
    if daily and daily.get("time"):
        for i, date in enumerate(daily["time"][:2]):
            label_day = "Today" if i == 0 else "Tomorrow"
            lines.append(
                f"{label_day} ({date}): {_describe_code(daily['weather_code'][i])}, "
                f"high {daily['temperature_2m_max'][i]}°C / low {daily['temperature_2m_min'][i]}°C."
            )
    return "\n".join(lines)
