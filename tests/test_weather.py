import unittest
from unittest.mock import patch

from jarvis.tools import weather
from jarvis.tools.weather import WeatherError


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict:
        return self._payload


_GEOCODE_PAYLOAD = {
    "results": [{"name": "Boston", "admin1": "Massachusetts", "country": "United States", "latitude": 42.36, "longitude": -71.06}]
}

_FORECAST_PAYLOAD = {
    "current": {"temperature_2m": 18.5, "apparent_temperature": 17.0, "weather_code": 2, "wind_speed_10m": 12.0},
    "daily": {
        "time": ["2026-09-06", "2026-09-07"],
        "weather_code": [2, 61],
        "temperature_2m_max": [20.0, 19.0],
        "temperature_2m_min": [12.0, 13.0],
    },
}

_IP_LOCATION_PAYLOAD = {"latitude": 40.7, "longitude": -74.0, "city": "New York", "region": "New York", "country_name": "United States"}


class WeatherTest(unittest.TestCase):
    @patch("requests.get")
    def test_named_location_returns_current_and_forecast(self, mock_get) -> None:
        mock_get.side_effect = [_FakeResponse(_GEOCODE_PAYLOAD), _FakeResponse(_FORECAST_PAYLOAD)]

        result = weather.get_weather("Boston")

        self.assertIn("Boston", result)
        self.assertIn("partly cloudy", result)
        self.assertIn("18.5", result)
        self.assertIn("Tomorrow", result)
        self.assertIn("light rain", result)  # code 61

    @patch("requests.get")
    def test_current_location_uses_ip_geolocation(self, mock_get) -> None:
        mock_get.side_effect = [_FakeResponse(_IP_LOCATION_PAYLOAD), _FakeResponse(_FORECAST_PAYLOAD)]

        result = weather.get_weather("current")

        self.assertIn("New York", result)
        # first call must be the IP geolocation lookup, not the geocoder
        first_call_url = mock_get.call_args_list[0].args[0]
        self.assertIn("ipapi.co", first_call_url)

    @patch("requests.get")
    def test_empty_location_defaults_to_current(self, mock_get) -> None:
        mock_get.side_effect = [_FakeResponse(_IP_LOCATION_PAYLOAD), _FakeResponse(_FORECAST_PAYLOAD)]
        result = weather.get_weather("")
        self.assertIn("New York", result)

    @patch("requests.get")
    def test_unknown_place_raises(self, mock_get) -> None:
        mock_get.return_value = _FakeResponse({"results": []})
        with self.assertRaises(WeatherError):
            weather.get_weather("Nowhereville")

    @patch("requests.get")
    def test_ip_geolocation_failure_raises(self, mock_get) -> None:
        mock_get.return_value = _FakeResponse({"error": True})
        with self.assertRaises(WeatherError):
            weather.get_weather("current")


if __name__ == "__main__":
    unittest.main()
