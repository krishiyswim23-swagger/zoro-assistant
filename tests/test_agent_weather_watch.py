import unittest
from unittest.mock import patch

from jarvis.agents.weather_watch import WeatherWatchAgent
from jarvis.tools.weather import WeatherError


class WeatherWatchAgentTest(unittest.TestCase):
    def test_returns_first_line_of_the_weather_report(self) -> None:
        agent = WeatherWatchAgent(location="Boston")
        with patch("jarvis.agents.weather_watch.get_weather", return_value="Weather for Boston:\nRight now: sunny."):
            result = agent.run_once()
        self.assertEqual(result, "Weather for Boston:")

    def test_weather_error_is_reported_not_raised(self) -> None:
        agent = WeatherWatchAgent()
        with patch("jarvis.agents.weather_watch.get_weather", side_effect=WeatherError("network down")):
            result = agent.run_once()  # must not raise
        self.assertIn("network down", result)

    def test_has_stable_id_and_name(self) -> None:
        agent = WeatherWatchAgent()
        self.assertEqual(agent.id, "weather_watch")
        self.assertEqual(agent.name, "Weather Watch")


if __name__ == "__main__":
    unittest.main()
