"""A background agent that periodically refreshes the weather for a place,
so the dashboard shows a live, real reading rather than only fetching
weather when the user explicitly asks."""

from __future__ import annotations

from jarvis.agents.base import Agent
from jarvis.tools.weather import WeatherError, get_weather


class WeatherWatchAgent(Agent):
    def __init__(self, location: str = "current") -> None:
        super().__init__(agent_id="weather_watch", name="Weather Watch")
        self._location = location

    def run_once(self) -> str:
        try:
            report = get_weather(self._location)
        except WeatherError as exc:
            return f"couldn't refresh weather: {exc}"
        first_line = report.splitlines()[0] if report else "no data"
        return first_line
