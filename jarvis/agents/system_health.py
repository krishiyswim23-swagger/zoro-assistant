"""A background agent that periodically checks whether JARVIS's optional
subsystems have what they need — the AI brain (API key configured), and the
voice/vision packages (importable, not necessarily an open device right
now: this deliberately never opens the mic or camera itself, since another
thread may have one open — see the checks below) — so the dashboard
reflects real, current diagnostic state rather than only what was true at
startup."""

from __future__ import annotations

from jarvis.agents.base import Agent
from jarvis.brain.ai_brain import AIBrain


class SystemHealthAgent(Agent):
    def __init__(self, brain: AIBrain) -> None:
        super().__init__(agent_id="system_health", name="System Health")
        self._brain = brain

    def _check_microphone_library(self) -> bool:
        # Deliberately just checks that the required packages are
        # importable, not that the mic can actually be opened right now: if
        # SpeechInput.listen() has it open on another thread (voice mode),
        # a second open here could contend for the same exclusive device.
        try:
            import pyaudio  # noqa: F401
            import speech_recognition  # noqa: F401
        except ImportError:
            return False
        return True

    def _check_camera_library(self) -> bool:
        # Deliberately just checks that opencv is importable, not that a
        # camera device can actually be opened: if the hologram's own
        # HandTracker has the camera open (run() with real gesture tracking),
        # a second VideoCapture() from this background thread could contend
        # for the same exclusive device handle and glitch the live feed.
        try:
            import cv2  # noqa: F401
        except ImportError:
            return False
        return True

    def run_once(self) -> str:
        brain_ok = self._brain.is_available()
        mic_lib_ok = self._check_microphone_library()
        camera_lib_ok = self._check_camera_library()
        parts = [
            f"brain {'ok' if brain_ok else 'no API key'}",
            f"voice libs {'ok' if mic_lib_ok else 'not installed'}",
            f"vision libs {'ok' if camera_lib_ok else 'not installed'}",
        ]
        return ", ".join(parts)
