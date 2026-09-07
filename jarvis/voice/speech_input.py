"""Stage 2: microphone input + speech recognition.

Captures audio with sounddevice and transcribes it with the SpeechRecognition
library. Raises VoiceUnavailableError if no microphone / recognizer is usable
so callers (main.py) can fall back to typed input instead of crashing.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class VoiceUnavailableError(Exception):
    """Raised when microphone input can't be captured or understood."""


class SpeechInput:
    def __init__(self, pause_threshold: float = 0.8, brain_activity=None) -> None:
        try:
            import speech_recognition as sr
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise VoiceUnavailableError("SpeechRecognition is not installed") from exc

        self._sr = sr
        self._recognizer = sr.Recognizer()
        self._recognizer.pause_threshold = pause_threshold
        self._brain_activity = brain_activity

    def listen(self, timeout: float = 5.0, phrase_time_limit: float = 12.0) -> str:
        """Record one utterance from the default microphone and transcribe it."""
        try:
            with self._sr.Microphone() as source:
                self._recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio = self._recognizer.listen(
                    source, timeout=timeout, phrase_time_limit=phrase_time_limit
                )
        except OSError as exc:  # pragma: no cover - environment dependent
            raise VoiceUnavailableError("No microphone available") from exc
        except (ImportError, AttributeError) as exc:  # pragma: no cover - environment dependent
            # sr.Microphone() only imports pyaudio lazily, at construction time —
            # not when `import speech_recognition` succeeds in __init__ above —
            # so a missing pyaudio install surfaces here, not at SpeechInput().
            # SpeechRecognition itself catches the resulting ModuleNotFoundError
            # and re-raises it as AttributeError("Could not find PyAudio..."),
            # so both exception types need to be treated as "mic unavailable."
            raise VoiceUnavailableError(f"microphone backend unavailable: {exc}") from exc
        except self._sr.WaitTimeoutError as exc:
            raise VoiceUnavailableError("Listening timed out") from exc

        try:
            text = self._recognizer.recognize_google(audio)
            if self._brain_activity is not None:
                self._brain_activity.pulse("sensory")
            return text
        except self._sr.UnknownValueError as exc:
            raise VoiceUnavailableError("Could not understand audio") from exc
        except self._sr.RequestError as exc:
            raise VoiceUnavailableError(f"Speech recognition service error: {exc}") from exc
