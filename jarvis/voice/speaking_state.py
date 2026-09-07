"""Stage 6: a thread-safe "is JARVIS talking right now, and how open should
its mouth be" signal, shared between the TTS engine (writer, on whatever
thread is running the assistant loop) and the hologram's WebSocket broadcast
loop (reader) so the holographic face's mouth can move in sync with actual
speech.

Deliberately has no dependency on pyttsx3 or the renderer — jarvis/voice/
speech_output.py feeds it from TTS engine callbacks, and jarvis/vision/
holographic_web.py reads it each frame. A clock function can be injected so
its timing logic is unit-testable without real wall-clock waits.
"""

from __future__ import annotations

import math
import threading
import time

_PULSE_DECAY_SECONDS = 0.25
_FALLBACK_WOBBLE_RATE = 10.0  # radians/sec, only used when no word-boundary pulses arrive


class SpeakingState:
    def __init__(self, clock=time.monotonic) -> None:
        self._clock = clock
        self._lock = threading.Lock()
        self._is_speaking = False
        self._started_at = 0.0
        self._last_pulse_at: float | None = None

    def set_speaking(self, speaking: bool) -> None:
        with self._lock:
            self._is_speaking = speaking
            self._started_at = self._clock()
            self._last_pulse_at = None

    def pulse(self) -> None:
        """Call on each word/syllable boundary the TTS engine reports, if it reports any."""
        with self._lock:
            self._last_pulse_at = self._clock()

    def is_speaking(self) -> bool:
        with self._lock:
            return self._is_speaking

    def mouth_openness(self) -> float:
        """0.0 (closed) to 1.0 (wide open). 0.0 whenever not speaking."""
        with self._lock:
            if not self._is_speaking:
                return 0.0
            started_at = self._started_at
            last_pulse_at = self._last_pulse_at

        now = self._clock()

        if last_pulse_at is not None:
            # Normal case: the engine reports word boundaries. The mouth
            # opens on each pulse and eases shut again before the next one —
            # naturally at rest (0.0) during ordinary gaps between words.
            since_pulse = now - last_pulse_at
            return max(0.0, 1.0 - since_pulse / _PULSE_DECAY_SECONDS)

        # No word-boundary pulse has arrived since speech started — the
        # engine doesn't report them at all, so keep the mouth moving with a
        # generic wobble rather than freezing it shut for the whole utterance.
        elapsed = now - started_at
        return 0.5 + 0.5 * math.sin(elapsed * _FALLBACK_WOBBLE_RATE)
