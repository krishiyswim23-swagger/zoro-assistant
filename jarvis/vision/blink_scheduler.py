"""Stage 6: schedules idle eye-blinks for the holographic face.

Periodic, with pseudo-random spacing so it doesn't look robotically
metronomic, and driven by an injectable clock (like
jarvis.voice.speaking_state.SpeakingState) so it's unit-testable without
real time passing.
"""

from __future__ import annotations

import time

_BLINK_DURATION_SECONDS = 0.15
_MIN_INTERVAL_SECONDS, _MAX_INTERVAL_SECONDS = 2.5, 5.5
_LCG_A, _LCG_C, _LCG_M = 1103515245, 12345, 0x7FFFFFFF  # same small deterministic PRNG as background.py


class BlinkScheduler:
    def __init__(self, clock=time.monotonic, seed: int = 1) -> None:
        self._clock = clock
        self._prng_state = seed
        self._next_blink_at = self._clock() + self._random_interval()
        self._blink_started_at: float | None = None

    def _random_interval(self) -> float:
        self._prng_state = (self._prng_state * _LCG_A + _LCG_C) & _LCG_M
        fraction = self._prng_state / _LCG_M
        return _MIN_INTERVAL_SECONDS + fraction * (_MAX_INTERVAL_SECONDS - _MIN_INTERVAL_SECONDS)

    def blink_amount(self) -> float:
        """0.0 (eyes open) to 1.0 (eyes fully closed) — call once per render frame."""
        now = self._clock()

        if self._blink_started_at is None:
            if now < self._next_blink_at:
                return 0.0
            self._blink_started_at = now

        elapsed = now - self._blink_started_at
        if elapsed >= _BLINK_DURATION_SECONDS:
            self._blink_started_at = None
            self._next_blink_at = now + self._random_interval()
            return 0.0

        half = _BLINK_DURATION_SECONDS / 2
        if elapsed < half:
            return elapsed / half
        return 1.0 - (elapsed - half) / half
