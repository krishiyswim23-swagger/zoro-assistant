"""Stage 6 (brain-graph edition): a thread-safe "how active is each part of
JARVIS's pipeline right now" signal, driving the neural-activity
visualization in jarvis/vision/web/hologram.js.

Each region below maps to a real, existing part of JARVIS — this isn't
neuroscience, it's a brain-inspired label for an actual pipeline stage, so
the visualization reflects what JARVIS is actually doing instead of being
purely decorative:

    prefrontal    -- AIBrain reasoning about a request (jarvis/brain/ai_brain.py)
    association   -- Router matching/routing a request (jarvis/router.py)
    reflex_arc    -- Router's instant pattern-matched commands (no AI brain call)
    sensory       -- input received (speech_input, or hand/gesture tracking)
    language      -- a reply is generated or spoken (ai_brain / speech_output)
    motor         -- a tool or system command actually executes
    memory        -- MemoryStore reads/writes (facts/projects/preferences)
    predictive    -- the weather tool (the one tool that's explicitly forward-looking)
    brainstem     -- baseline "the process is alive" heartbeat, independent of activity
    cerebellum    -- camera/gesture coordination; only ever pulsed if a HandTracker is active

Like SpeakingState/BlinkScheduler, an injectable clock keeps this
unit-testable without real wall-clock waits, and each region decays back to
0 over time rather than staying lit forever after one pulse.
"""

from __future__ import annotations

import threading
import time

REGIONS = (
    "prefrontal",
    "association",
    "reflex_arc",
    "sensory",
    "language",
    "motor",
    "memory",
    "predictive",
    "brainstem",
    "cerebellum",
)

# Regions with no real implementation behind them yet — the UI shows these
# as "planned", never lit, regardless of any pulse() call (defensive: a
# pulse to a not-yet-real region should never falsely read as live).
PLANNED_REGIONS = frozenset({"cerebellum"})

_DECAY_SECONDS = 1.5


class BrainActivity:
    def __init__(self, clock=time.monotonic) -> None:
        self._clock = clock
        self._lock = threading.Lock()
        self._pulsed_at: dict[str, float] = {}
        self._pulsed_amount: dict[str, float] = {}

    def pulse(self, region: str, amount: float = 1.0) -> None:
        """Record activity in `region`, decaying back to 0 over `_DECAY_SECONDS`."""
        if region not in REGIONS or region in PLANNED_REGIONS:
            return
        amount = max(0.0, min(1.0, amount))
        with self._lock:
            self._pulsed_at[region] = self._clock()
            self._pulsed_amount[region] = amount

    def level(self, region: str) -> float:
        """Current activity level for `region`, 0.0 (idle) to 1.0 (just pulsed)."""
        if region in PLANNED_REGIONS:
            return 0.0
        with self._lock:
            pulsed_at = self._pulsed_at.get(region)
            amount = self._pulsed_amount.get(region, 0.0)
        if pulsed_at is None:
            return 0.0
        elapsed = self._clock() - pulsed_at
        return max(0.0, amount * (1.0 - elapsed / _DECAY_SECONDS))

    def snapshot(self) -> dict[str, float]:
        """All regions' current levels, for broadcasting to the browser."""
        return {region: self.level(region) for region in REGIONS}
