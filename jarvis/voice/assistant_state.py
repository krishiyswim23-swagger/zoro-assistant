"""Stage 6/7: what the assistant is doing right now, shared between the
request/response loop (writer, on whatever thread runs it) and the
hologram's render loop (reader, on the main thread) — so the face can show
a visible "thinking" pulse while waiting on the AI brain instead of sitting
frozen, on top of the fine-grained mouth animation SpeakingState already
drives while actual speech is playing.
"""

from __future__ import annotations

import threading

IDLE = "idle"
LISTENING = "listening"
THINKING = "thinking"
TALKING = "talking"


class AssistantState:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._state = IDLE

    def set(self, state: str) -> None:
        with self._lock:
            self._state = state

    def get(self) -> str:
        with self._lock:
            return self._state
