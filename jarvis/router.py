"""Routes a request to a known Stage 1 command, or falls back to the Stage 3 AI brain.

This is the piece that replaces the old hardcoded

    if command == "open chrome": ...

dispatch: a short list of regexes still catches the handful of actions
JARVIS can *do* on your machine (open an app, tell the time, search the
web, lock the screen), and anything that doesn't match one of those is
handed to the general-purpose AI brain instead of falling through to
"sorry, I didn't understand that."
"""

from __future__ import annotations

import re

from jarvis.brain.ai_brain import AIBrain, BrainError, BrainUnavailableError
from jarvis.commands.system_commands import ApplicationNotFoundError, get_date, get_time, lock_workstation, open_application
from jarvis.commands.web_search import search_web

EXIT_PHRASES = {"exit", "quit", "goodbye", "bye", "shut down jarvis", "that's all jarvis"}
RESET_PHRASES = {"forget this conversation", "forget our conversation", "clear your memory", "new topic", "let's start over"}

_PATTERNS = [
    (re.compile(r"^(?:open|launch|start)\s+(?:the\s+)?(.+)$", re.I), "open"),
    (re.compile(r"^(?:search|google)\s+(?:the web\s+)?for\s+(.+)$", re.I), "search"),
    (re.compile(r"^look up\s+(.+)$", re.I), "search"),
    (re.compile(r"^what(?:'s| is)\s+the\s+time\b", re.I), "time"),
    (re.compile(r"^what\s+time\s+is\s+it\b", re.I), "time"),
    (re.compile(r"^what(?:'s| is)\s+(?:the\s+)?(?:today'?s\s+)?date\b", re.I), "date"),
    (re.compile(r"^what\s+day\s+is\s+it\b", re.I), "date"),
    (re.compile(r"^lock\s+(?:my\s+)?(?:computer|workstation|screen|pc)\b", re.I), "lock"),
]


class ExitRequested(Exception):
    """Raised when the user asks to end the session."""


class Router:
    def __init__(self, brain: AIBrain, brain_activity=None) -> None:
        self._brain = brain
        self._brain_activity = brain_activity

    def handle(self, text: str) -> str:
        text = text.strip()
        if not text:
            return ""

        if self._brain_activity is not None:
            self._brain_activity.pulse("association")

        lowered = text.lower().rstrip(".!?")
        if lowered in EXIT_PHRASES:
            raise ExitRequested()
        if lowered in RESET_PHRASES:
            self._brain.reset()
            return "Alright, fresh start — what's next?"

        for pattern, kind in _PATTERNS:
            match = pattern.match(text)
            if not match:
                continue
            if self._brain_activity is not None:
                self._brain_activity.pulse("reflex_arc")
            return self._dispatch(kind, match)

        return self._ask_brain(text)

    def _dispatch(self, kind: str, match: re.Match) -> str:
        if kind == "open":
            name = match.group(1).strip()
            try:
                return open_application(name)
            except ApplicationNotFoundError:
                return f"I don't know how to open {name} yet — I'll ask the AI brain instead.\n" + self._ask_brain(
                    f"The user asked to open the application '{name}', which isn't in my "
                    "launcher list. Briefly let them know and, if you can, tell them how "
                    "they'd normally open it themselves."
                )
        if kind == "search":
            return search_web(match.group(1).strip())
        if kind == "time":
            return f"It's {get_time()}."
        if kind == "date":
            return f"Today is {get_date()}."
        if kind == "lock":
            return lock_workstation()
        raise AssertionError(f"unhandled intent kind: {kind}")

    def _ask_brain(self, text: str) -> str:
        try:
            return self._brain.think(text)
        except BrainUnavailableError as exc:
            return str(exc)
        except BrainError as exc:
            return str(exc)
