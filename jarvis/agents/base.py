"""Background "agents": real tasks that run on their own schedule,
independent of whether the user is actively talking to JARVIS — the live
task cards in the browser dashboard (jarvis/vision/web/hologram.js) reflect
these agents' actual status, not decoration.

Each Agent subclass implements `run_once()` (one unit of real work,
returning a short human-readable result); AgentManager runs each registered
agent on its own background thread at its own interval and tracks live
status (idle/running/complete/error) for broadcasting.
"""

from __future__ import annotations

import threading

IDLE = "idle"
RUNNING = "running"
COMPLETE = "complete"
ERROR = "error"


class Agent:
    """Base class for one background task."""

    def __init__(self, agent_id: str, name: str) -> None:
        self.id = agent_id
        self.name = name

    def run_once(self) -> str:
        """Do one unit of real work and return a short human-readable result.
        Subclasses must implement this. Any exception raised here is caught
        by AgentManager and shown as that agent's error, not raised further —
        one agent failing must never kill the manager or another agent."""
        raise NotImplementedError


class _AgentRuntime:
    def __init__(self, agent: Agent, interval_seconds: float) -> None:
        self.agent = agent
        self.interval_seconds = interval_seconds
        self.thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._status = IDLE
        self._detail = ""

    def set_status(self, status: str, detail: str = "") -> None:
        with self._lock:
            self._status = status
            self._detail = detail

    def snapshot(self) -> dict:
        with self._lock:
            return {"id": self.agent.id, "name": self.agent.name, "status": self._status, "detail": self._detail}


class AgentManager:
    """Runs each registered agent on its own background thread at its own
    interval. Agents keep running whether or not the user is actively
    talking to JARVIS — this is intentionally independent of the
    request/response loop."""

    def __init__(self) -> None:
        self._runtimes: list[_AgentRuntime] = []
        self._stop_event = threading.Event()

    def register(self, agent: Agent, interval_seconds: float) -> None:
        self._runtimes.append(_AgentRuntime(agent, interval_seconds))

    def start(self) -> None:
        self._stop_event.clear()
        for runtime in self._runtimes:
            thread = threading.Thread(target=self._run_loop, args=(runtime,), daemon=True)
            runtime.thread = thread
            thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        self._stop_event.set()
        for runtime in self._runtimes:
            if runtime.thread is not None:
                runtime.thread.join(timeout=timeout)

    def snapshot(self) -> list[dict]:
        """Current status of every registered agent, for broadcasting."""
        return [runtime.snapshot() for runtime in self._runtimes]

    def _run_loop(self, runtime: _AgentRuntime) -> None:
        while not self._stop_event.is_set():
            runtime.set_status(RUNNING, "running...")
            try:
                result = runtime.agent.run_once()
                runtime.set_status(COMPLETE, result)
            except Exception as exc:  # noqa: BLE001 - one agent's failure must never kill the manager
                runtime.set_status(ERROR, str(exc))
            self._stop_event.wait(timeout=runtime.interval_seconds)
