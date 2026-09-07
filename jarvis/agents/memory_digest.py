"""A background agent that periodically reports on the state of long-term
memory — how many facts and projects exist. Deliberately doesn't call the
AI brain itself (no summarization via Gemini): an autonomous background
task silently spending API quota/rate-limit budget on its own schedule
would be surprising, so this reports plain, real counts instead."""

from __future__ import annotations

from jarvis.agents.base import Agent
from jarvis.memory.store import MemoryStore


class MemoryDigestAgent(Agent):
    def __init__(self, memory_store: MemoryStore) -> None:
        super().__init__(agent_id="memory_digest", name="Memory Digest")
        self._memory = memory_store

    def run_once(self) -> str:
        facts = self._memory.fact_count()
        projects = self._memory.list_projects()
        project_part = f"{len(projects)} project(s)" if projects else "no projects yet"
        return f"{facts} fact(s) remembered, {project_part}"
