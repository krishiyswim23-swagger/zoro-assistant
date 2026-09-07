"""Stage 5 memory operations, exposed as tool functions bound to a MemoryStore."""

from __future__ import annotations

from jarvis.memory.store import MemoryStore, ProjectNotFoundError


def remember_fact(store: MemoryStore, content: str, tags: list[str] | None = None) -> str:
    store.remember(content, tags)
    return f"Remembered: {content}"


def recall_facts(store: MemoryStore, query: str, limit: int = 5) -> str:
    facts = store.recall(query, limit=limit)
    if not facts:
        return f"No stored memories match '{query}'."
    return "\n".join(f"- {fact['content']}" for fact in facts)


def create_project(store: MemoryStore, name: str) -> str:
    store.create_project(name)
    return f"Project '{name}' is ready."


def add_project_note(store: MemoryStore, project: str, note: str) -> str:
    store.add_project_note(project, note)
    return f"Added a note to project '{project}'."


def get_project_notes(store: MemoryStore, project: str) -> str:
    try:
        notes = store.get_project_notes(project)
    except ProjectNotFoundError:
        return f"No project named '{project}' exists yet."
    if not notes:
        return f"Project '{project}' has no notes yet."
    return "\n".join(f"- {note['content']}" for note in notes)


def list_projects(store: MemoryStore) -> str:
    projects = store.list_projects()
    if not projects:
        return "No projects yet."
    return ", ".join(projects)


def set_preference(store: MemoryStore, key: str, value: str) -> str:
    store.set_preference(key, value)
    return f"Saved preference: {key} = {value}"


def get_preference(store: MemoryStore, key: str) -> str:
    value = store.get_preference(key)
    return value if value is not None else f"No preference set for '{key}'."
