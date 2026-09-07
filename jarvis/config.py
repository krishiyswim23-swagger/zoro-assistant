"""Runtime configuration for JARVIS, loaded from environment / .env."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
JARVIS_MODEL = os.getenv("JARVIS_MODEL", "gemini-3.6-flash")

# How many past turns (user + assistant) the AI brain keeps as short-term memory.
MAX_HISTORY_TURNS = int(os.getenv("JARVIS_MAX_HISTORY_TURNS", "20"))

# Stage 4: safety bound on how many tool-call round-trips the agentic loop
# will make for a single request before it has to give a final answer anyway.
MAX_TOOL_ITERATIONS = int(os.getenv("JARVIS_MAX_TOOL_ITERATIONS", "8"))

# Stage 4: file tools (search_files/read_document) are confined to this
# directory tree — JARVIS will never read or list files outside of it,
# even if asked to. Defaults to the user's Documents folder if it exists,
# otherwise their home directory.
_default_files_root = Path.home() / "Documents"
if not _default_files_root.is_dir():
    _default_files_root = Path.home()
FILES_ROOT = Path(os.getenv("JARVIS_FILES_ROOT", str(_default_files_root))).expanduser().resolve()

# Stage 4: the run_python tool lets the AI brain execute code it wrote. It is
# off by default because a tool result from web_research/read_document is
# untrusted third-party text the model reads — a prompt-injection attempt
# there could otherwise try to talk the model into running something
# harmful. Turn it on only if you understand and accept that trade-off.
ENABLE_CODE_TOOL = os.getenv("JARVIS_ENABLE_CODE_TOOL", "false").lower() in {"1", "true", "yes"}
CODE_TOOL_TIMEOUT_SECONDS = int(os.getenv("JARVIS_CODE_TOOL_TIMEOUT", "10"))

# Stage 5: where persistent memory (facts/projects/preferences) is stored.
DATA_DIR = Path(os.getenv("JARVIS_DATA_DIR", str(Path.home() / ".jarvis"))).expanduser()
MEMORY_DB_PATH = str(DATA_DIR / "memory.db")
