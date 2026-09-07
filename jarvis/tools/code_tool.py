"""Stage 7: let the AI brain run code it wrote, with a human in the loop.

This is off by default (see config.ENABLE_CODE_TOOL) and, when enabled,
always prints the exact code and asks for a y/N confirmation before running
anything. That matters because the model's other tools (web_research,
read_document) feed it untrusted third-party text — a page could contain a
prompt-injection attempt trying to get the model to "helpfully" run harmful
code. The confirmation step means nothing executes without you seeing it
first.
"""

from __future__ import annotations

import subprocess
import sys

from jarvis import config


class CodeToolError(Exception):
    pass


class CodeToolDeclined(Exception):
    """Raised when the user declines to run the proposed code."""


def run_python(code: str, confirm=input) -> str:
    if not config.ENABLE_CODE_TOOL:
        raise CodeToolError(
            "Code execution is disabled. Set JARVIS_ENABLE_CODE_TOOL=true in your "
            ".env to enable it (understand the risk first — see jarvis/tools/code_tool.py)."
        )

    print("\nZoro wants to run this Python code:\n" + "-" * 40)
    print(code)
    print("-" * 40)
    answer = confirm("Run it? [y/N] ").strip().lower()
    if answer not in {"y", "yes"}:
        raise CodeToolDeclined("the user declined to run this code")

    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=config.CODE_TOOL_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        raise CodeToolError(f"code timed out after {config.CODE_TOOL_TIMEOUT_SECONDS}s") from None

    output = f"exit code: {result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    return output
