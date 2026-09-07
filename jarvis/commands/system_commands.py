"""Stage 1: basic desktop actions — opening apps, time/date, safe system actions."""

from __future__ import annotations

import datetime
import os
import platform
import shutil
import subprocess

# Aliases -> the token used to actually launch the app on each platform.
_WINDOWS_APPS = {
    "chrome": "chrome",
    "google chrome": "chrome",
    "edge": "msedge",
    "microsoft edge": "msedge",
    "firefox": "firefox",
    "notepad": "notepad",
    "calculator": "calc",
    "calc": "calc",
    "explorer": "explorer",
    "file explorer": "explorer",
    "files": "explorer",
    "word": "winword",
    "excel": "excel",
    "powerpoint": "powerpnt",
    "spotify": "spotify",
    "vscode": "code",
    "vs code": "code",
    "code": "code",
    "cmd": "cmd",
    "command prompt": "cmd",
    "terminal": "wt",
    "paint": "mspaint",
}

_MACOS_APPS = {
    "chrome": "Google Chrome",
    "google chrome": "Google Chrome",
    "safari": "Safari",
    "firefox": "Firefox",
    "notepad": "TextEdit",
    "textedit": "TextEdit",
    "calculator": "Calculator",
    "calc": "Calculator",
    "finder": "Finder",
    "files": "Finder",
    "word": "Microsoft Word",
    "excel": "Microsoft Excel",
    "powerpoint": "Microsoft PowerPoint",
    "spotify": "Spotify",
    "vscode": "Visual Studio Code",
    "vs code": "Visual Studio Code",
    "code": "Visual Studio Code",
    "terminal": "Terminal",
}

_LINUX_APPS = {
    "chrome": ["google-chrome", "google-chrome-stable", "chromium", "chromium-browser"],
    "google chrome": ["google-chrome", "google-chrome-stable", "chromium"],
    "firefox": ["firefox"],
    "notepad": ["gedit", "kate", "nano"],
    "calculator": ["gnome-calculator", "kcalc", "xcalc"],
    "calc": ["gnome-calculator", "kcalc", "xcalc"],
    "files": ["nautilus", "dolphin", "pcmanfm"],
    "explorer": ["nautilus", "dolphin", "pcmanfm"],
    "vscode": ["code"],
    "vs code": ["code"],
    "code": ["code"],
    "terminal": ["gnome-terminal", "konsole", "xterm"],
    "spotify": ["spotify"],
}


class ApplicationNotFoundError(Exception):
    """Raised when JARVIS doesn't know how to open the requested application."""


def open_application(name: str) -> str:
    """Launch a desktop application by common name. Returns a status message."""
    key = name.strip().lower()
    system = platform.system()

    if system == "Windows":
        target = _WINDOWS_APPS.get(key)
        if not target:
            raise ApplicationNotFoundError(key)
        os.startfile(target)  # noqa: S606 - user-requested, known safe token
        return f"Opening {name}."

    if system == "Darwin":
        target = _MACOS_APPS.get(key)
        if not target:
            raise ApplicationNotFoundError(key)
        subprocess.Popen(["open", "-a", target])
        return f"Opening {name}."

    # Linux and anything else: try each known candidate binary in turn.
    candidates = _LINUX_APPS.get(key, [key])
    for binary in candidates:
        path = shutil.which(binary)
        if path:
            subprocess.Popen([path])
            return f"Opening {name}."
    raise ApplicationNotFoundError(key)


def get_time() -> str:
    return datetime.datetime.now().strftime("%I:%M %p").lstrip("0")


def get_date() -> str:
    return datetime.datetime.now().strftime("%A, %B %d, %Y")


def lock_workstation() -> str:
    """Lock the screen. Safe and reversible (unlocking just needs the password)."""
    system = platform.system()
    if system == "Windows":
        import ctypes

        ctypes.windll.user32.LockWorkStation()
        return "Locking the workstation."
    if system == "Darwin":
        subprocess.Popen(
            ["/System/Library/CoreServices/Menu Extras/User.menu/Contents/Resources/CGSession", "-suspend"]
        )
        return "Locking the workstation."
    subprocess.Popen(["loginctl", "lock-session"])
    return "Locking the workstation."
