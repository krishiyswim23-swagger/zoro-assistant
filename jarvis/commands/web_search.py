"""Stage 1: search the web from a spoken/typed query."""

from __future__ import annotations

import webbrowser
from urllib.parse import quote_plus


def search_web(query: str) -> str:
    """Open the default browser on a search results page. Returns a status message."""
    url = f"https://www.google.com/search?q={quote_plus(query)}"
    webbrowser.open(url)
    return f"Searching the web for {query}."
