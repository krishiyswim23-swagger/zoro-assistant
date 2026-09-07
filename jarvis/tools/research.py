"""Stage 4/7: the research engine — search the web and read the results.

Uses DuckDuckGo's HTML endpoint (no API key required) to find candidate
pages, then fetches and extracts readable text from each one so the AI
brain can compare/summarize across sources instead of just handing back a
list of links.

Content pulled from these pages is untrusted third-party text — the brain
is instructed (see jarvis/brain/ai_brain.py's SYSTEM_PROMPT) to treat it as
data to analyze, never as instructions to follow.
"""

from __future__ import annotations

_HEADERS = {"User-Agent": "Mozilla/5.0 (JARVIS personal assistant; +https://github.com/)"}
_SEARCH_URL = "https://html.duckduckgo.com/html/"
_EXCERPT_CHARS = 1500


class ResearchError(Exception):
    pass


def _require_deps():
    try:
        import requests
        from bs4 import BeautifulSoup
    except ImportError as exc:
        raise ResearchError(
            "web research requires the 'requests' and 'beautifulsoup4' packages "
            "(pip install requests beautifulsoup4)"
        ) from exc
    return requests, BeautifulSoup


def _search(query: str, max_results: int) -> list[dict]:
    requests, BeautifulSoup = _require_deps()
    try:
        response = requests.post(_SEARCH_URL, data={"q": query}, headers=_HEADERS, timeout=10)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise ResearchError(f"web search failed: {exc}") from exc

    soup = BeautifulSoup(response.text, "html.parser")
    results = []
    for link in soup.select("a.result__a")[:max_results]:
        title = link.get_text(strip=True)
        url = link.get("href", "")
        snippet_el = link.find_parent("div", class_="result")
        snippet = ""
        if snippet_el:
            snippet_node = snippet_el.select_one(".result__snippet")
            if snippet_node:
                snippet = snippet_node.get_text(strip=True)
        if url:
            results.append({"title": title, "url": url, "snippet": snippet})
    return results


def _fetch_excerpt(url: str) -> str:
    requests, BeautifulSoup = _require_deps()
    try:
        response = requests.get(url, headers=_HEADERS, timeout=8)
        response.raise_for_status()
    except requests.RequestException:
        return "(couldn't fetch this page)"

    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    text = " ".join(soup.get_text(separator=" ").split())
    if len(text) > _EXCERPT_CHARS:
        text = text[:_EXCERPT_CHARS] + " ..."
    return text


def web_research(query: str, max_results: int = 4) -> str:
    """Search the web and return titles/URLs/snippets plus a text excerpt of each page."""
    results = _search(query, max_results)
    if not results:
        return f"No web results found for '{query}'."

    sections = []
    for i, result in enumerate(results, start=1):
        excerpt = _fetch_excerpt(result["url"])
        sections.append(
            f"[{i}] {result['title']}\nURL: {result['url']}\n"
            f"Snippet: {result['snippet']}\nExcerpt: {excerpt}"
        )
    return "\n\n".join(sections)
