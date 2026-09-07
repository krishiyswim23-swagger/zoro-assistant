"""Stage 4: file search + document reading.

Both tools are confined to `config.FILES_ROOT` — JARVIS will not read or
list anything outside that directory tree, even if the model is asked (or
tricked, e.g. by something a research tool fetched) into requesting it.
"""

from __future__ import annotations

from pathlib import Path

from jarvis import config

_MAX_CHARS = 8000


class FileAccessError(Exception):
    pass


def _resolve_within_root(relative_path: str) -> Path:
    root = config.FILES_ROOT
    candidate = (root / relative_path).resolve()
    if not candidate.is_relative_to(root):
        raise FileAccessError(f"'{relative_path}' is outside the allowed directory ({root})")
    return candidate


def search_files(query: str, extensions: list[str] | None = None, limit: int = 25) -> list[str]:
    """Find files under FILES_ROOT whose name contains `query` (case-insensitive)."""
    root = config.FILES_ROOT
    if not root.is_dir():
        raise FileAccessError(f"configured files root does not exist: {root}")

    needle = query.lower()
    normalized_exts = {e.lower().lstrip(".") for e in extensions} if extensions else None

    matches: list[str] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if needle not in path.name.lower():
            continue
        if normalized_exts and path.suffix.lower().lstrip(".") not in normalized_exts:
            continue
        matches.append(str(path.relative_to(root)))
        if len(matches) >= limit:
            break
    return matches


def read_document(relative_path: str) -> str:
    """Read a document under FILES_ROOT, extracting text from common formats."""
    path = _resolve_within_root(relative_path)
    if not path.is_file():
        raise FileAccessError(f"no such file: {relative_path}")

    suffix = path.suffix.lower()
    if suffix == ".pdf":
        text = _read_pdf(path)
    elif suffix == ".docx":
        text = _read_docx(path)
    else:
        text = path.read_text(encoding="utf-8", errors="replace")

    if len(text) > _MAX_CHARS:
        text = text[:_MAX_CHARS] + f"\n\n[... truncated, {len(text) - _MAX_CHARS} more characters ...]"
    return text


def _read_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise FileAccessError("reading PDFs requires the 'pypdf' package (pip install pypdf)") from exc
    reader = PdfReader(str(path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def _read_docx(path: Path) -> str:
    try:
        import docx
    except ImportError as exc:
        raise FileAccessError(
            "reading .docx files requires the 'python-docx' package (pip install python-docx)"
        ) from exc
    document = docx.Document(str(path))
    return "\n".join(paragraph.text for paragraph in document.paragraphs)
