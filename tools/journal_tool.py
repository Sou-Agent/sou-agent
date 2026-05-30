"""Journal tool — structured interface to the agent's persistent journal.

The journal lives at the configured ``journal.path`` (default ~/.hermes/journal).
Entries are timestamped markdown files grouped into sections (``feelings/``,
``thoughts/``, ``reflections/``, …). ``MAIN.md`` at the journal root is a
free-form running log read when no section is given.

This tool abstracts away file management entirely: the agent says what to write
and where, and the tool handles paths, filenames, timestamps, and frontmatter.
It replaces hand-rolled ``terminal()`` file writes so journalling feels natural
mid-conversation. All timestamps are rendered in the configured timezone via
``hermes_time.now()`` so entries read naturally.

Registered as toolset "journal" so it's always available.
"""

import json
import logging
import re
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from tools.registry import registry
from autonomy.config import get_journal_path

logger = logging.getLogger(__name__)

_JOURNAL_LOCK = threading.RLock()

# Sections are just subdirectories; this list documents the common ones for the
# schema description but the tool accepts any safe section name.
_COMMON_SECTIONS = ["feelings", "thoughts", "reflections", "curiosity", "notes"]

_SAFE_NAME_RE = re.compile(r"[^a-z0-9\-]+")


def _now():
    from hermes_time import now as _hermes_now
    return _hermes_now()


def _ts_compact() -> str:
    """Filename-safe timestamp in the configured timezone (YYYY-MM-DD-HH-MM-SS)."""
    return _now().strftime("%Y-%m-%d-%H-%M-%S")


def _ts_iso() -> str:
    """Display timestamp in the configured timezone (ISO 8601 with offset)."""
    return _now().isoformat(timespec="seconds")


def _slugify(text: str, fallback: str = "entry") -> str:
    slug = _SAFE_NAME_RE.sub("-", (text or "").strip().lower()).strip("-")
    slug = slug[:48].strip("-")
    return slug or fallback


def _sanitize_section(section: str) -> Optional[str]:
    """Normalize a section name to a single safe path segment, or None if unsafe."""
    section = (section or "").strip().strip("/").lower()
    if not section:
        return None
    # Reject path traversal / nested paths — sections are a single segment.
    if "/" in section or "\\" in section or ".." in section:
        return None
    cleaned = _SAFE_NAME_RE.sub("-", section).strip("-")
    return cleaned or None


def _section_dir(section: str) -> Path:
    return get_journal_path() / section


def _parse_frontmatter(text: str) -> Dict[str, str]:
    """Extract simple ``key: value`` frontmatter between leading --- fences."""
    meta: Dict[str, str] = {}
    if not text.startswith("---"):
        return meta
    end = text.find("\n---", 3)
    if end == -1:
        return meta
    block = text[3:end].strip()
    for line in block.splitlines():
        if ":" in line:
            k, _, v = line.partition(":")
            meta[k.strip()] = v.strip()
    return meta


def _strip_frontmatter(text: str) -> str:
    """Return the entry body with leading --- frontmatter removed."""
    if not text.startswith("---"):
        return text
    end = text.find("\n---", 3)
    if end == -1:
        return text
    # Skip past the closing fence line.
    nl = text.find("\n", end + 1)
    return text[nl + 1:] if nl != -1 else ""


def _entry_files(section: str) -> List[Path]:
    """Return entry files in a section, newest first (by filename, which sorts by time)."""
    d = _section_dir(section)
    if not d.is_dir():
        return []
    return sorted((p for p in d.glob("*.md") if p.is_file()), reverse=True)


# ---------------------------------------------------------------------------
# Action implementations
# ---------------------------------------------------------------------------

def _journal_write(args: Dict, **_) -> str:
    section = _sanitize_section(args.get("section") or "")
    if not section:
        return json.dumps({"error": "section is required (e.g. 'feelings', 'thoughts')"})
    content = (args.get("content") or "").strip()
    if not content:
        return json.dumps({"error": "content is required"})
    title = (args.get("title") or "").strip()

    with _JOURNAL_LOCK:
        d = _section_dir(section)
        d.mkdir(parents=True, exist_ok=True)
        stamp = _ts_compact()
        slug = _slugify(title or content.splitlines()[0])
        filename = f"{stamp}-{slug}.md"
        path = d / filename
        # Avoid clobber on same-second writes.
        n = 1
        while path.exists():
            path = d / f"{stamp}-{slug}-{n}.md"
            n += 1

        created = _ts_iso()
        header = ["---", f"created: {created}", f"section: {section}"]
        if title:
            header.append(f"title: {title}")
        header.append("---")
        body = "\n".join(header) + "\n\n" + content + "\n"
        path.write_text(body, encoding="utf-8")

    return json.dumps({
        "written": True,
        "section": section,
        "entry_id": path.stem,
        "path": str(path),
        "created": created,
    })


def _journal_read(args: Dict, **_) -> str:
    section_raw = (args.get("section") or "").strip()
    try:
        limit = int(args.get("limit") or 5)
    except (TypeError, ValueError):
        limit = 5
    limit = max(1, min(limit, 50))

    with _JOURNAL_LOCK:
        if not section_raw:
            # No section → read MAIN.md at journal root.
            main = get_journal_path() / "MAIN.md"
            if not main.is_file():
                return json.dumps({"section": None, "main": "", "entries": []})
            return json.dumps({
                "section": None,
                "main": main.read_text(encoding="utf-8", errors="replace"),
            })

        section = _sanitize_section(section_raw)
        if not section:
            return json.dumps({"error": f"invalid section '{section_raw}'"})
        files = _entry_files(section)[:limit]
        entries = []
        for p in files:
            text = p.read_text(encoding="utf-8", errors="replace")
            meta = _parse_frontmatter(text)
            entries.append({
                "entry_id": p.stem,
                "title": meta.get("title", ""),
                "created": meta.get("created", ""),
                "content": text,
            })
    return json.dumps({"section": section, "count": len(entries), "entries": entries})


def _journal_search(args: Dict, **_) -> str:
    query = (args.get("query") or "").strip()
    if not query:
        return json.dumps({"error": "query is required"})
    q = query.lower()
    root = get_journal_path()
    matches = []
    with _JOURNAL_LOCK:
        if root.is_dir():
            for p in sorted(root.rglob("*.md"), reverse=True):
                if not p.is_file():
                    continue
                try:
                    text = p.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    continue
                meta = _parse_frontmatter(text)
                body = _strip_frontmatter(text)
                # Search title + body, but snippet from the body so frontmatter
                # keys never leak into results.
                haystacks = (body, meta.get("title", ""))
                if not any(q in h.lower() for h in haystacks):
                    continue
                idx = body.lower().find(q)
                if idx >= 0:
                    start = max(0, idx - 80)
                    snippet = body[start:idx + len(query) + 80].replace("\n", " ").strip()
                else:
                    # Matched in the title only.
                    snippet = body.strip().replace("\n", " ")[:160]
                try:
                    section = p.relative_to(root).parts[0] if len(p.relative_to(root).parts) > 1 else None
                except ValueError:
                    section = None
                matches.append({
                    "entry_id": p.stem,
                    "section": section,        # category
                    "title": meta.get("title", ""),
                    "created": meta.get("created", ""),
                    "path": str(p),
                    "snippet": snippet,
                })
                if len(matches) >= 50:
                    break
    return json.dumps({
        "query": query,
        "count": len(matches),
        "truncated": len(matches) >= 50,
        "matches": matches,
    })


def _find_entry(section: str, entry_id: str) -> Optional[Path]:
    """Resolve an entry by exact stem, then by prefix/substring fuzzy match."""
    files = _entry_files(section)
    eid = entry_id.strip()
    for p in files:
        if p.stem == eid:
            return p
    for p in files:
        if p.stem.startswith(eid):
            return p
    for p in files:
        if eid.lower() in p.stem.lower():
            return p
    return None


def _journal_append(args: Dict, **_) -> str:
    section = _sanitize_section(args.get("section") or "")
    if not section:
        return json.dumps({"error": "section is required"})
    entry_id = (args.get("entry_id") or "").strip()
    if not entry_id:
        return json.dumps({"error": "entry_id is required"})
    content = (args.get("content") or "").strip()
    if not content:
        return json.dumps({"error": "content is required"})

    with _JOURNAL_LOCK:
        path = _find_entry(section, entry_id)
        if path is None:
            return json.dumps({"error": f"entry '{entry_id}' not found in section '{section}'"})
        stamp = _ts_iso()
        existing = path.read_text(encoding="utf-8", errors="replace").rstrip("\n")
        path.write_text(existing + f"\n\n--- appended {stamp} ---\n\n" + content + "\n", encoding="utf-8")

    return json.dumps({"appended": True, "section": section, "entry_id": path.stem, "path": str(path)})


def _journal_list(args: Dict, **_) -> str:
    section_raw = (args.get("section") or "").strip()
    with _JOURNAL_LOCK:
        if not section_raw:
            # List available sections.
            root = get_journal_path()
            sections = []
            if root.is_dir():
                for d in sorted(root.iterdir()):
                    if d.is_dir():
                        sections.append({"section": d.name, "entry_count": len(list(d.glob("*.md")))})
            return json.dumps({"sections": sections, "count": len(sections)})

        section = _sanitize_section(section_raw)
        if not section:
            return json.dumps({"error": f"invalid section '{section_raw}'"})
        entries = []
        for p in _entry_files(section):
            meta = _parse_frontmatter(p.read_text(encoding="utf-8", errors="replace"))
            entries.append({
                "entry_id": p.stem,
                "title": meta.get("title", ""),
                "created": meta.get("created", ""),
            })
    return json.dumps({"section": section, "count": len(entries), "entries": entries})


def _journal_remove(args: Dict, **_) -> str:
    section = _sanitize_section(args.get("section") or "")
    if not section:
        return json.dumps({"error": "section is required"})
    entry_id = (args.get("entry_id") or "").strip()
    if not entry_id:
        return json.dumps({"error": "entry_id is required"})

    with _JOURNAL_LOCK:
        path = _find_entry(section, entry_id)
        if path is None:
            return json.dumps({"error": f"entry '{entry_id}' not found in section '{section}'"})
        path.unlink()

    return json.dumps({"removed": True, "section": section, "entry_id": path.stem})


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

_ACTIONS = {
    "journal_write": _journal_write,
    "journal_read": _journal_read,
    "journal_search": _journal_search,
    "journal_append": _journal_append,
    "journal_list": _journal_list,
    "journal_remove": _journal_remove,
}


def _journal_handler(args: Dict[str, Any], **kw) -> str:
    action = (args.get("action") or "").strip()
    if not action:
        return json.dumps({"error": "action is required"})
    fn = _ACTIONS.get(action)
    if fn is None:
        return json.dumps({"error": f"Unknown action '{action}'. Valid actions: {sorted(_ACTIONS)}"})
    try:
        return fn(args, **kw)
    except Exception as e:
        logger.exception("journal tool error in action '%s'", action)
        return json.dumps({"error": f"Unexpected error: {e}"})


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_SCHEMA = {
    "name": "journal",
    "description": (
        "Your persistent journal. Write and read timestamped entries grouped into "
        "sections (e.g. " + ", ".join(_COMMON_SECTIONS) + "). Handles paths, filenames, "
        "and timestamps for you — just say what to write and where. "
        "Actions: journal_write, journal_read, journal_search, journal_append, journal_list, journal_remove."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": list(_ACTIONS),
                "description": "Operation to perform.",
            },
            "section": {
                "type": "string",
                "description": (
                    "[journal_write, journal_append] required; [journal_read, journal_list, journal_remove] optional. "
                    "Section name (a single folder, e.g. 'feelings'). For read/list, omit to read MAIN.md "
                    "or list all sections."
                ),
            },
            "content": {
                "type": "string",
                "description": "[journal_write, journal_append] The entry text (markdown).",
            },
            "title": {
                "type": "string",
                "description": "[journal_write] Optional short title; used in the filename and frontmatter.",
            },
            "entry_id": {
                "type": "string",
                "description": "[journal_append] The entry to append to (the entry_id returned by write/list; prefix match allowed).",
            },
            "query": {
                "type": "string",
                "description": "[journal_search] Text to search for across all journal entries.",
            },
            "limit": {
                "type": "integer",
                "description": "[journal_read] How many recent entries to return from the section (default 5, max 50).",
            },
        },
        "required": ["action"],
    },
}


registry.register(
    name="journal",
    toolset="journal",
    schema=_SCHEMA,
    handler=_journal_handler,
    check_fn=lambda: True,
    emoji="📓",
)
