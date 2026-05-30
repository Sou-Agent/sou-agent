"""People & channel watch-lists — ``{journal}/people/*.yaml`` and ``channels/*.yaml``.

These are the agent's own curated lists of who and what she keeps up with. They
are authoritative for the autonomy collector: a person/channel matters only if
its file has ``watch: true``. The agent manages these herself during sessions
(via the journal/terminal tools); this module just reads and minimally updates
them.

Person entry::

    name: Bailey
    contact: Bailey            # resolves via contacts.json
    watch: true
    silence_threshold_hours: 36   # optional per-person override
    last_seen: 2026-05-30T14:00:00Z
    notes: "..."

Channel entry::

    channel_id: "123456789"
    name: homebase-general
    watch: true
    description: "..."
    notes: "..."
"""

import logging
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from autonomy.config import get_journal_path

logger = logging.getLogger(__name__)

_WATCH_LOCK = threading.RLock()


def _people_dir() -> Path:
    return get_journal_path() / "people"


def _channels_dir() -> Path:
    return get_journal_path() / "channels"


def _read_yaml(path: Path) -> Optional[Dict[str, Any]]:
    try:
        import yaml
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data
    except Exception as e:
        logger.debug("watchlists: failed to read %s: %s", path, e)
    return None


def _write_yaml(path: Path, data: Dict[str, Any]) -> None:
    import yaml
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")


def _load_dir(directory: Path) -> List[Tuple[Path, Dict[str, Any]]]:
    out: List[Tuple[Path, Dict[str, Any]]] = []
    if not directory.is_dir():
        return out
    with _WATCH_LOCK:
        for p in sorted(directory.glob("*.yaml")):
            data = _read_yaml(p)
            if data is not None:
                out.append((p, data))
    return out


def load_people(watched_only: bool = True) -> List[Tuple[Path, Dict[str, Any]]]:
    entries = _load_dir(_people_dir())
    if watched_only:
        entries = [(p, d) for p, d in entries if d.get("watch") is True]
    return entries


def load_channels(watched_only: bool = True) -> List[Tuple[Path, Dict[str, Any]]]:
    entries = _load_dir(_channels_dir())
    if watched_only:
        entries = [(p, d) for p, d in entries if d.get("watch") is True]
    return entries


def find_person(name_or_contact: str) -> Optional[Tuple[Path, Dict[str, Any]]]:
    """Match a person by name, contact ref, or filename stem (case-insensitive)."""
    key = (name_or_contact or "").strip().lower()
    if not key:
        return None
    for path, data in _load_dir(_people_dir()):
        candidates = {
            str(data.get("name", "")).lower(),
            str(data.get("contact", "")).lower(),
            path.stem.lower(),
        }
        aliases = data.get("aliases")
        if isinstance(aliases, list):
            candidates.update(str(a).lower() for a in aliases)
        if key in candidates and key:
            return path, data
    return None


def set_last_seen(name_or_contact: str, when_iso: str) -> bool:
    """Update ``last_seen`` on a person entry. Returns True if a file was written."""
    with _WATCH_LOCK:
        found = find_person(name_or_contact)
        if found is None:
            return False
        path, data = found
        data["last_seen"] = when_iso
        _write_yaml(path, data)
        return True
