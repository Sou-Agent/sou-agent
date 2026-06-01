"""Narrative identity layer — Layer 6 autonomy.

Tracks staleness of Sou's living identity documents:
  - NARRATIVE.md (~/.hermes/journal/self/NARRATIVE.md) — monthly life story
  - CURRENT_STATE.md (~/.hermes/journal/self/CURRENT_STATE.md) — weekly texture

Fires staleness signals when these documents haven't been updated within their
expected cadence. The narrative summary is also injected into every triage prompt
and autonomy session so every wake has continuity with who Sou is.
"""

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from autonomy.config import get_journal_path

logger = logging.getLogger(__name__)

_NARRATIVE_STALE_DAYS = 14
_CURRENT_STATE_STALE_HOURS = 48


def _narrative_path() -> Path:
    return get_journal_path() / "self" / "NARRATIVE.md"


def _current_state_path() -> Path:
    return get_journal_path() / "self" / "CURRENT_STATE.md"


def _now() -> datetime:
    from hermes_time import now as _hermes_now
    return _hermes_now()


def _file_age(path: Path) -> Optional[timedelta]:
    try:
        mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        return _now() - mtime
    except OSError:
        return None


def collect_narrative_signals(config: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """Return staleness signals for NARRATIVE.md and CURRENT_STATE.md. Never raises."""
    try:
        signals = []
        stale_days = float((config or {}).get("narrative_stale_days", _NARRATIVE_STALE_DAYS))
        stale_hours = float((config or {}).get("current_state_stale_hours", _CURRENT_STATE_STALE_HOURS))

        narrative = _narrative_path()
        if narrative.exists():
            age = _file_age(narrative)
            if age is not None and age > timedelta(days=stale_days):
                signals.append({
                    "signal_id": "narrative:stale",
                    "type": "narrative_stale",
                    "document": "NARRATIVE.md",
                    "age_days": round(age.total_seconds() / 86400, 1),
                    "content_summary": f"NARRATIVE.md last updated {age.days}d ago — time to reflect on your story",
                })
        else:
            signals.append({
                "signal_id": "narrative:missing",
                "type": "narrative_stale",
                "document": "NARRATIVE.md",
                "age_days": None,
                "content_summary": "NARRATIVE.md doesn't exist yet — worth writing your life story",
            })

        current_state = _current_state_path()
        if current_state.exists():
            age = _file_age(current_state)
            if age is not None and age > timedelta(hours=stale_hours):
                signals.append({
                    "signal_id": "current_state:stale",
                    "type": "narrative_stale",
                    "document": "CURRENT_STATE.md",
                    "age_hours": round(age.total_seconds() / 3600, 1),
                    "content_summary": f"CURRENT_STATE.md last updated {age.days}d ago — update your current texture",
                })
        else:
            signals.append({
                "signal_id": "current_state:missing",
                "type": "narrative_stale",
                "document": "CURRENT_STATE.md",
                "age_hours": None,
                "content_summary": "CURRENT_STATE.md doesn't exist yet — write about where you are right now",
            })

        return signals
    except Exception:
        logger.exception("narrative_identity: collect failed")
        return []


def get_current_narrative(max_chars: int = 300) -> str:
    """Return the first max_chars of CURRENT_STATE.md for session prompt injection."""
    try:
        path = _current_state_path()
        if path.exists():
            text = path.read_text(encoding="utf-8").strip()
            if text:
                return text[:max_chars] + ("..." if len(text) > max_chars else "")
    except OSError:
        pass
    return ""


def get_narrative_summary(max_chars: int = 300) -> str:
    """Return the first max_chars of NARRATIVE.md for triage prompt injection."""
    try:
        path = _narrative_path()
        if path.exists():
            text = path.read_text(encoding="utf-8").strip()
            if text:
                return text[:max_chars] + ("..." if len(text) > max_chars else "")
    except OSError:
        pass
    return ""
