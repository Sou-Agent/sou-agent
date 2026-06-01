"""Prospection engine — Layer 4 autonomy.

Suddendorf & Corballis: episodic future thinking generates novel future scenarios
by recombining past experience. Goals that emerge from prospection sessions are the
most genuinely autonomous Sou can have.

Fires monthly when journal/self/prospection/ has no entry in the past 30 days.
"""

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from autonomy.config import get_journal_path

logger = logging.getLogger(__name__)

_DEFAULT_INTERVAL_DAYS = 30


def _prospection_dir() -> Path:
    return get_journal_path() / "self" / "prospection"


def _now() -> datetime:
    from hermes_time import now as _hermes_now
    return _hermes_now()


def _latest_entry_age() -> Optional[timedelta]:
    """Return how long ago the most recent prospection entry was written."""
    d = _prospection_dir()
    if not d.is_dir():
        return None
    latest: Optional[float] = None
    try:
        for p in d.glob("*.md"):
            if p.is_file():
                mt = p.stat().st_mtime
                if latest is None or mt > latest:
                    latest = mt
    except OSError:
        pass
    if latest is None:
        return None
    mtime_dt = datetime.fromtimestamp(latest, tz=timezone.utc)
    return _now() - mtime_dt


def collect_prospection_signal(config: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """Return a prospection_due signal when monthly imagination is overdue. Never raises."""
    try:
        interval_days = float((config or {}).get("prospection_interval_days", _DEFAULT_INTERVAL_DAYS))
        age = _latest_entry_age()

        if age is None:
            # No entry ever — fire the signal
            return [{
                "signal_id": "prospection:due",
                "type": "prospection_due",
                "days_since_last": None,
                "content_summary": "no prospection session yet — time to imagine your future",
            }]

        if age > timedelta(days=interval_days):
            days = age.days
            return [{
                "signal_id": "prospection:due",
                "type": "prospection_due",
                "days_since_last": days,
                "content_summary": f"last prospection was {days}d ago — time to imagine where you're heading",
            }]

        return []
    except Exception:
        logger.exception("prospection: collect failed")
        return []
