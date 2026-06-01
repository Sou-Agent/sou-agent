"""Offline processing (DMN) — Layer 7 autonomy.

The Default Mode Network finding: creative insights and novel goal formation happen
during *undirected rest*, not task execution. This module schedules non-directive
"mind-wandering" sessions during quiet periods.

Fires when:
  - No external signals are present
  - Enough time has passed since the last offline session (default 4h)
  - There has been at least one prior session (to have content to process)

State: last_offline_session tracked in ~/.hermes/autonomy/offline_state.json
"""

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from autonomy.config import get_autonomy_state_dir

logger = logging.getLogger(__name__)

_DEFAULT_INTERVAL_HOURS = 4


def _state_path() -> Path:
    return get_autonomy_state_dir() / "offline_state.json"


def _now() -> datetime:
    from hermes_time import now as _hermes_now
    return _hermes_now()


def _parse_iso(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt
    except (ValueError, TypeError):
        return None


def _load_state() -> Dict[str, Any]:
    path = _state_path()
    try:
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
    except (json.JSONDecodeError, OSError):
        pass
    return {"last_offline_at": None}


def record_offline_session() -> None:
    """Call after an offline session completes to reset the interval timer."""
    path = _state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    from hermes_time import now as _hermes_now
    state = _load_state()
    state["last_offline_at"] = _hermes_now().isoformat(timespec="seconds")
    path.write_text(json.dumps(state, indent=2), encoding="utf-8")


def collect_offline_signal(
    config: Optional[Dict[str, Any]] = None,
    has_other_signals: bool = False,
) -> List[Dict[str, Any]]:
    """Return an offline_due signal when quiet time is available. Never raises."""
    try:
        if has_other_signals:
            return []

        interval_h = float((config or {}).get("offline_session_interval_hours", _DEFAULT_INTERVAL_HOURS))
        state = _load_state()
        last_offline = _parse_iso(state.get("last_offline_at"))
        now = _now()

        if last_offline is None:
            hours_since = None
        else:
            hours_since = (now - last_offline).total_seconds() / 3600.0

        if last_offline is not None and hours_since is not None and hours_since < interval_h:
            return []

        return [{
            "signal_id": "offline:due",
            "type": "offline_due",
            "hours_since_last": round(hours_since, 1) if hours_since is not None else None,
            "content_summary": (
                f"offline processing window available — "
                + (f"{hours_since:.1f}h since last quiet session" if hours_since else "no prior offline session")
            ),
        }]
    except Exception:
        logger.exception("offline_processing: collect failed")
        return []
