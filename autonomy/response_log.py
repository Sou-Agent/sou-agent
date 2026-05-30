"""Response log — tracks what the agent has already acted on autonomously.

Without this, the collector re-surfaces the same signals every cycle: a message
already responded to keeps reappearing, and a message deliberately *held* would
vanish once the channel watermark passes it.

The log lives at ``<HERMES_HOME>/autonomy/response_log.json`` and records one
entry per (signal, decision):

  - ``respond`` / ``self_reflect`` / ``ignore`` → signal id marked handled,
    never re-surfaced.
  - ``hold`` → recorded with ``re_evaluate_after``. The channel watermark still
    advances (so we don't re-fetch from the API), but ``get_held_signals()``
    re-injects the held signal into the snapshot once that time passes, tagged
    so the aux model knows she already chose to sit with it.

Signal id formats:
  - Discord message:   ``discord:{channel_id}/msg:{message_id}``
  - Intent:            ``intent:{uuid}``
  - Contact silence:   ``contact:{name}``
  - Curiosity thread:  ``curiosity:{hash}``
"""

import json
import logging
import threading
from datetime import timedelta
from typing import Any, Dict, List, Optional, Set

from autonomy.config import get_autonomy_state_dir

logger = logging.getLogger(__name__)

_LOG_LOCK = threading.RLock()
_PRUNE_DAYS = 30

# Decisions that permanently consume a signal (vs. hold, which re-surfaces).
_HANDLED_ACTIONS = {"respond", "self_reflect", "ignore"}


def _log_path():
    return get_autonomy_state_dir() / "response_log.json"


def _now():
    from hermes_time import now as _hermes_now
    return _hermes_now()


def _parse_iso(value: str):
    if not value:
        return None
    try:
        from datetime import datetime
        return datetime.fromisoformat(value)
    except (ValueError, TypeError):
        return None


def _load() -> List[Dict[str, Any]]:
    path = _log_path()
    try:
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return data
    except (json.JSONDecodeError, OSError) as e:
        logger.debug("response_log: failed to read %s: %s", path, e)
    return []


def _save(entries: List[Dict[str, Any]]) -> None:
    path = _log_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entries, indent=2), encoding="utf-8")


def _prune(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Drop entries older than _PRUNE_DAYS (by decided_at)."""
    cutoff = _now() - timedelta(days=_PRUNE_DAYS)
    kept = []
    for e in entries:
        decided = _parse_iso(e.get("decided_at", ""))
        if decided is None or decided >= cutoff:
            kept.append(e)
    return kept


def get_handled_ids() -> Set[str]:
    """Return signal ids that have been permanently acted on (respond/reflect/ignore).

    Held signals are intentionally excluded — they are meant to come back.
    """
    handled: Set[str] = set()
    with _LOG_LOCK:
        for e in _load():
            if e.get("action") in _HANDLED_ACTIONS and e.get("signal_id"):
                handled.add(e["signal_id"])
    return handled


def get_held_signals() -> List[Dict[str, Any]]:
    """Return held signals whose ``re_evaluate_after`` has passed.

    Each is tagged ``held=True`` with ``held_since`` so the aux model knows the
    agent already chose to sit with it. A signal later acted on (respond/ignore/
    self_reflect) is suppressed even if an older hold entry exists.
    """
    now = _now()
    handled = get_handled_ids()
    out: List[Dict[str, Any]] = []
    seen: Set[str] = set()
    with _LOG_LOCK:
        # Walk newest-first so the latest hold for a signal wins.
        for e in reversed(_load()):
            if e.get("action") != "hold":
                continue
            sid = e.get("signal_id")
            if not sid or sid in seen or sid in handled:
                continue
            reeval = _parse_iso(e.get("re_evaluate_after", ""))
            if reeval is not None and reeval > now:
                continue  # not yet time to reconsider
            seen.add(sid)
            out.append({
                "signal_id": sid,
                "signal_type": e.get("signal_type"),
                "source_summary": e.get("source_summary", ""),
                "held_since": e.get("decided_at", ""),
                "hold_reason": e.get("hold_reason", ""),
                "held": True,
            })
    return out


def record_decision(
    signal_id: str,
    action: str,
    signal_type: Optional[str] = None,
    source_summary: str = "",
    session_id: Optional[str] = None,
    hold_reason: str = "",
    re_evaluate_after: Optional[str] = None,
    linked_intent_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Append one decision to the log and return the stored entry."""
    entry: Dict[str, Any] = {
        "signal_id": signal_id,
        "signal_type": signal_type,
        "source_summary": source_summary,
        "action": action,
        "decided_at": _now().isoformat(timespec="seconds"),
    }
    if session_id:
        entry["session_id"] = session_id
    if action == "hold":
        if re_evaluate_after is None:
            from autonomy.config import get_autonomy_config
            hours = get_autonomy_config().get("hold_reevaluate_hours", 6)
            re_evaluate_after = (_now() + timedelta(hours=float(hours))).isoformat(timespec="seconds")
        entry["re_evaluate_after"] = re_evaluate_after
        if hold_reason:
            entry["hold_reason"] = hold_reason
    if linked_intent_id:
        entry["linked_intent_id"] = linked_intent_id

    with _LOG_LOCK:
        entries = _load()
        entries.append(entry)
        _save(_prune(entries))
    return entry


def record_decisions(signals: List[Dict[str, Any]], session_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """Record a batch of decided signals (from a WakeDecision)."""
    recorded = []
    for sig in signals or []:
        sid = sig.get("signal_id") or sig.get("source")
        action = (sig.get("action") or "").strip().lower()
        if not sid or not action:
            continue
        recorded.append(record_decision(
            signal_id=sid,
            action=action,
            signal_type=sig.get("type") or sig.get("signal_type"),
            source_summary=sig.get("content_summary") or sig.get("source_summary", ""),
            session_id=session_id,
            hold_reason=sig.get("hold_reason", ""),
        ))
    return recorded


def link_intent(signal_id: str, intent_id: str) -> bool:
    """Attach ``linked_intent_id`` to the most recent hold entry for a signal."""
    with _LOG_LOCK:
        entries = _load()
        for e in reversed(entries):
            if e.get("signal_id") == signal_id and e.get("action") == "hold":
                e["linked_intent_id"] = intent_id
                _save(entries)
                return True
    return False
