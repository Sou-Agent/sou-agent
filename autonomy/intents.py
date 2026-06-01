"""Intent store — file-backed CRUD over ``{journal}/intents/*.yaml``.

An intent is something the agent wants to follow up on later. Intents are the
highest-priority input to the autonomy loop: the collector checks them first,
evaluates any structured ``condition`` against live state, and surfaces
triggered ones to the aux model.

Each intent records its ``origin`` — why it was created:
  - ``free``         spontaneous note during any session (no linked signal)
  - ``hold``         created while holding a specific signal, to come back to it
  - ``self_reflect`` realized during reflection that a follow-up is wanted
  - ``cron``         created during a scheduled cron session
  - ``myself``       something Sou wants to do for herself — not reactive, purely self-directed
When origin is ``hold`` or ``self_reflect``, ``source_signal_id`` links back to
the originating entry in the response log, so the full history is reconstructable.

This module is the single owner of intent file I/O. ``tools/intent_tool.py`` is
a thin tool wrapper over it; ``collector.py`` reads through it too.
"""

import logging
import threading
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from autonomy.config import get_journal_path

logger = logging.getLogger(__name__)

_INTENTS_LOCK = threading.RLock()

VALID_STATUSES = ("pending", "triggered", "completed", "dismissed", "waiting")
VALID_PRIORITIES = ("low", "normal", "high")
VALID_ORIGINS = (
    "free", "hold", "self_reflect", "cron", "myself",
    "drive",        # arose from a motivational drive session
    "world_model",  # arose from a stale world model node
    "research",     # arose from a research session
    "prospection",  # arose from a prospection (future imagination) session
    "recur",        # automatically re-created from a recurring intent
)
VALID_AFFECTS = ("seeking", "care", "play", "grief", "anxious", "obligated", "excited", "curious")
VALID_ENERGY_COSTS = ("low", "medium", "high")


def _now_iso() -> str:
    from hermes_time import now as _hermes_now
    return _hermes_now().isoformat(timespec="seconds")


def _intents_dir() -> Path:
    return get_journal_path() / "intents"


def _read_yaml(path: Path) -> Optional[Dict[str, Any]]:
    try:
        import yaml
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data
    except Exception as e:
        logger.debug("intents: failed to read %s: %s", path, e)
    return None


def _write_yaml(path: Path, data: Dict[str, Any]) -> None:
    import yaml
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")


def load_all() -> List[Tuple[Path, Dict[str, Any]]]:
    """Return ``[(path, intent_dict), …]`` for every intent file (any status)."""
    d = _intents_dir()
    out: List[Tuple[Path, Dict[str, Any]]] = []
    if not d.is_dir():
        return out
    with _INTENTS_LOCK:
        for p in sorted(d.glob("*.yaml")):
            data = _read_yaml(p)
            if data is not None:
                out.append((p, data))
    return out


def list_intents(status: Optional[str] = None, origin: Optional[str] = None) -> List[Dict[str, Any]]:
    """Return intent dicts, optionally filtered by status and/or origin."""
    result = []
    for _path, intent in load_all():
        if status and intent.get("status") != status:
            continue
        if origin and intent.get("origin") != origin:
            continue
        result.append(intent)
    return result


def add_intent(
    description: str,
    condition: Optional[str] = None,
    priority: str = "normal",
    origin: str = "free",
    source_signal_id: Optional[str] = None,
    affect: Optional[str] = None,
    recur: Optional[str] = None,
    energy_cost: Optional[str] = None,
    depends_on: Optional[List[str]] = None,
    tags: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Create a new pending intent and return it."""
    description = (description or "").strip()
    if not description:
        raise ValueError("description is required")
    priority = priority if priority in VALID_PRIORITIES else "normal"
    origin = origin if origin in VALID_ORIGINS else "free"

    now = _now_iso()
    intent: Dict[str, Any] = {
        "id": str(uuid.uuid4()),
        "created": now,
        "updated": now,
        "description": description,
        "condition": (condition or "").strip() or None,
        "priority": priority,
        "status": "pending",
        "origin": origin,
        "surface_count": 0,
    }
    if source_signal_id:
        intent["source_signal_id"] = source_signal_id
    if affect and affect in VALID_AFFECTS:
        intent["affect"] = affect
    if recur:
        intent["recur"] = recur.strip()
    if energy_cost and energy_cost in VALID_ENERGY_COSTS:
        intent["energy_cost"] = energy_cost
    if depends_on:
        intent["depends_on"] = [str(d) for d in depends_on]
    if tags:
        intent["tags"] = [str(t) for t in tags]

    with _INTENTS_LOCK:
        _write_yaml(_intents_dir() / f"{intent['id']}.yaml", intent)
    return intent


def find_intent(id_or_description: str) -> Optional[Tuple[Path, Dict[str, Any]]]:
    """Resolve an intent by exact id, id prefix, then fuzzy description substring."""
    key = (id_or_description or "").strip()
    if not key:
        return None
    entries = load_all()
    # Exact id.
    for path, intent in entries:
        if intent.get("id") == key:
            return path, intent
    # Id prefix.
    for path, intent in entries:
        if str(intent.get("id", "")).startswith(key):
            return path, intent
    # Description substring (case-insensitive).
    low = key.lower()
    for path, intent in entries:
        if low in str(intent.get("description", "")).lower():
            return path, intent
    return None


def _update_fields(id_or_description: str, updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    with _INTENTS_LOCK:
        found = find_intent(id_or_description)
        if found is None:
            return None
        path, intent = found
        intent.update(updates)
        intent["updated"] = _now_iso()
        _write_yaml(path, intent)
        return intent


def set_status(id_or_description: str, status: str, resolution: Optional[str] = None) -> Optional[Dict[str, Any]]:
    if status not in VALID_STATUSES:
        raise ValueError(f"invalid status '{status}'")
    updates: Dict[str, Any] = {"status": status}
    if resolution:
        updates["resolution"] = resolution.strip()
    return _update_fields(id_or_description, updates)


def complete_intent(id_or_description: str, resolution: Optional[str] = None) -> Optional[Dict[str, Any]]:
    return set_status(id_or_description, "completed", resolution=resolution)


def dismiss_intent(id_or_description: str) -> Optional[Dict[str, Any]]:
    return set_status(id_or_description, "dismissed")


def update_intent(id_or_description: str, field: str, value: Any) -> Optional[Dict[str, Any]]:
    """Update a single mutable field (description, condition, priority, affect, energy_cost, tags)."""
    field = (field or "").strip().lower()
    allowed = {"description", "condition", "priority", "affect", "energy_cost", "tags", "narrative_aligned"}
    if field not in allowed:
        raise ValueError(f"field must be one of {sorted(allowed)}")
    if field == "priority" and value not in VALID_PRIORITIES:
        raise ValueError(f"priority must be one of {VALID_PRIORITIES}")
    if field == "condition":
        value = (str(value).strip() or None) if value is not None else None
    if field == "affect" and value not in VALID_AFFECTS:
        raise ValueError(f"affect must be one of {VALID_AFFECTS}")
    if field == "energy_cost" and value not in VALID_ENERGY_COSTS:
        raise ValueError(f"energy_cost must be one of {VALID_ENERGY_COSTS}")
    return _update_fields(id_or_description, {field: value})


def snooze_intent(id_or_description: str, duration: str) -> Optional[Dict[str, Any]]:
    """Snooze an intent for a duration ('3d', '24h', '2w'). Soft-dismiss without loss."""
    from autonomy.collector import _parse_duration
    from hermes_time import now as _hermes_now
    dur = _parse_duration(duration)
    if dur is None:
        raise ValueError(f"unrecognized duration '{duration}' — use e.g. '3d', '24h', '1w'")
    snoozed_until = (_hermes_now() + dur).isoformat(timespec="seconds")
    return _update_fields(id_or_description, {"snoozed_until": snoozed_until})


def add_progress(id_or_description: str, note: str) -> Optional[Dict[str, Any]]:
    """Append a timestamped progress note to an intent without completing it."""
    note = (note or "").strip()
    if not note:
        raise ValueError("note is required")
    with _INTENTS_LOCK:
        found = find_intent(id_or_description)
        if found is None:
            return None
        path, intent = found
        progress = intent.get("progress") or []
        progress.append({"at": _now_iso(), "note": note})
        intent["progress"] = progress
        intent["updated"] = _now_iso()
        _write_yaml(path, intent)
        return intent


def set_waiting(id_or_description: str, unblock_condition: str) -> Optional[Dict[str, Any]]:
    """Set an intent to waiting status with an unblocking condition."""
    unblock_condition = (unblock_condition or "").strip()
    if not unblock_condition:
        raise ValueError("unblock_condition is required")
    return _update_fields(id_or_description, {
        "status": "waiting",
        "condition": unblock_condition,
    })
