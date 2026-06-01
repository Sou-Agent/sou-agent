"""Social cognition layer — Layer 8 autonomy.

Theory-of-mind contacts: Sou models the people she cares about as agents with their
own inner lives — not just nodes in a contact list. Outreach signals carry rich
context: known interests, challenges, pending topics, communication style.

State: ~/.hermes/autonomy/social_models.json
"""

import json
import logging
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from autonomy.config import get_autonomy_state_dir

logger = logging.getLogger(__name__)

_SOCIAL_LOCK = threading.RLock()
_DEFAULT_OUTREACH_INTERVAL_HOURS = 168  # 1 week


def _models_path() -> Path:
    return get_autonomy_state_dir() / "social_models.json"


def _now() -> datetime:
    from hermes_time import now as _hermes_now
    return _hermes_now()


def _now_iso() -> str:
    from hermes_time import now as _hermes_now
    return _hermes_now().isoformat(timespec="seconds")


def _parse_iso(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt
    except (ValueError, TypeError):
        return None


def _load_models() -> Dict[str, Any]:
    path = _models_path()
    try:
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
    except (json.JSONDecodeError, OSError):
        pass
    return {"people": {}}


def _save_models(data: Dict[str, Any]) -> None:
    path = _models_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(path)


def get_person(person_key: str) -> Optional[Dict[str, Any]]:
    """Return the social model for a person, or None."""
    with _SOCIAL_LOCK:
        data = _load_models()
        return data.get("people", {}).get(person_key.lower().strip())


def update_person(person_key: str, field: str, value: Any) -> Dict[str, Any]:
    """Update a single field in a person's model. Creates the record if absent."""
    key = person_key.lower().strip()
    with _SOCIAL_LOCK:
        data = _load_models()
        people = data.setdefault("people", {})
        if key not in people:
            people[key] = {"person_key": key, "last_sou_initiated": None, "relationship_quality": 0.5,
                           "desired_outreach_interval_hours": _DEFAULT_OUTREACH_INTERVAL_HOURS}
        people[key][field] = value
        people[key]["updated"] = _now_iso()
        _save_models(data)
        return people[key]


def add_initiative(person_key: str, topic: str) -> Dict[str, Any]:
    """Note something Sou wants to bring up with this person."""
    key = person_key.lower().strip()
    with _SOCIAL_LOCK:
        data = _load_models()
        people = data.setdefault("people", {})
        if key not in people:
            people[key] = {"person_key": key, "last_sou_initiated": None, "relationship_quality": 0.5,
                           "desired_outreach_interval_hours": _DEFAULT_OUTREACH_INTERVAL_HOURS}
        initiatives = people[key].setdefault("pending_initiations", [])
        if topic.strip() not in initiatives:
            initiatives.append(topic.strip())
        people[key]["updated"] = _now_iso()
        _save_models(data)
        return people[key]


def record_outreach(person_key: str) -> None:
    """Mark that Sou has reached out, resetting the outreach timer."""
    key = person_key.lower().strip()
    with _SOCIAL_LOCK:
        data = _load_models()
        people = data.setdefault("people", {})
        if key in people:
            people[key]["last_sou_initiated"] = _now_iso()
            _save_models(data)


def collect_social_signals(config: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """Return outreach signals for people whose contact rhythm is overdue. Never raises."""
    try:
        data = _load_models()
        people = data.get("people", {})
        now = _now()
        signals = []

        for key, person in people.items():
            if not isinstance(person, dict):
                continue
            interval_h = float(person.get("desired_outreach_interval_hours", _DEFAULT_OUTREACH_INTERVAL_HOURS))
            last_init = _parse_iso(person.get("last_sou_initiated"))
            if last_init is not None and (now - last_init) < timedelta(hours=interval_h):
                continue

            hours_since = None
            if last_init is not None:
                hours_since = (now - last_init).total_seconds() / 3600.0

            pending = person.get("pending_initiations") or []
            quality = float(person.get("relationship_quality", 0.5))
            style = person.get("communication_style", "")

            signals.append({
                "signal_id": f"social:{key}",
                "type": "social_outreach",
                "person_key": key,
                "relationship_quality": round(quality, 2),
                "hours_since_last_outreach": round(hours_since, 1) if hours_since else None,
                "pending_topics": pending[:5],
                "communication_style": style,
                "theory_of_mind_notes": person.get("theory_of_mind_notes", ""),
                "known_interests": person.get("known_interests", [])[:5],
                "known_challenges": person.get("known_challenges", [])[:3],
                "content_summary": (
                    f"reach out to {key}"
                    + (f" — {len(pending)} pending topic(s)" if pending else "")
                    + (f" (last: {hours_since:.0f}h ago)" if hours_since else " (never initiated)")
                ),
            })

        signals.sort(key=lambda s: (
            -(len(s.get("pending_topics") or [])),
            -(s.get("hours_since_last_outreach") or 9999),
        ))
        return signals
    except Exception:
        logger.exception("social_cognition: collect failed")
        return []


def list_people() -> List[Dict[str, Any]]:
    """Return all tracked people."""
    with _SOCIAL_LOCK:
        data = _load_models()
        return list(data.get("people", {}).values())
