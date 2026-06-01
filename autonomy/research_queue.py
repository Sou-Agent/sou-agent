"""Research queue — deferred questions Sou wants to research.

When curiosity sessions or reflections surface a question worth deeper investigation,
it can be queued here. The collector fires focused inward research sessions.

State: ~/.hermes/autonomy/research_queue.json
"""

import json
import logging
import threading
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from autonomy.config import get_autonomy_state_dir

logger = logging.getLogger(__name__)

_RQ_LOCK = threading.RLock()
VALID_PRIORITIES = ("low", "normal", "high")
_DEFAULT_MIN_WAIT_HOURS = 1


def _rq_path() -> Path:
    return get_autonomy_state_dir() / "research_queue.json"


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


def _load_queue() -> Dict[str, Any]:
    path = _rq_path()
    try:
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
    except (json.JSONDecodeError, OSError):
        pass
    return {"items": {}}


def _save_queue(data: Dict[str, Any]) -> None:
    path = _rq_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(path)


def add_question(
    question: str,
    priority: str = "normal",
    tags: Optional[List[str]] = None,
    source: str = "",
    min_wait_hours: float = _DEFAULT_MIN_WAIT_HOURS,
) -> Dict[str, Any]:
    """Queue a research question. Returns the created item dict."""
    question = (question or "").strip()
    if not question:
        raise ValueError("question is required")
    priority = priority if priority in VALID_PRIORITIES else "normal"
    with _RQ_LOCK:
        data = _load_queue()
        items = data.setdefault("items", {})
        item_id = str(uuid.uuid4())[:12]
        item: Dict[str, Any] = {
            "id": item_id,
            "question": question,
            "priority": priority,
            "tags": tags or [],
            "source": source.strip(),
            "min_wait_hours": float(min_wait_hours),
            "status": "pending",
            "created": _now_iso(),
            "queued_since": _now_iso(),
        }
        items[item_id] = item
        _save_queue(data)
        return item


def complete_question(item_id: str, summary: str = "") -> Optional[Dict[str, Any]]:
    """Mark a research question as answered."""
    with _RQ_LOCK:
        data = _load_queue()
        item = data.get("items", {}).get(item_id)
        if item is None:
            return None
        item["status"] = "completed"
        item["completed"] = _now_iso()
        if summary:
            item["summary"] = summary.strip()
        data["items"][item_id] = item
        _save_queue(data)
        return item


def list_queue(status: str = "pending") -> List[Dict[str, Any]]:
    """Return queued research items."""
    with _RQ_LOCK:
        data = _load_queue()
        items = list(data.get("items", {}).values())
        if status:
            items = [i for i in items if i.get("status") == status]
        items.sort(key=lambda i: (
            {"high": 0, "normal": 1, "low": 2}.get(i.get("priority", "normal"), 1),
            str(i.get("queued_since") or ""),
        ))
        return items


def collect_research_signals(config: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """Return signals for queued research questions that are ready. Never raises."""
    try:
        items = list_queue(status="pending")
        now = _now()
        signals = []
        for item in items[:3]:  # Surface up to 3 at once
            queued = _parse_iso(item.get("queued_since"))
            min_wait_h = float(item.get("min_wait_hours", _DEFAULT_MIN_WAIT_HOURS))
            if queued is not None and (now - queued) < timedelta(hours=min_wait_h):
                continue
            signals.append({
                "signal_id": f"research:{item['id']}",
                "type": "research_queued",
                "item_id": item["id"],
                "question": item["question"],
                "priority": item.get("priority", "normal"),
                "tags": item.get("tags", []),
                "content_summary": f"queued research: {item['question'][:100]}",
            })
        return signals
    except Exception:
        logger.exception("research_queue: collect failed")
        return []
