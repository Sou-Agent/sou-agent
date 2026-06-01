"""World model — what Sou is actively tracking.

Sou can declare she's following a topic, person, question, or project. Each node
has an engagement interval. When a node goes stale, the collector fires a signal
prompting her to re-engage.

State: ~/.hermes/autonomy/world_model.json
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

_WM_LOCK = threading.RLock()
VALID_NODE_TYPES = ("topic", "person", "project", "question")
_DEFAULT_INTERVAL_HOURS = 168  # 1 week


def _wm_path() -> Path:
    return get_autonomy_state_dir() / "world_model.json"


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


def _load_wm() -> Dict[str, Any]:
    path = _wm_path()
    try:
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
    except (json.JSONDecodeError, OSError):
        pass
    return {"nodes": {}}


def _save_wm(data: Dict[str, Any]) -> None:
    path = _wm_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(path)


def add_node(
    label: str,
    node_type: str = "topic",
    interval_hours: float = _DEFAULT_INTERVAL_HOURS,
    notes: str = "",
) -> Dict[str, Any]:
    """Add a world model node. Returns the created node dict."""
    label = (label or "").strip()
    if not label:
        raise ValueError("label is required")
    node_type = node_type if node_type in VALID_NODE_TYPES else "topic"
    with _WM_LOCK:
        data = _load_wm()
        nodes = data.setdefault("nodes", {})
        node_id = str(uuid.uuid4())[:12]
        node: Dict[str, Any] = {
            "id": node_id,
            "label": label,
            "type": node_type,
            "interval_hours": float(interval_hours),
            "last_engaged": _now_iso(),
            "notes": notes.strip(),
            "active": True,
            "created": _now_iso(),
        }
        nodes[node_id] = node
        _save_wm(data)
        return node


def engage_node(node_id: str) -> Optional[Dict[str, Any]]:
    """Mark a node as engaged now, resetting its staleness timer."""
    with _WM_LOCK:
        data = _load_wm()
        node = data.get("nodes", {}).get(node_id)
        if node is None:
            return None
        node["last_engaged"] = _now_iso()
        data["nodes"][node_id] = node
        _save_wm(data)
        return node


def remove_node(node_id: str) -> bool:
    """Deactivate a node (soft delete)."""
    with _WM_LOCK:
        data = _load_wm()
        node = data.get("nodes", {}).get(node_id)
        if node is None:
            return False
        node["active"] = False
        data["nodes"][node_id] = node
        _save_wm(data)
        return True


def list_nodes(active_only: bool = True) -> List[Dict[str, Any]]:
    """Return all (active) world model nodes."""
    with _WM_LOCK:
        data = _load_wm()
        nodes = list(data.get("nodes", {}).values())
        if active_only:
            nodes = [n for n in nodes if n.get("active", True)]
        return nodes


def collect_world_model_signals(config: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """Return signals for stale world model nodes. Never raises."""
    try:
        nodes = list_nodes(active_only=True)
        now = _now()
        signals = []
        for node in nodes:
            interval_h = float(node.get("interval_hours", _DEFAULT_INTERVAL_HOURS))
            last_engaged = _parse_iso(node.get("last_engaged"))
            if last_engaged is None:
                hours_since = None
                is_stale = True
            else:
                hours_since = (now - last_engaged).total_seconds() / 3600.0
                is_stale = hours_since >= interval_h

            if is_stale:
                signals.append({
                    "signal_id": f"world_model:{node['id']}",
                    "type": "world_model_stale",
                    "node_id": node["id"],
                    "label": node.get("label", ""),
                    "node_type": node.get("type", "topic"),
                    "hours_since": round(hours_since, 1) if hours_since else None,
                    "notes": node.get("notes", ""),
                    "content_summary": (
                        f"tracking '{node.get('label')}' ({node.get('type', 'topic')})"
                        + (f" — last engaged {hours_since:.0f}h ago" if hours_since else " — never engaged")
                    ),
                })
        return signals
    except Exception:
        logger.exception("world_model: collect failed")
        return []
