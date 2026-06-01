"""Values layer — Layer 3 autonomy.

Sou's reflectively endorsed values: things she stands for even when it's costly.
Written by Sou during reflection sessions (not configured by the developer).
The developer provides an initial seed via SOUL.md; Sou grows her values through experience.

State: ~/.hermes/autonomy/values.yaml
"""

import logging
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from autonomy.config import get_autonomy_state_dir

logger = logging.getLogger(__name__)

_VALUES_LOCK = threading.RLock()
_VALUES_STALE_DAYS = 30


def _values_path() -> Path:
    return get_autonomy_state_dir() / "values.yaml"


def _now() -> datetime:
    from hermes_time import now as _hermes_now
    return _hermes_now()


def _now_iso() -> str:
    from hermes_time import now as _hermes_now
    return _hermes_now().isoformat(timespec="seconds")


def _load_values() -> List[Dict[str, Any]]:
    path = _values_path()
    if not path.exists():
        return []
    try:
        import yaml
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            vals = data.get("values", [])
            if isinstance(vals, list):
                return [v for v in vals if isinstance(v, dict)]
    except Exception as e:
        logger.debug("values: load failed: %s", e)
    return []


def _save_values(values: List[Dict[str, Any]]) -> None:
    import yaml
    path = _values_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump({"values": values}, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def list_values() -> List[Dict[str, Any]]:
    """Return all current values."""
    with _VALUES_LOCK:
        return _load_values()


def get_top_values(n: int = 3) -> List[Dict[str, Any]]:
    """Return the most recently reflected-on values (up to n)."""
    vals = list_values()
    # Sort by last_reflected descending; values without a date go last
    vals.sort(key=lambda v: str(v.get("last_reflected") or ""), reverse=True)
    return vals[:n]


def get_top_values_text(n: int = 3) -> str:
    """Return a compact text block of the top values for prompt injection."""
    vals = get_top_values(n)
    if not vals:
        return ""
    lines = []
    for v in vals:
        stmt = v.get("statement", "")
        domain = v.get("domain", "")
        lines.append(f"- {stmt}" + (f" ({domain})" if domain else ""))
    return "\n".join(lines)


def add_value(statement: str, domain: str = "", source: str = "") -> Dict[str, Any]:
    """Add a new value. Returns the created value dict."""
    import uuid
    statement = (statement or "").strip()
    if not statement:
        raise ValueError("statement is required")
    with _VALUES_LOCK:
        vals = _load_values()
        now = _now_iso()
        val: Dict[str, Any] = {
            "id": f"val_{str(uuid.uuid4())[:8]}",
            "statement": statement,
            "domain": (domain or "").strip(),
            "last_reflected": now[:10],  # date only
        }
        if source:
            val["source"] = source.strip()
        vals.append(val)
        _save_values(vals)
        return val


def reflect_value(id_or_statement: str, revised_statement: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Mark a value as reflected-on today; optionally revise its statement."""
    key = (id_or_statement or "").strip().lower()
    if not key:
        return None
    with _VALUES_LOCK:
        vals = _load_values()
        for val in vals:
            if val.get("id") == id_or_statement or key in str(val.get("statement", "")).lower():
                val["last_reflected"] = _now_iso()[:10]
                if revised_statement:
                    val["statement"] = revised_statement.strip()
                _save_values(vals)
                return val
    return None


def collect_values_signals(config: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """Return signals for values that haven't been reflected on in >30 days. Never raises."""
    try:
        stale_days = float((config or {}).get("values_stale_days", _VALUES_STALE_DAYS))
        vals = list_values()
        if not vals:
            return []
        now_date = _now().date()
        stale = []
        for val in vals:
            last_ref = val.get("last_reflected")
            if not last_ref:
                stale.append(val)
                continue
            try:
                ref_date = datetime.fromisoformat(str(last_ref)[:10]).date()
                if (now_date - ref_date).days > stale_days:
                    stale.append(val)
            except (ValueError, TypeError):
                stale.append(val)

        if not stale:
            return []
        return [{
            "signal_id": "values:stale",
            "type": "values_stale",
            "stale_count": len(stale),
            "stale_values": [v.get("statement", "")[:80] for v in stale[:3]],
            "content_summary": f"{len(stale)} value(s) haven't been reflected on in >{int(stale_days)} days",
        }]
    except Exception:
        logger.exception("values: collect_values_signals failed")
        return []
