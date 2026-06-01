"""Metacognitive loop — Layer 5 autonomy.

Tracks per-domain session success rates. Fires a signal when a domain is
consistently failing, prompting Sou to reflect on strategy before acting again.

State: ~/.hermes/autonomy/metacognition_log.json
"""

import json
import logging
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from autonomy.config import get_autonomy_state_dir

logger = logging.getLogger(__name__)

_META_LOCK = threading.RLock()
_LOW_SUCCESS_THRESHOLD = 0.4
_STREAK_THRESHOLD = 5


def _meta_path() -> Path:
    return get_autonomy_state_dir() / "metacognition_log.json"


def _load_meta() -> Dict[str, Any]:
    path = _meta_path()
    try:
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
    except (json.JSONDecodeError, OSError):
        pass
    return {"domain_performance": {}, "detected_patterns": []}


def _save_meta(data: Dict[str, Any]) -> None:
    path = _meta_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(path)


def record_session_outcome(domain: str, success: bool) -> None:
    """Record whether a session in a given domain was successful."""
    with _META_LOCK:
        try:
            data = _load_meta()
            perf = data.setdefault("domain_performance", {})
            if domain not in perf:
                perf[domain] = {"success_rate": 0.5, "last_10": [], "total": 0}
            entry = perf[domain]
            last_10 = entry.get("last_10", [])
            last_10.append(1 if success else 0)
            last_10 = last_10[-10:]  # keep only last 10
            entry["last_10"] = last_10
            entry["total"] = entry.get("total", 0) + 1
            entry["success_rate"] = sum(last_10) / len(last_10) if last_10 else 0.5
            perf[domain] = entry

            # Update detected patterns
            patterns = data.get("detected_patterns", [])
            patterns = [p for p in patterns if p.get("domain") != domain]
            if len(last_10) >= 5:
                fail_streak = 0
                for v in reversed(last_10):
                    if v == 0:
                        fail_streak += 1
                    else:
                        break
                if fail_streak >= _STREAK_THRESHOLD or entry["success_rate"] < _LOW_SUCCESS_THRESHOLD:
                    patterns.append({
                        "type": "low_success_streak",
                        "domain": domain,
                        "streak": fail_streak,
                        "success_rate": round(entry["success_rate"], 2),
                        "flag": "strategy_review_needed",
                    })
            data["detected_patterns"] = patterns
            _save_meta(data)
        except Exception:
            logger.exception("metacognition: record_session_outcome failed for %s", domain)


def collect_metacognitive_signals(config: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """Return signals when a domain has consistently low success. Never raises."""
    try:
        data = _load_meta()
        signals = []
        for domain, perf in data.get("domain_performance", {}).items():
            if not isinstance(perf, dict):
                continue
            last_10 = perf.get("last_10", [])
            success_rate = float(perf.get("success_rate", 1.0))
            if len(last_10) < 5:
                continue

            fail_streak = 0
            for v in reversed(last_10):
                if v == 0:
                    fail_streak += 1
                else:
                    break

            if success_rate < _LOW_SUCCESS_THRESHOLD or fail_streak >= _STREAK_THRESHOLD:
                signals.append({
                    "signal_id": f"metacognitive:{domain}",
                    "type": "metacognitive",
                    "domain": domain,
                    "success_rate": round(success_rate, 2),
                    "streak": fail_streak,
                    "content_summary": (
                        f"{domain} success rate {success_rate:.0%} over last {len(last_10)} sessions"
                        + (f" — {fail_streak}-session fail streak" if fail_streak >= _STREAK_THRESHOLD else "")
                    ),
                })
        return signals
    except Exception:
        logger.exception("metacognition: collect failed")
        return []


def get_domain_performance() -> Dict[str, Any]:
    """Return all tracked domain performance data."""
    with _META_LOCK:
        return _load_meta().get("domain_performance", {})
