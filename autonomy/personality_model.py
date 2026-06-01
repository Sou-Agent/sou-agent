"""Learned personality model — Layer 2 autonomy.

Fits drive parameters (inhibition weights, satiation halflives, circadian offsets)
from motivational training data. Bootstrapped with neutral priors; confidence grows
as records accumulate. Run daily from the gateway ticker.

State: ~/.hermes/autonomy/personality.json
"""

import json
import logging
import math
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from autonomy.config import get_autonomy_state_dir

logger = logging.getLogger(__name__)

_PERSONALITY_LOCK = threading.RLock()

DRIVE_NAMES = ("curiosity", "connection", "expression", "reflection", "play", "growth")

_NEUTRAL_PRIORS: Dict[str, Any] = {
    "inhibition_matrix": {
        "curiosity→connection": 0.3, "curiosity→reflection": 0.2,
        "curiosity→expression": 0.25, "curiosity→play": 0.2, "curiosity→growth": 0.15,
        "connection→curiosity": 0.3, "connection→growth": 0.2, "connection→expression": 0.1,
        "connection→reflection": 0.25, "connection→play": 0.1,
        "expression→curiosity": 0.2, "expression→connection": 0.1,
        "expression→reflection": 0.15, "expression→play": 0.05, "expression→growth": 0.15,
        "reflection→curiosity": 0.25, "reflection→connection": 0.2,
        "reflection→expression": 0.15, "reflection→play": 0.3, "reflection→growth": 0.1,
        "play→curiosity": 0.1, "play→connection": 0.05, "play→reflection": 0.35,
        "play→expression": 0.05, "play→growth": 0.4,
        "growth→curiosity": 0.1, "growth→connection": 0.15, "growth→expression": 0.1,
        "growth→reflection": 0.1, "growth→play": 0.3,
    },
    "satiation_halflives_min": {
        "curiosity": 420, "connection": 380, "expression": 500,
        "reflection": 600, "play": 300, "growth": 480,
    },
    "circadian_phase_offsets_rad": {
        "curiosity": -0.52, "connection": 1.05, "expression": 0.0,
        "reflection": 2.61, "play": -1.0, "growth": -0.3,
    },
    "confidence": 0.0,
    "sessions_analyzed": 0,
    "last_fit": None,
}


def _personality_path() -> Path:
    return get_autonomy_state_dir() / "personality.json"


def load_personality() -> Dict[str, Any]:
    """Return current personality params (neutral priors until trained)."""
    path = _personality_path()
    try:
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
    except (json.JSONDecodeError, OSError):
        pass
    return dict(_NEUTRAL_PRIORS)


def _save_personality(data: Dict[str, Any]) -> None:
    path = _personality_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(path)


def fit_personality(motivational_data_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Fit personality parameters from motivational training records.

    Called daily from the gateway ticker. Safe to call at any time — returns
    current params and logs if there's insufficient data.
    """
    from hermes_constants import get_hermes_home
    if motivational_data_dir is None:
        motivational_data_dir = get_hermes_home() / "training_data" / "motivational"

    with _PERSONALITY_LOCK:
        try:
            return _fit_impl(motivational_data_dir)
        except Exception:
            logger.exception("personality_model: fit failed")
            return load_personality()


def _fit_impl(data_dir: Path) -> Dict[str, Any]:
    records = _load_all_records(data_dir)
    current = load_personality()
    n_analyzed = int(current.get("sessions_analyzed", 0))

    if len(records) < 10:
        logger.info("personality_model: %d records — below minimum, keeping current params", len(records))
        return current

    # EMA blend weight grows with data volume; max 30% new influence per fit
    alpha = min(0.3, len(records) / 500.0)
    prior_w = 1.0 - alpha
    priors = _NEUTRAL_PRIORS

    # --- Inhibition weights ---
    # succession_counts[A][B] = how often a drive-A session is followed by drive-B
    succession: Dict[str, Dict[str, int]] = {n: {} for n in DRIVE_NAMES}
    totals: Dict[str, int] = {n: 0 for n in DRIVE_NAMES}
    for i in range(len(records) - 1):
        prev_drives = (records[i].get("action") or {}).get("drives_targeted") or []
        next_drives = (records[i + 1].get("action") or {}).get("drives_targeted") or []
        for pd in prev_drives:
            if pd not in DRIVE_NAMES:
                continue
            totals[pd] = totals.get(pd, 0) + 1
            for nd in next_drives:
                if nd in DRIVE_NAMES:
                    succession[pd].setdefault(nd, 0)
                    succession[pd][nd] += 1

    inhibition = {}
    for a in DRIVE_NAMES:
        for b in DRIVE_NAMES:
            if a == b:
                continue
            key = f"{a}→{b}"
            total = totals.get(a, 0)
            prior_val = priors["inhibition_matrix"].get(key, 0.3)
            if total >= 5:
                p_b_given_a = succession[a].get(b, 0) / total
                learned = max(0.0, min(1.0, 1.0 - p_b_given_a))
                inhibition[key] = prior_w * prior_val + alpha * learned
            else:
                inhibition[key] = prior_val

    # --- Satiation halflives: median gap between same-drive sessions ≈ 2× halflife ---
    gaps: Dict[str, List[float]] = {n: [] for n in DRIVE_NAMES}
    last_ts: Dict[str, Optional[str]] = {n: None for n in DRIVE_NAMES}
    for rec in records:
        ts = rec.get("timestamp")
        if not ts:
            continue
        try:
            rec_dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
            if rec_dt.tzinfo is None:
                rec_dt = rec_dt.replace(tzinfo=timezone.utc)
        except (ValueError, TypeError):
            continue
        for d in ((rec.get("action") or {}).get("drives_targeted") or []):
            if d not in DRIVE_NAMES:
                continue
            if last_ts[d]:
                try:
                    prev_dt = datetime.fromisoformat(str(last_ts[d]).replace("Z", "+00:00"))
                    if prev_dt.tzinfo is None:
                        prev_dt = prev_dt.replace(tzinfo=timezone.utc)
                    gap_min = (rec_dt - prev_dt).total_seconds() / 60.0
                    if 30 < gap_min < 10000:
                        gaps[d].append(gap_min)
                except (ValueError, TypeError):
                    pass
            last_ts[d] = ts

    halflives = {}
    for d in DRIVE_NAMES:
        prior_hl = priors["satiation_halflives_min"].get(d, 480)
        if len(gaps[d]) >= 5:
            g = sorted(gaps[d])
            median = g[len(g) // 2]
            learned = max(60.0, min(1440.0, median / 2.0))
            halflives[d] = prior_w * prior_hl + alpha * learned
        else:
            halflives[d] = prior_hl

    # --- Circadian offsets: find peak hour per drive ---
    hour_counts: Dict[str, List[float]] = {n: [0.0] * 24 for n in DRIVE_NAMES}
    for rec in records:
        ts = rec.get("timestamp")
        if not ts:
            continue
        try:
            hour = datetime.fromisoformat(str(ts).replace("Z", "+00:00")).hour
        except (ValueError, TypeError):
            continue
        for d in ((rec.get("action") or {}).get("drives_targeted") or []):
            if d in DRIVE_NAMES:
                hour_counts[d][hour] += 1

    circ_offsets = {}
    for d in DRIVE_NAMES:
        prior_off = priors["circadian_phase_offsets_rad"].get(d, 0.0)
        total_h = sum(hour_counts[d])
        if total_h >= 10:
            peak_hour = hour_counts[d].index(max(hour_counts[d]))
            learned_off = -(peak_hour / 24.0) * 2 * math.pi
            circ_offsets[d] = prior_w * prior_off + alpha * learned_off
        else:
            circ_offsets[d] = prior_off

    total_sessions = n_analyzed + len(records)
    from hermes_time import now as _hermes_now
    personality = {
        "inhibition_matrix": inhibition,
        "satiation_halflives_min": halflives,
        "circadian_phase_offsets_rad": circ_offsets,
        "confidence": round(min(1.0, total_sessions / 500.0), 3),
        "sessions_analyzed": total_sessions,
        "last_fit": _hermes_now().isoformat(timespec="seconds"),
    }
    _save_personality(personality)
    logger.info("personality_model: fit complete — %d sessions, confidence=%.2f",
                total_sessions, personality["confidence"])
    return personality


def _load_all_records(data_dir: Path) -> List[Dict[str, Any]]:
    records = []
    if not data_dir.is_dir():
        return records
    for jsonl_path in sorted(data_dir.glob("*.jsonl")):
        try:
            for line in jsonl_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    if isinstance(rec, dict):
                        records.append(rec)
                except json.JSONDecodeError:
                    pass
        except OSError:
            pass
    records.sort(key=lambda r: str(r.get("timestamp") or ""))
    return records
