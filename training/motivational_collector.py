"""Motivational state-transition data collector.

Captures (observation, action, next_observation) tuples around autonomy sessions
for future world model training. Zero overhead when disabled in config.

Config gate: training.collect_motivational_data: false (default)
Storage: ~/.hermes/training_data/motivational/YYYY-MM-DD.jsonl
"""

import json
import logging
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_COLLECTOR_LOCK = threading.Lock()
_pending: Dict[str, Dict[str, Any]] = {}  # record_id → partial record


def is_motivational_collection_enabled(config: Optional[Dict[str, Any]] = None) -> bool:
    """Return True iff motivational data collection is configured on."""
    if config is None:
        try:
            from autonomy.config import _load_config
            config = _load_config()
        except Exception:
            return False
    training = config.get("training") or {}
    return bool(training.get("collect_motivational_data", False))


def _motivational_dir() -> Path:
    from hermes_constants import get_hermes_home
    return get_hermes_home() / "training_data" / "motivational"


def _today_jsonl() -> Path:
    date_str = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
    return _motivational_dir() / f"{date_str}.jsonl"


def _build_observation(motivational_state: Optional[Dict[str, Any]] = None,
                        context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Assemble the full observation vector including identity dimensions."""
    obs: Dict[str, Any] = {}

    # --- Drive state ---
    if motivational_state:
        obs["drives"] = motivational_state.get("drives", {})
        obs["sdt_needs"] = motivational_state.get("sdt_needs", {})
        obs["temporal"] = motivational_state.get("temporal", {})
        obs["rumination_level"] = motivational_state.get("rumination_level", 0)
    else:
        try:
            from autonomy.motivational_state import get_current_state
            state = get_current_state()
            obs["drives"] = state.get("drives", {})
            obs["sdt_needs"] = state.get("sdt_needs", {})
            obs["temporal"] = state.get("temporal", {})
            obs["rumination_level"] = state.get("rumination_level", 0)
        except Exception:
            obs["drives"] = {}
            obs["sdt_needs"] = {}
            obs["temporal"] = {}
            obs["rumination_level"] = 0

    # --- Session context ---
    obs["context"] = context or {}

    # --- Identity dimensions ---
    identity: Dict[str, Any] = {}
    try:
        from autonomy.values import list_values
        from datetime import date
        vals = list_values()
        today = date.today()
        alignment_scores = {}
        for v in vals:
            last_ref = v.get("last_reflected")
            if last_ref:
                try:
                    ref_date = datetime.fromisoformat(str(last_ref)[:10]).date()
                    days_since = (today - ref_date).days
                    alignment_scores[v.get("id", "")] = max(0.0, 1.0 - days_since / 30.0)
                except (ValueError, TypeError):
                    alignment_scores[v.get("id", "")] = 0.5
        identity["value_alignment_scores"] = alignment_scores
    except Exception:
        identity["value_alignment_scores"] = {}

    try:
        from autonomy.metacognition import get_domain_performance
        perf = get_domain_performance()
        identity["domain_confidence"] = {
            d: round(float(p.get("success_rate", 0.5)), 2)
            for d, p in perf.items()
        }
    except Exception:
        identity["domain_confidence"] = {}

    obs["identity"] = identity

    # --- Social state ---
    social: Dict[str, Any] = {}
    try:
        from autonomy.social_cognition import list_people
        people = list_people()
        social["relationship_quality_vector"] = {
            p.get("person_key", ""): round(float(p.get("relationship_quality", 0.5)), 2)
            for p in people if p.get("person_key")
        }
        social["pending_outreach_count"] = sum(
            1 for p in people
            if p.get("pending_initiations")
        )
    except Exception:
        pass
    obs["social"] = social

    # --- Intent affect distribution ---
    try:
        from autonomy import intents as intent_store
        pending = intent_store.list_intents(status="pending")
        affect_dist: Dict[str, int] = {}
        for it in pending:
            affect = it.get("affect")
            if affect:
                affect_dist[affect] = affect_dist.get(affect, 0) + 1
        obs["intent_affect_distribution"] = affect_dist
    except Exception:
        obs["intent_affect_distribution"] = {}

    now_iso = datetime.now(tz=timezone.utc).isoformat(timespec="seconds")
    obs["captured_at"] = now_iso
    return obs


def capture_pre_session(
    motivational_state: Optional[Dict[str, Any]] = None,
    context: Optional[Dict[str, Any]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """Snapshot state before a session. Returns record_id for post-session pairing."""
    if not is_motivational_collection_enabled(config):
        return None
    try:
        record_id = str(uuid.uuid4())
        observation = _build_observation(motivational_state, context)
        with _COLLECTOR_LOCK:
            _pending[record_id] = {
                "record_id": record_id,
                "timestamp": datetime.now(tz=timezone.utc).isoformat(timespec="seconds"),
                "observation": observation,
            }
        return record_id
    except Exception:
        logger.debug("motivational_collector: capture_pre_session failed", exc_info=True)
        return None


def capture_post_session(
    record_id: str,
    session_result: Optional[Dict[str, Any]] = None,
    next_motivational_state: Optional[Dict[str, Any]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> None:
    """Complete the record after a session ends. Writes to daily JSONL."""
    if not is_motivational_collection_enabled(config):
        return
    try:
        with _COLLECTOR_LOCK:
            partial = _pending.pop(record_id, None)
        if partial is None:
            return

        next_obs = _build_observation(next_motivational_state)
        result = session_result or {}

        # Compute session quality score heuristic
        journal_writes = int(result.get("journal_writes", 0))
        intents_created = int(result.get("intents_created", 0))
        intents_completed = int(result.get("intents_completed", 0))
        duration_min = float(result.get("duration_minutes", 0))
        outreach_responded = bool(result.get("outreach_responded", False))
        drives_targeted = result.get("drives_targeted", [])

        quality = (
            0.3 * min(1.0, journal_writes / max(1, result.get("expected_journal_writes", 1)))
            + 0.2 * min(1.0, (intents_created + intents_completed) / 2.0)
            + 0.2 * min(1.0, duration_min / 20.0)
            + 0.3 * (1 if outreach_responded else 0)
        )

        # Drive satisfaction deltas
        prev_drives = (partial.get("observation") or {}).get("drives", {})
        next_drives = next_obs.get("drives", {})
        drive_deltas: Dict[str, float] = {}
        for d in drives_targeted:
            prev_eff = _drive_effective(prev_drives.get(d, {}))
            next_eff = _drive_effective(next_drives.get(d, {}))
            drive_deltas[d] = round(next_eff - prev_eff, 3)

        record = {
            **partial,
            "action": {
                "session_type": result.get("session_type", "inward"),
                "drives_targeted": drives_targeted,
                "affect_targeted": result.get("affect_targeted"),
                "tools_used": result.get("tools_used", []),
                "duration_minutes": duration_min,
                "intents_created": intents_created,
                "intents_completed": intents_completed,
                "journal_entries_written": journal_writes,
                "outreach_sent": bool(result.get("outreach_sent", False)),
            },
            "next_observation": next_obs,
            "outcome": {
                "session_quality_score": round(quality, 3),
                "drive_satisfaction_deltas": drive_deltas,
            },
        }

        _append_record(record)
        logger.debug("motivational_collector: record %s written", record_id)
    except Exception:
        logger.debug("motivational_collector: capture_post_session failed", exc_info=True)


def _drive_effective(drv: Dict[str, Any]) -> float:
    raw = float(drv.get("raw_level", 0.0))
    sat = float(drv.get("satiation", 0.0))
    opp = float(drv.get("opponent_charge", 0.0))
    return raw * (1.0 - sat) * (1.0 - 0.5 * opp)


def _append_record(record: Dict[str, Any]) -> None:
    path = _today_jsonl()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def flush_pending(session_id: Optional[str] = None) -> None:
    """Discard any pending pre-session records that were never completed.

    Called by create_session_in before overwriting a channel's session pointer,
    to ensure orphaned records don't accumulate.
    """
    with _COLLECTOR_LOCK:
        if session_id:
            to_remove = [rid for rid, rec in _pending.items()
                         if str(rec.get("session_id", "")) == session_id]
            for rid in to_remove:
                _pending.pop(rid, None)


def get_trajectory(date: Optional[str] = None) -> List[Dict[str, Any]]:
    """Load all records for a given date (YYYY-MM-DD), ordered by timestamp."""
    if date is None:
        date = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
    path = _motivational_dir() / f"{date}.jsonl"
    records = []
    try:
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
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


