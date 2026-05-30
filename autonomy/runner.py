"""Autonomy runner — ties collector → aux model → session spawn together.

Ticked by the gateway's cron ticker thread (see ``gateway/run.py``), the same
way ``maybe_run_curator`` piggy-backs on it. ``maybe_run_autonomy_cycle`` is
cheap to call frequently: it self-gates on ``autonomy.aux_interval_minutes`` and
returns immediately when autonomy is disabled or the interval hasn't elapsed.

One cycle:
    collect_state()  → (no signals? stop, ~free)
        → decide_wake()  → (no wake? stop, one cheap call)
            → spawn_autonomy_session()  → full session
    → commit_cycle()  (advance watermarks + timestamps)

Concurrency: a module-level lock ensures only one cycle runs at a time, so a
long full session can't overlap the next tick.
"""

import logging
import threading
from datetime import datetime, timezone
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_CYCLE_LOCK = threading.Lock()


def _now():
    from hermes_time import now as _hermes_now
    return _hermes_now()


def _parse_iso(value):
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (ValueError, TypeError):
        return None


def _interval_elapsed(config: Dict[str, Any]) -> bool:
    """True if at least aux_interval_minutes have passed since the last cycle."""
    from autonomy.collector import _load_cycle_state

    interval_min = float(config.get("aux_interval_minutes", 7) or 7)
    last = _parse_iso(_load_cycle_state().get("last_cycle_at"))
    if last is None:
        return True
    return (_now() - last).total_seconds() >= interval_min * 60


def run_autonomy_cycle(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Run one full cycle unconditionally (ignores the interval gate).

    Returns a small status dict for logging/testing. Never raises.
    """
    from autonomy.config import get_autonomy_config
    from autonomy import collector, aux_model, session_spawn

    if config is None:
        config = get_autonomy_config()

    status: Dict[str, Any] = {"collected": False, "had_signals": False, "woke": False, "fired": False}
    try:
        snapshot = collector.collect_state(config)
        status["collected"] = True
        status["had_signals"] = snapshot.get("has_signals", False)

        if not snapshot.get("has_signals"):
            collector.commit_cycle(snapshot, session_fired=False)
            return status

        decision = aux_model.decide_wake(snapshot)
        status["woke"] = decision.get("wake", False)

        fired = False
        if decision.get("wake"):
            result = session_spawn.spawn_autonomy_session(decision, config, snapshot=snapshot)
            fired = result.get("fired", False)
            status["fired"] = fired
            status["session_id"] = result.get("session_id", "")

        collector.commit_cycle(snapshot, session_fired=fired)
    except Exception:
        logger.exception("autonomy: cycle failed")
    return status


def maybe_run_autonomy_cycle() -> Optional[Dict[str, Any]]:
    """Run a cycle iff autonomy is enabled and the interval has elapsed.

    Safe to call on every cron tick. Returns the cycle status if it ran,
    otherwise None. Non-blocking on lock contention (skips if a cycle is
    already in flight).
    """
    from autonomy.config import get_autonomy_config

    config = get_autonomy_config()
    if not config.get("enabled"):
        return None
    if not _interval_elapsed(config):
        return None

    if not _CYCLE_LOCK.acquire(blocking=False):
        logger.debug("autonomy: previous cycle still running, skipping tick")
        return None
    try:
        logger.info("autonomy: running cycle")
        return run_autonomy_cycle(config)
    finally:
        _CYCLE_LOCK.release()
