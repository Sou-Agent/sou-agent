"""Manual autonomy trigger tool.

Allows the agent to inspect and force-run an autonomy cycle outside the normal
interval gate. Useful for debugging the autonomy loop or forcing a cycle when
signals are known to be present.
"""

import json
import logging

from tools.registry import registry

logger = logging.getLogger(__name__)


def _run(args: dict, **_) -> str:
    from autonomy.config import get_autonomy_config
    from autonomy.runner import run_autonomy_cycle
    from autonomy import collector, aux_model
    from autonomy.runner import _CYCLE_LOCK

    config = get_autonomy_config()

    if not config.get("enabled"):
        logger.info("autonomy_trigger: autonomy is disabled in config")
        return json.dumps({"error": "autonomy is disabled — set autonomy.enabled: true in config.yaml"})

    dry_run = bool(args.get("dry_run", False))

    if not _CYCLE_LOCK.acquire(blocking=False):
        logger.warning("autonomy_trigger: a cycle is already in flight, refusing to overlap")
        return json.dumps({"error": "a cycle is already running"})

    try:
        logger.info("autonomy_trigger: collecting state snapshot")
        snapshot = collector.collect_state(config)

        has_signals = snapshot.get("has_signals", False)
        signal_counts = {
            "discord": len(snapshot.get("discord_signals") or []),
            "contact_silence": len(snapshot.get("contact_signals") or []),
            "curiosity": len(snapshot.get("curiosity_signals") or []),
            "triggered_intents": len(snapshot.get("triggered_intents") or []),
            "held_signals": len(snapshot.get("held_signals") or []),
        }
        logger.info("autonomy_trigger: snapshot collected — has_signals=%s counts=%s", has_signals, signal_counts)

        if not has_signals:
            logger.info("autonomy_trigger: no signals found, cycle would stop here")
            collector.commit_cycle(snapshot, session_fired=False)
            return json.dumps({
                "result": "no_signals",
                "signal_counts": signal_counts,
                "note": "no actionable signals; cycle committed without waking",
            })

        logger.info("autonomy_trigger: running aux model triage")
        decision = aux_model.decide_wake(snapshot)
        wake = decision.get("wake", False)
        logger.info(
            "autonomy_trigger: aux decision — wake=%s session_type=%s reason=%s signals=%d",
            wake,
            decision.get("session_type"),
            decision.get("reason", "")[:120],
            len(decision.get("signals") or []),
        )

        if dry_run:
            logger.info("autonomy_trigger: dry_run=true — skipping session spawn")
            return json.dumps({
                "result": "dry_run",
                "signal_counts": signal_counts,
                "wake_decision": decision,
                "note": "dry_run=true: aux model ran but no session was spawned",
            })

        if not wake:
            logger.info("autonomy_trigger: aux model chose not to wake")
            collector.commit_cycle(snapshot, session_fired=False)
            return json.dumps({
                "result": "no_wake",
                "signal_counts": signal_counts,
                "wake_decision": decision,
            })

        logger.info("autonomy_trigger: spawning autonomy session")
        from autonomy import session_spawn
        spawn_result = session_spawn.spawn_autonomy_session(decision, config)
        fired = spawn_result.get("fired", False)
        logger.info("autonomy_trigger: spawn complete — fired=%s session_id=%s", fired, spawn_result.get("session_id"))

        collector.commit_cycle(snapshot, session_fired=fired)
        return json.dumps({
            "result": "fired" if fired else "spawn_failed",
            "signal_counts": signal_counts,
            "wake_decision": decision,
            "session_id": spawn_result.get("session_id", ""),
        })

    except Exception as e:
        logger.exception("autonomy_trigger: unexpected error")
        return json.dumps({"error": f"cycle failed: {e}"})
    finally:
        _CYCLE_LOCK.release()


_SCHEMA = {
    "name": "manually_trigger_autonomy",
    "description": (
        "Force an autonomy cycle immediately, bypassing the interval gate. "
        "Collects the current signal snapshot, runs the aux triage model, and "
        "spawns a session if warranted. Set dry_run=true to inspect the snapshot "
        "and aux decision without spawning a session."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "dry_run": {
                "type": "boolean",
                "description": "If true, collect state and run aux triage but do not spawn a session. Default false.",
            },
        },
        "required": [],
    },
}

registry.register(
    name="manually_trigger_autonomy",
    toolset="autonomy",
    schema=_SCHEMA,
    handler=_run,
    check_fn=lambda: True,
    emoji="⚙️",
)
