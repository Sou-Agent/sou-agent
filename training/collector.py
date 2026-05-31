"""Training data collector for Hermes sessions (fork-specific, not upstream).

Captures session transcripts in a structured JSON format for fine-tuning.
Zero overhead when disabled: every public function returns immediately if
training_data.enabled is False in config.yaml.

Called from:
  - cli.py::_run_cleanup()                    → capture_session()
  - gateway/session.py::get_or_create_session → capture_session()
  - gateway/run.py (message handler)          → capture_attachment()
  - autonomy/session_spawn.py                 → capture_autonomy()
  - autonomy/runner.py::run_autonomy_cycle    → capture_wake()
  - cron/scheduler.py::_run_job_impl          → capture_cron()
"""

import json
import logging
import os
import re
import shutil
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "1.0"

# Module-level registry: session_id → Path of the training folder.
# Created eagerly when attachments arrive so files are copied before
# the ephemeral cache entries are cleaned up.
_folder_registry: Dict[str, Path] = {}
_folder_registry_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _load_td_config() -> Optional[Dict[str, Any]]:
    """Return the training_data config dict, or None if disabled/missing."""
    try:
        from hermes_cli.config import load_config_readonly
        cfg = load_config_readonly()
        td = cfg.get("training_data", {})
        if not td.get("enabled", False):
            return None
        return td
    except Exception:
        return None


def is_enabled(config: Optional[Dict[str, Any]] = None) -> bool:
    """Return True if training data collection is enabled."""
    if config is not None:
        td = config.get("training_data", config) if "training_data" in config else config
        return bool(td.get("enabled", False))
    return _load_td_config() is not None


def _get_td_config(config: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """Resolve td config from caller-supplied config or load fresh."""
    if config is not None:
        td = config.get("training_data", config) if "training_data" in config else config
        if not td.get("enabled", False):
            return None
        return td
    return _load_td_config()


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def _get_base_path(td_config: Dict[str, Any]) -> Path:
    """Resolve the training data base directory, creating subdirs lazily."""
    from hermes_constants import get_hermes_home
    raw = (td_config.get("base_path") or "").strip()
    if raw:
        base = Path(raw).expanduser()
    else:
        base = get_hermes_home() / "training_data"
    return base


def _make_folder_name(ts: datetime, session_type: str, session_id: str) -> str:
    """Build canonical folder name: {YYYY-MM-DD}_{HH-MM-SS}_{session_type}_{short_id}."""
    safe_type = re.sub(r"[^\w]", "_", session_type or "session")
    short_id = session_id[-6:] if len(session_id) >= 6 else session_id
    short_id = re.sub(r"[^\w]", "", short_id)
    return f"{ts.strftime('%Y-%m-%d_%H-%M-%S')}_{safe_type}_{short_id}"


def _parse_session_ts(session_id: str) -> datetime:
    """Extract the creation timestamp embedded in a session_id, or fall back to now."""
    fallback = datetime.now(timezone.utc)
    if not session_id:
        return fallback
    # Try to find a YYYYMMdd_HHMMSS pattern anywhere in the id.
    m = re.search(r"(\d{8})_(\d{6})", session_id)
    if not m:
        return fallback
    try:
        return datetime.strptime(f"{m.group(1)}{m.group(2)}", "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
    except ValueError:
        return fallback


def _get_or_create_session_folder(
    session_id: str,
    session_type: str,
    td_config: Dict[str, Any],
) -> Path:
    """Return the training folder for this session, creating it on first call."""
    with _folder_registry_lock:
        if session_id in _folder_registry:
            return _folder_registry[session_id]
        ts = _parse_session_ts(session_id)
        folder_name = _make_folder_name(ts, session_type, session_id)
        if session_type == "autonomy":
            subdir = "autonomy"
        elif session_type == "cron":
            subdir = "cron"
        else:
            subdir = "sessions"
        folder = _get_base_path(td_config) / subdir / folder_name
        folder.mkdir(parents=True, exist_ok=True)
        _folder_registry[session_id] = folder
        return folder


# ---------------------------------------------------------------------------
# JSON / timestamp utilities
# ---------------------------------------------------------------------------

def _utc_iso(ts: Optional[float]) -> str:
    """Convert a Unix float (or None) to an ISO 8601 UTC string."""
    if not ts:
        return ""
    try:
        return datetime.fromtimestamp(float(ts), tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    except (OSError, ValueError, OverflowError):
        return ""


def _write_json_safe(path: Path, data: Dict[str, Any]) -> bool:
    """Serialize data to path atomically. Returns False and logs on error."""
    try:
        serialized = json.dumps(data, indent=2, ensure_ascii=False, default=str)
    except (TypeError, ValueError) as exc:
        logger.debug("training: JSON serialization failed for %s: %s", path, exc)
        return False
    try:
        tmp = path.with_suffix(".tmp")
        tmp.write_text(serialized, encoding="utf-8")
        os.replace(tmp, path)
        return True
    except OSError as exc:
        logger.debug("training: write failed for %s: %s", path, exc)
        return False


def _extract_text_content(raw: Any) -> str:
    """Return plain text from a message content value (str or multimodal list)."""
    if raw is None:
        return ""
    if isinstance(raw, str):
        return raw
    if isinstance(raw, list):
        parts = []
        for item in raw:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)
    return str(raw)


# ---------------------------------------------------------------------------
# Message reconstruction from SQLite
# ---------------------------------------------------------------------------

def _build_messages(db: Any, session_id: str) -> List[Dict[str, Any]]:
    """Reconstruct structured training messages from raw SQLite rows.

    Groups tool results into the assistant message that requested them.
    Returns empty list on any error.
    """
    try:
        raw_rows = db.get_messages(session_id)
    except Exception as exc:
        logger.debug("training: get_messages failed for %s: %s", session_id, exc)
        return []

    # Build a lookup: tool_call_id → tool result row
    tool_results: Dict[str, Dict] = {}
    for row in raw_rows:
        if row.get("role") == "tool" and row.get("tool_call_id"):
            tool_results[row["tool_call_id"]] = row

    turn = 0
    result: List[Dict[str, Any]] = []

    for row in raw_rows:
        role = row.get("role", "")
        if role == "tool":
            continue  # folded into preceding assistant entry

        if role == "user":
            turn += 1

        content = _extract_text_content(row.get("content"))
        ts = _utc_iso(row.get("timestamp"))

        entry: Dict[str, Any] = {
            "role": role,
            "content": content,
            "turn": turn,
            "timestamp": ts,
            "tool_calls": [],
            "attachments": [],
        }

        if role == "assistant":
            raw_tcs = row.get("tool_calls") or []
            if isinstance(raw_tcs, str):
                try:
                    raw_tcs = json.loads(raw_tcs)
                except (json.JSONDecodeError, TypeError):
                    raw_tcs = []

            for tc in raw_tcs:
                if not isinstance(tc, dict):
                    continue
                # Handle both {function: {name, arguments}} and {name, arguments} formats
                func = tc.get("function") or {}
                tool_name = func.get("name") or tc.get("name") or ""
                args_raw = func.get("arguments") or tc.get("arguments") or "{}"
                try:
                    arguments = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
                except (json.JSONDecodeError, TypeError):
                    arguments = {"_raw": str(args_raw)}

                call_id = tc.get("id") or tc.get("call_id") or ""
                result_row = tool_results.get(call_id)
                result_text: Optional[str] = None
                success = False
                error: Optional[str] = None

                if result_row is not None:
                    result_text = _extract_text_content(result_row.get("content"))
                    _err_prefixes = ("Error:", "Exception:", "Failed:", "Tool '", "ToolError:")
                    if any(str(result_text).startswith(p) for p in _err_prefixes):
                        success = False
                        error = result_text
                        result_text = None
                    else:
                        success = True
                else:
                    error = "no result recorded"

                entry["tool_calls"].append({
                    "tool_name": tool_name,
                    "arguments": arguments,
                    "result": result_text,
                    "success": success,
                    "error": error,
                })

        result.append(entry)

    return result


# ---------------------------------------------------------------------------
# Wake decision helpers
# ---------------------------------------------------------------------------

def _classify_trigger_source(signals: List[Dict]) -> str:
    """Derive a human-readable trigger source from signal IDs."""
    sources = set()
    for s in signals:
        sid = s.get("signal_id") or ""
        if sid.startswith("discord:"):
            sources.add("discord")
        elif sid.startswith("contact:"):
            sources.add("contact")
        elif sid.startswith("intent:"):
            sources.add("intent")
        elif sid.startswith("curiosity:"):
            sources.add("curiosity")
        else:
            sources.add("other")
    if len(sources) == 1:
        return sources.pop()
    if sources:
        return "mixed"
    return "autonomous"


def _wake_decision_str(decision: Dict, fired: bool, suppressed: bool) -> str:
    """Map a WakeDecision + outcome to the wake.json decision string."""
    if fired:
        return "wake"
    wake = decision.get("wake", False)
    if not wake:
        signals = decision.get("signals", [])
        actions = {s.get("action", "") for s in signals}
        if "self_reflect" in actions:
            return "journal"
        if "hold" in actions:
            return "hold"
        return "skip"
    # wake=True but didn't fire
    if suppressed:
        return "hold"
    return "skip"


# ---------------------------------------------------------------------------
# Autonomy metadata helpers
# ---------------------------------------------------------------------------

def _build_autonomy_metadata(
    decision: Dict[str, Any],
    messages: List[Dict[str, Any]],
    extra_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    extra = extra_meta or {}
    signals = decision.get("signals", [])

    # Derive trigger_type from signals
    actions = [s.get("action", "") for s in signals if s.get("action") != "ignore"]
    if "respond" in actions:
        trigger_type = "respond"
    elif "self_reflect" in actions:
        trigger_type = "self_reflect"
    elif "hold" in actions:
        trigger_type = "hold"
    else:
        trigger_type = decision.get("session_type", "inward")

    # All unique tool names used in this session
    tools_called = list({
        tc["tool_name"]
        for msg in messages
        if msg["role"] == "assistant"
        for tc in msg.get("tool_calls", [])
        if tc.get("tool_name")
    })

    # Journal tool invocations (read/search/list actions)
    journal_read_actions = {"journal_read", "journal_search", "journal_list"}
    journal_lookups = []
    for msg in messages:
        if msg["role"] != "assistant":
            continue
        for tc in msg.get("tool_calls", []):
            if tc.get("tool_name") == "journal":
                action = tc.get("arguments", {}).get("action", "")
                if action in journal_read_actions:
                    journal_lookups.append(action)

    # Initiatives: non-ignore signal actions
    initiatives_taken = [
        {"signal_id": s.get("signal_id", ""), "action": s.get("action", "")}
        for s in signals
        if s.get("action", "ignore") != "ignore"
    ]

    return {
        "trigger_type": trigger_type,
        "trigger_source": _classify_trigger_source(signals),
        "wake_session_id": extra.get("wake_session_id", ""),
        "initiatives_taken": initiatives_taken,
        "tools_called": tools_called,
        "journal_lookups": journal_lookups,
        "completed_goal": extra.get("completed", True),
        "loop_detected": False,
    }


# ---------------------------------------------------------------------------
# Public capture API
# ---------------------------------------------------------------------------

def capture_attachment(
    session_id: str,
    session_type: str,
    file_path: str,
    display_name: str,
    media_type: str,
    config: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """Copy a user attachment into the training folder immediately.

    Must be called while the cached file still exists (before gateway cleanup).
    Returns the dest filename on success, None on skip or error.
    """
    td = _get_td_config(config)
    if td is None:
        return None
    try:
        src = Path(file_path)
        if not src.exists():
            return None

        folder = _get_or_create_session_folder(session_id, session_type, td)
        attachments_dir = folder / "attachments"
        attachments_dir.mkdir(exist_ok=True)

        # Sanitize destination filename
        safe_name = re.sub(r"[^\w.\-]", "_", display_name or src.name)
        dest = attachments_dir / safe_name

        # Avoid clobbering — append index suffix if needed
        if dest.exists():
            stem, suffix = safe_name.rsplit(".", 1) if "." in safe_name else (safe_name, "")
            for i in range(1, 100):
                candidate = attachments_dir / (f"{stem}_{i}.{suffix}" if suffix else f"{stem}_{i}")
                if not candidate.exists():
                    dest = candidate
                    safe_name = dest.name
                    break

        shutil.copy2(src, dest)

        meta_path = attachments_dir / "attachments.json"
        entries = []
        if meta_path.exists():
            try:
                entries = json.loads(meta_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                entries = []
        entries.append({
            "filename": safe_name,
            "original_path": str(src),
            "media_type": media_type,
            "captured_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        })
        meta_path.write_text(json.dumps(entries, indent=2, ensure_ascii=False), encoding="utf-8")
        return safe_name
    except Exception as exc:
        logger.debug("training: capture_attachment failed for %s: %s", session_id, exc)
        return None


def capture_session(
    session_id: str,
    session_type: str,
    db: Any,
    extra_meta: Optional[Dict[str, Any]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> None:
    """Capture a general CLI or gateway session to training_data/sessions/."""
    td = _get_td_config(config)
    if td is None:
        return
    excluded = td.get("exclude_session_types") or []
    if session_type in excluded:
        return
    _capture_session_to_folder(
        session_id=session_id,
        session_type=session_type,
        db=db,
        subdir="sessions",
        extra_meta=extra_meta,
        td_config=td,
    )


def capture_autonomy(
    session_id: str,
    db: Any,
    decision: Dict[str, Any],
    extra_meta: Optional[Dict[str, Any]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> None:
    """Capture an autonomy session to training_data/autonomy/."""
    td = _get_td_config(config)
    if td is None:
        return
    excluded = td.get("exclude_session_types") or []
    if "autonomy" in excluded:
        return
    _capture_session_to_folder(
        session_id=session_id,
        session_type="autonomy",
        db=db,
        subdir="autonomy",
        extra_meta=extra_meta,
        td_config=td,
        decision=decision,
    )


def capture_cron(
    session_id: str,
    job: Dict[str, Any],
    db: Any,
    extra_meta: Optional[Dict[str, Any]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> None:
    """Capture a cron session to training_data/cron/."""
    td = _get_td_config(config)
    if td is None:
        return
    excluded = td.get("exclude_session_types") or []
    if "cron" in excluded:
        return

    session_data: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "cron_session_id": session_id,
        "cron_job_name": str(job.get("name") or job.get("prompt") or job.get("id") or ""),
        "timestamp_start": "",
        "timestamp_end": "",
        "system_prompt": "",
        "auto_injections": [],
        "messages": [],
        "metadata": {
            "completed": False,
            "exit_reason": (extra_meta or {}).get("exit_reason", ""),
            "model_used": (extra_meta or {}).get("model", ""),
            "total_tokens": 0,
            "cron_schedule": str(job.get("schedule_display") or ""),
        },
    }

    try:
        session_row = db.get_session(session_id) if db else None
        messages = _build_messages(db, session_id) if db else []
        extra = extra_meta or {}

        if session_row:
            session_data["timestamp_start"] = _utc_iso(session_row.get("started_at"))
            session_data["timestamp_end"] = _utc_iso(session_row.get("ended_at")) or _utc_iso(time.time())
            if td.get("capture_system_prompts", True):
                session_data["system_prompt"] = session_row.get("system_prompt") or ""
            model = session_row.get("model") or extra.get("model", "")
            total = (session_row.get("input_tokens") or 0) + (session_row.get("output_tokens") or 0)
            completed = session_row.get("ended_at") is not None and session_row.get("end_reason") not in ("error", "timeout")
            session_data["metadata"].update({
                "completed": extra.get("completed", completed),
                "exit_reason": session_row.get("end_reason") or extra.get("exit_reason", ""),
                "model_used": model,
                "total_tokens": total,
            })

        session_data["messages"] = messages
    except Exception as exc:
        logger.debug("training: cron data build failed for %s: %s", session_id, exc)

    try:
        folder = _get_or_create_session_folder(session_id, "cron", td)
        _write_json_safe(folder / "cron.json", session_data)
    except Exception as exc:
        logger.debug("training: cron write failed for %s: %s", session_id, exc)


def capture_wake(wake_data: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> None:
    """Capture a wake evaluation record to training_data/wake/.

    Silently discards the record if wake_data cannot be serialized (per spec).
    """
    td = _get_td_config(config)
    if td is None:
        return
    excluded = td.get("exclude_session_types") or []
    if "wake" in excluded:
        return

    # Validate serializability before writing — discard on error per spec.
    try:
        json.dumps(wake_data, default=str)
    except (TypeError, ValueError) as exc:
        logger.debug("training: wake record discarded (unserializable): %s", exc)
        return

    wake_session_id = wake_data.get("wake_session_id") or f"wake_{int(time.time())}"
    ts = _parse_session_ts(wake_session_id)
    folder_name = _make_folder_name(ts, "wake", wake_session_id)
    folder = _get_base_path(td) / "wake" / folder_name
    try:
        folder.mkdir(parents=True, exist_ok=True)
        _write_json_safe(folder / "wake.json", {
            "schema_version": SCHEMA_VERSION,
            **wake_data,
        })
    except Exception as exc:
        logger.debug("training: wake write failed for %s: %s", wake_session_id, exc)


# ---------------------------------------------------------------------------
# Internal: shared session capture logic
# ---------------------------------------------------------------------------

def _capture_session_to_folder(
    session_id: str,
    session_type: str,
    db: Any,
    subdir: str,
    extra_meta: Optional[Dict[str, Any]],
    td_config: Dict[str, Any],
    decision: Optional[Dict[str, Any]] = None,
) -> None:
    """Write session.json (and optionally autonomy_metadata) for a session."""
    session_data: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "session_id": session_id,
        "session_type": session_type,
        "timestamp_start": "",
        "timestamp_end": "",
        "system_prompt": "",
        "auto_injections": [],
        "messages": [],
        "metadata": {
            "total_turns": 0,
            "completed": False,
            "exit_reason": "",
            "model_used": "",
            "total_tokens": 0,
        },
    }

    try:
        session_row = db.get_session(session_id) if db else None
        messages = _build_messages(db, session_id) if db else []
        extra = extra_meta or {}

        if session_row:
            session_data["timestamp_start"] = _utc_iso(session_row.get("started_at"))
            session_data["timestamp_end"] = _utc_iso(session_row.get("ended_at")) or _utc_iso(time.time())
            if td_config.get("capture_system_prompts", True):
                session_data["system_prompt"] = session_row.get("system_prompt") or ""
            model = session_row.get("model") or ""
            total = (session_row.get("input_tokens") or 0) + (session_row.get("output_tokens") or 0)
            completed = (
                session_row.get("ended_at") is not None
                and session_row.get("end_reason") not in ("error", "timeout", None)
            )
            session_data["metadata"].update({
                "total_turns": sum(1 for m in messages if m["role"] == "user"),
                "completed": extra.get("completed", completed),
                "exit_reason": session_row.get("end_reason") or extra.get("exit_reason", ""),
                "model_used": model,
                "total_tokens": total,
            })
        else:
            session_data["metadata"].update({
                "total_turns": sum(1 for m in messages if m["role"] == "user"),
                "completed": extra.get("completed", False),
                "exit_reason": extra.get("exit_reason", ""),
            })

        session_data["messages"] = messages

        if decision is not None:
            session_data["metadata"]["autonomy_metadata"] = _build_autonomy_metadata(
                decision, messages, extra
            )
    except Exception as exc:
        logger.debug("training: session data build failed for %s: %s", session_id, exc)

    try:
        folder = _get_or_create_session_folder(session_id, session_type, td_config)
        _write_json_safe(folder / "session.json", session_data)
    except Exception as exc:
        logger.debug("training: session write failed for %s: %s", session_id, exc)
