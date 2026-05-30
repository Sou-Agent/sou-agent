"""Outreach support for autonomy outward sessions.

Two jobs:
  1. Create a gateway session *up front* for an outreach target, so it exists
     before the autonomy agent sends anything and a recipient's reply continues
     the SAME session (hop-into-able) rather than spawning a cold one.
  2. Hold the per-run map of prepared sessions in a contextvar so the
     ``send_into_session`` tool can resolve a short ``ref`` to a concrete
     session + platform target.

Sessions are created via the live gateway's ``SessionStore`` (reachable
in-process through ``gateway/run.py::_gateway_runner_ref``); the autonomy loop
runs inside the gateway process, so this is available. Everything degrades to a
no-op when the gateway isn't live, so importing/using this off-gateway is safe.
"""

import contextvars
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Per-run registry of prepared sessions, keyed by a short ref (contact name /
# channel name). Set by session_spawn before the agent runs; read by the
# send_into_session tool. copy_context() in session_spawn propagates this into
# the agent's worker thread.
prepared_sessions: contextvars.ContextVar[Dict[str, Dict[str, Any]]] = contextvars.ContextVar(
    "autonomy_prepared_sessions", default={}
)


def _runner():
    try:
        from gateway.run import _gateway_runner_ref
        return _gateway_runner_ref()
    except Exception:
        return None


def ensure_hop_in_session(
    platform: str,
    chat_id: str,
    user_id: Optional[str] = None,
    chat_name: Optional[str] = None,
    chat_type: str = "dm",
    thread_id: Optional[str] = None,
    label: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Create (or fetch) the gateway session for an outreach target.

    Returns ``{session_id, platform, chat_id, thread_id, user_id, label}`` or
    None when the gateway isn't live / creation failed.
    """
    runner = _runner()
    if runner is None or getattr(runner, "session_store", None) is None:
        logger.debug("outreach: no live gateway runner; cannot pre-create session")
        return None

    try:
        from gateway.config import Platform
        from gateway.session import SessionSource

        source = SessionSource(
            platform=Platform(platform),
            chat_id=str(chat_id),
            chat_name=chat_name,
            chat_type=chat_type,
            user_id=str(user_id) if user_id else None,
            thread_id=str(thread_id) if thread_id else None,
        )
        entry = runner.session_store.get_or_create_session(source)
        return {
            "session_id": entry.session_id,
            "platform": platform,
            "chat_id": str(chat_id),
            "thread_id": str(thread_id) if thread_id else None,
            "user_id": str(user_id) if user_id else None,
            "label": label or chat_name or str(chat_id),
        }
    except Exception:
        logger.exception("outreach: ensure_hop_in_session failed for %s:%s", platform, chat_id)
        return None


def resolve_targets(contact_name: str) -> List[Dict[str, Any]]:
    """Resolve a contact name to reachable platform targets from contacts.json.

    Returns ``[{platform, chat_id, user_id}, …]`` — one per platform the
    contact has an address on. The platform entry's address field is heuristic:
    we look for common id keys (``chat_id``, ``user_id``, ``id``, ``address``,
    ``username``).
    """
    import json
    from hermes_constants import get_hermes_home

    out: List[Dict[str, Any]] = []
    try:
        path = get_hermes_home() / "contacts.json"
        if not path.exists():
            return out
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return out

    key = (contact_name or "").strip().lower()
    for contact in data.get("contacts", []):
        names = {str(contact.get("name", "")).lower()}
        names.update(str(a).lower() for a in contact.get("aliases", []) if a)
        if key not in names:
            continue
        for platform, plat in (contact.get("platforms") or {}).items():
            if not isinstance(plat, dict):
                continue
            chat_id = (
                plat.get("chat_id") or plat.get("user_id") or plat.get("id")
                or plat.get("address") or plat.get("username")
            )
            if chat_id:
                out.append({
                    "platform": platform,
                    "chat_id": str(chat_id),
                    "user_id": str(plat.get("user_id") or plat.get("id") or "") or None,
                })
        break
    return out


def set_prepared(sessions: Dict[str, Dict[str, Any]]) -> None:
    """Install the prepared-session map for the current run."""
    prepared_sessions.set(dict(sessions))


def get_prepared(ref: str) -> Optional[Dict[str, Any]]:
    """Look up a prepared session by ref (case-insensitive)."""
    table = prepared_sessions.get()
    if not table:
        return None
    if ref in table:
        return table[ref]
    low = (ref or "").strip().lower()
    for k, v in table.items():
        if k.lower() == low:
            return v
    return None


def list_prepared() -> Dict[str, Dict[str, Any]]:
    return dict(prepared_sessions.get() or {})


def discord_channel_for_session(session_id_or_ref: str) -> Optional[Dict[str, Any]]:
    """Resolve a prepared ref or session_id to a Discord channel target.

    Returns ``{channel_id, thread_id}`` when the session is a Discord session,
    else None. ``channel_id`` is the effective channel to operate on — the
    thread id when the session is a thread (Discord threads are addressed as
    channels), otherwise the chat id. Lets the Discord tools accept a
    ``session_id`` in place of a raw ``channel_id`` during the autonomy loop.
    """
    target = get_prepared(session_id_or_ref) or find_session_origin(session_id_or_ref)
    if not target:
        return None
    if str(target.get("platform", "")).lower() != "discord":
        return None
    thread_id = target.get("thread_id")
    return {
        "channel_id": str(thread_id) if thread_id else str(target.get("chat_id")),
        "thread_id": str(thread_id) if thread_id else None,
        "session_id": target.get("session_id"),
    }


def find_session_origin(session_id: str) -> Optional[Dict[str, Any]]:
    """Read a session's origin (platform/chat_id/thread_id) from sessions.json."""
    import json
    from hermes_constants import get_hermes_home

    try:
        index = get_hermes_home() / "sessions" / "sessions.json"
        if not index.exists():
            return None
        data = json.loads(index.read_text(encoding="utf-8"))
    except Exception:
        return None

    for _key, entry in data.items():
        if entry.get("session_id") != session_id:
            continue
        origin = entry.get("origin") or {}
        platform = origin.get("platform") or entry.get("platform")
        chat_id = origin.get("chat_id")
        if not platform or not chat_id:
            return None
        return {
            "session_id": session_id,
            "platform": str(platform),
            "chat_id": str(chat_id),
            "thread_id": str(origin.get("thread_id")) if origin.get("thread_id") else None,
            "user_id": str(origin.get("user_id")) if origin.get("user_id") else None,
            "label": origin.get("chat_name") or str(chat_id),
        }
    return None
