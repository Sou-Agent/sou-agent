"""send_into_session — speak into a specific gateway session.

The autonomy outreach system creates a gateway session *up front* for each
person/channel Sou is reaching out to. This tool sends a message into one of
those prepared sessions (or any existing session found via ``session_search``),
using the same multi-platform delivery as ``send_message`` but bound to an
exact ``session_id`` — so the message lands in that conversation and the
recipient can reply straight back into it.

Resolve order:
  - ``ref``        — a short handle for a session prepared this run
                     (offered in the autonomy wake prompt, e.g. "bailey").
  - ``session_id`` — an explicit session id (e.g. one returned by
                     ``session_search``) to continue an existing conversation.

Privacy: only the ``message`` you pass is delivered and recorded in that
session. Your internal autonomy reasoning lives in a separate private session
and is never sent or mirrored here.

Registered as toolset "outreach".
"""

import json
import logging
from typing import Any, Dict, Optional

from tools.registry import registry

logger = logging.getLogger(__name__)


def _resolve_target(args: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    from autonomy import outreach

    ref = (args.get("ref") or "").strip()
    if ref:
        return outreach.get_prepared(ref)
    session_id = (args.get("session_id") or "").strip()
    if session_id:
        prepared = outreach.get_prepared(session_id)
        if prepared:
            return prepared
        return outreach.find_session_origin(session_id)
    return None


def _build_target_string(target: Dict[str, Any]) -> str:
    platform = target["platform"]
    chat_id = target["chat_id"]
    thread_id = target.get("thread_id")
    s = f"{platform}:{chat_id}"
    if thread_id:
        s += f":{thread_id}"
    return s


def _append_to_session(session_id: str, message: str) -> bool:
    """Append the sent message to the exact session transcript (no heuristic)."""
    db = None
    try:
        from hermes_state import SessionDB
        db = SessionDB()
        db.append_message(session_id=session_id, role="assistant", content=message)
        return True
    except Exception:
        logger.debug("send_into_session: transcript append failed", exc_info=True)
        return False
    finally:
        if db is not None:
            try:
                db.close()
            except Exception:
                pass


def _send_into_session(args: Dict[str, Any], **_) -> str:
    message = (args.get("message") or "").strip()
    if not message:
        return json.dumps({"error": "message is required"})

    target = _resolve_target(args)
    if target is None:
        return json.dumps({
            "error": (
                "No session resolved. Pass 'ref' (a prepared session handle from your wake "
                "prompt) or 'session_id' (e.g. from session_search)."
            )
        })

    target_str = _build_target_string(target)

    # Reuse send_message's full multi-platform delivery path.
    try:
        from tools.send_message_tool import send_message_tool
        send_raw = send_message_tool({"action": "send", "target": target_str, "message": message})
        send_result = json.loads(send_raw)
    except Exception as e:
        logger.exception("send_into_session: delivery failed")
        return json.dumps({"error": f"delivery failed: {e}"})

    if isinstance(send_result, dict) and send_result.get("error"):
        return json.dumps({"error": send_result["error"], "target": target_str})

    # Append exactly to the known session so the recipient's reply continues it.
    appended = _append_to_session(target["session_id"], message)

    return json.dumps({
        "sent": True,
        "target": target_str,
        "session_id": target["session_id"],
        "label": target.get("label"),
        "recorded_in_session": appended,
    })


_ACTIONS = {"send_into_session": _send_into_session}


def _handler(args: Dict[str, Any], **kw) -> str:
    # Single-action tool, but keep the dispatch shape consistent with siblings.
    try:
        return _send_into_session(args, **kw)
    except Exception as e:
        logger.exception("send_into_session tool error")
        return json.dumps({"error": f"Unexpected error: {e}"})


_SCHEMA = {
    "name": "send_into_session",
    "description": (
        "Send a message into a specific conversation/session and record it there, so the "
        "recipient can reply straight back into the same thread. Multi-platform (Telegram, "
        "Slack, Signal, Discord, …). Use 'ref' for a session prepared for you this wake "
        "(listed in your wake prompt), or 'session_id' (e.g. from session_search) to continue "
        "an existing conversation. Only the message you pass is delivered — your internal "
        "reasoning stays private."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "ref": {
                "type": "string",
                "description": "Handle of a session prepared for this wake (e.g. 'bailey', '#homebase').",
            },
            "session_id": {
                "type": "string",
                "description": "Explicit session id to send into (e.g. from session_search). Use instead of 'ref'.",
            },
            "message": {
                "type": "string",
                "description": "The message to send. Supports MEDIA:<path> like send_message.",
            },
        },
        "required": ["message"],
    },
}


registry.register(
    name="send_into_session",
    toolset="outreach",
    schema=_SCHEMA,
    handler=_handler,
    check_fn=lambda: True,
    emoji="📨",
)
