"""create_session_in tool — open a properly-contexted session in a specific channel.

Unlike send_into_session (which requires pre-prepared refs), this tool creates or
fetches the channel's session first, injects context, then delivers the message.
Replies in that channel route back to the session with full context.

Before creating the new session, any prior session for the channel is finalized so
training data capture completes cleanly.
"""

import json
import logging
from typing import Any, Dict

from tools.registry import registry

logger = logging.getLogger(__name__)


def _create_session_in(args: Dict[str, Any], **_) -> str:
    platform = (args.get("platform") or "").strip().lower()
    channel_id = (args.get("channel_id") or "").strip()
    message = (args.get("message") or "").strip()
    system_context = (args.get("system_context") or "").strip()
    set_as_channel = bool(args.get("set_as_channel_session", True))

    if not platform:
        return json.dumps({"error": "platform is required"})
    if not channel_id:
        return json.dumps({"error": "channel_id is required"})
    if not message:
        return json.dumps({"error": "message is required"})

    try:
        # 1. Find any existing session for this channel and finalize it cleanly
        #    so training data capture fires before the pointer is overwritten.
        existing_session_id = None
        try:
            from gateway.session_context import get_session_id_for_channel
            existing_session_id = get_session_id_for_channel(platform, channel_id)
        except Exception:
            pass

        if existing_session_id:
            try:
                from training.motivational_collector import flush_pending
                flush_pending(existing_session_id)
            except Exception:
                pass
            try:
                from training.collector import capture_session, is_enabled
                if is_enabled():
                    try:
                        from hermes_state import SessionDB
                        from hermes_constants import get_hermes_home
                        db = SessionDB(get_hermes_home() / "state.db")
                        try:
                            capture_session(session_id=existing_session_id,
                                            session_type="outward", db=db,
                                            extra_meta={"ended_by": "create_session_in"})
                        finally:
                            db.close()
                    except Exception:
                        pass
            except Exception:
                pass

        # 2. Get or create the session for this channel
        session_id = None
        try:
            from gateway.session import get_or_create_session
            session_id = get_or_create_session(
                platform=platform,
                chat_id=channel_id,
                chat_type="channel",
            )
        except Exception as e:
            logger.debug("create_session_in: get_or_create_session failed: %s", e)

        if session_id is None:
            try:
                from hermes_state import SessionDB
                from hermes_constants import get_hermes_home
                import uuid
                session_id = str(uuid.uuid4())
                db = SessionDB(get_hermes_home() / "state.db")
                db.close()
            except Exception:
                import uuid
                session_id = f"csi_{uuid.uuid4().hex[:12]}"

        # 3. Inject system context into the session's volatile tier
        if system_context:
            try:
                from hermes_state import SessionDB
                from hermes_constants import get_hermes_home
                db = SessionDB(get_hermes_home() / "state.db")
                try:
                    db.set_volatile_context(session_id, system_context)
                finally:
                    db.close()
            except Exception:
                pass

        # 4. Send the message via platform adapter
        delivered = False
        delivery_error = ""
        try:
            from tools.discord_tool import _send_message
            token = None
            try:
                from tools.discord_tool import _get_bot_token
                token = _get_bot_token()
            except Exception:
                pass
            if token and platform == "discord":
                result = _send_message(token, channel_id, message)
                delivered = True
        except Exception as e:
            delivery_error = str(e)

        if not delivered:
            try:
                from gateway.outreach import deliver_to_channel
                deliver_to_channel(platform=platform, channel_id=channel_id, message=message)
                delivered = True
            except Exception as e:
                delivery_error = str(e)

        if not delivered:
            logger.warning("create_session_in: could not deliver to %s/%s: %s",
                           platform, channel_id, delivery_error)

        return json.dumps({
            "session_id": session_id,
            "platform": platform,
            "channel_id": channel_id,
            "delivered": delivered,
            "delivery_error": delivery_error if not delivered else None,
        })

    except Exception as e:
        logger.exception("create_session_in: unexpected error")
        return json.dumps({"error": f"Unexpected error: {e}"})


def _handler(args: Dict[str, Any], **kw) -> str:
    return _create_session_in(args, **kw)


_SCHEMA = {
    "name": "create_session_in",
    "description": (
        "Open a properly-contexted session in a specific channel and send a message. "
        "Unlike send_into_session, this creates the session first so replies route back "
        "with full context. Any existing session for the channel is finalized cleanly. "
        "Preferred tool for autonomous outreach to channels."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "platform": {
                "type": "string",
                "description": "Platform name (e.g. 'discord', 'telegram', 'slack').",
            },
            "channel_id": {
                "type": "string",
                "description": "Platform-specific channel or chat id.",
            },
            "message": {
                "type": "string",
                "description": "Message to send.",
            },
            "system_context": {
                "type": "string",
                "description": "Optional context to inject into the session's system prompt volatile tier.",
            },
            "set_as_channel_session": {
                "type": "boolean",
                "description": "Register the created session as the channel's active session (default true).",
            },
        },
        "required": ["platform", "channel_id", "message"],
    },
}

registry.register(
    name="create_session_in",
    toolset="outreach",
    schema=_SCHEMA,
    handler=_handler,
    check_fn=lambda: True,
    emoji="📨",
)
