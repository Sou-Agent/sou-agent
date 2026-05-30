"""Discord outbound messaging tools: send, DM, and react.

Exposes native send/DM/react capabilities to the agent, routed through the
live DiscordAdapter so every call is recorded in the session transcript and
the sent message is injected into the target channel's session history.

All three tools are registered in the ``discord`` toolset and require
DISCORD_BOT_TOKEN in the environment.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from tools.registry import registry

logger = logging.getLogger(__name__)

_CONTACTS_PATH = Path.home() / ".hermes" / "contacts.json"
_DISCORD_FILE_SIZE_LIMIT = 25 * 1024 * 1024  # 25 MB


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_contacts() -> List[Dict[str, Any]]:
    try:
        if _CONTACTS_PATH.exists():
            return json.loads(_CONTACTS_PATH.read_text()).get("contacts", [])
    except (json.JSONDecodeError, OSError):
        pass
    return []


def _fuzzy_match_contacts(contacts: List[Dict], query: str, platform: str = None) -> List[Dict]:
    """Return contacts ranked by fuzzy match against name, aliases, and platform usernames.

    Score: name exact=4, name prefix=3, name contains=2, alias match=2, platform field=1.
    """
    q = query.lstrip("@#").strip().lower()
    results = []
    for c in contacts:
        if platform and platform not in c.get("platforms", {}):
            continue
        score = 0
        name = c.get("name", "").lower()
        if name == q:
            score = 4
        elif name.startswith(q):
            score = 3
        elif q in name:
            score = 2
        if score == 0:
            for alias in c.get("aliases", []):
                if q in alias.lower():
                    score = 2
                    break
        if score == 0:
            for plat_data in c.get("platforms", {}).values():
                for v in plat_data.values():
                    if isinstance(v, str) and q in v.lower():
                        score = 1
                        break
                if score:
                    break
        if score > 0:
            results.append((score, c))
    results.sort(key=lambda x: -x[0])
    return [c for _, c in results]


def _resolve_discord_user_id(user: str) -> Optional[str]:
    """Resolve a name/username/ID to a Discord user_id string, or None."""
    # Numeric string → direct user ID
    if user.isdigit():
        return user
    # Contact fuzzy match
    contacts = _load_contacts()
    matches = _fuzzy_match_contacts(contacts, user, platform="discord")
    if matches:
        return matches[0].get("platforms", {}).get("discord", {}).get("user_id")
    return None


def _resolve_guild_id_for_search() -> Optional[str]:
    """Best-effort: return the first guild ID from the gateway runner, for member search fallback."""
    try:
        from gateway.run import _gateway_runner_ref
        from gateway.config import Platform
        runner = _gateway_runner_ref()
        if runner is None:
            return None
        adapter = runner.adapters.get(Platform.DISCORD)
        if adapter is None:
            return None
        client = getattr(adapter, "_client", None)
        if client is None:
            return None
        guilds = list(getattr(client, "guilds", []))
        return str(guilds[0].id) if guilds else None
    except Exception:
        return None


def _inject_sent_into_channel(channel_id: str, content: str) -> None:
    """Append the sent content as an observed assistant message to the target channel's session."""
    try:
        from gateway.run import _gateway_runner_ref
        from gateway.config import Platform
        runner = _gateway_runner_ref()
        if runner is None:
            return
        best = None
        for entry in runner.session_store._entries.values():
            origin = getattr(entry, "origin", None)
            if (origin
                    and str(getattr(origin, "chat_id", "")) == str(channel_id)
                    and getattr(origin, "platform", None) == Platform.DISCORD):
                updated = getattr(entry, "updated_at", None)
                if best is None or (updated and updated > getattr(best, "updated_at", None)):
                    best = entry
        if best is None:
            return
        runner.session_store.append_to_transcript(best.session_id, {
            "role": "assistant",
            "content": content,
            "observed": True,
        })
    except Exception as e:
        logger.debug("_inject_sent_into_channel failed for %s: %s", channel_id, e)


def _resolve_channel_id(channel_id: str) -> str:
    """Resolve a channel name like '#bot-home' to its numeric ID, or return as-is."""
    if channel_id.startswith("#") or not channel_id.isdigit():
        try:
            from gateway.channel_directory import resolve_channel_name
            resolved = resolve_channel_name("discord", channel_id)
            if resolved:
                return resolved
        except Exception:
            pass
    return channel_id


def _channel_from_session(session_id: str) -> Optional[str]:
    """Resolve an autonomy session ref/id to a Discord channel id, or None.

    Lets discord_send / discord_react be driven by a ``session_id`` during the
    autonomy loop. Lazily imported to avoid a hard dependency on autonomy.
    """
    try:
        from autonomy.outreach import discord_channel_for_session
        resolved = discord_channel_for_session(session_id)
        return resolved["channel_id"] if resolved else None
    except Exception:
        return None


def _get_discord_adapter():
    """Return the live DiscordAdapter from the gateway runner, or None."""
    try:
        from gateway.run import _gateway_runner_ref
        from gateway.config import Platform
        runner = _gateway_runner_ref()
        if runner is None:
            return None
        return runner.adapters.get(Platform.DISCORD)
    except Exception:
        return None


def _validate_file(file_path: str) -> Optional[str]:
    """Return an error string if the file is invalid, or None if it's fine."""
    if not Path(file_path).exists():
        return f"File not found: {file_path}"
    size = os.path.getsize(file_path)
    if size > _DISCORD_FILE_SIZE_LIMIT:
        return f"File exceeds Discord's 25 MB upload limit ({size / (1024 * 1024):.1f} MB)"
    return None


# ---------------------------------------------------------------------------
# Tool handlers
# ---------------------------------------------------------------------------

async def _discord_send_handler(args: Dict[str, Any], **_kw) -> str:
    channel_id = (args.get("channel_id") or "").strip()
    session_id = (args.get("session_id") or "").strip()
    message = (args.get("message") or "").strip() or None
    file_path = (args.get("file_path") or "").strip() or None
    reply_to = (args.get("reply_to") or "").strip() or None

    # Allow a session_id in place of channel_id (autonomy loop). Explicit
    # channel_id wins when both are given.
    if not channel_id and session_id:
        resolved = _channel_from_session(session_id)
        if resolved:
            channel_id = resolved
        else:
            return json.dumps({"error": f"session_id '{session_id}' did not resolve to a Discord channel."})

    if not channel_id:
        return json.dumps({"error": "channel_id or session_id is required"})
    if not message and not file_path:
        return json.dumps({"error": "At least one of 'message' or 'file_path' must be provided"})

    if file_path:
        err = _validate_file(file_path)
        if err:
            return json.dumps({"error": err})

    resolved_id = _resolve_channel_id(channel_id)
    adapter = _get_discord_adapter()
    if adapter is None:
        return json.dumps({"error": "Discord adapter is not running. Is the gateway connected to Discord?"})

    await adapter.send_typing(resolved_id)
    try:
        if file_path:
            result = await adapter.send_document(resolved_id, file_path, caption=message)
        else:
            metadata = {"reply_to": reply_to} if reply_to else None
            result = await adapter.send(chat_id=resolved_id, content=message, reply_to=reply_to, metadata=metadata)
    except Exception as e:
        return json.dumps({"error": f"Send failed: {e}"})
    finally:
        await adapter.stop_typing(resolved_id)

    if not result.success:
        return json.dumps({"error": result.error or "Send failed"})

    inject_content = message or f"[file: {Path(file_path).name}]"
    _inject_sent_into_channel(resolved_id, inject_content)
    return json.dumps({"success": True, "message_id": result.message_id, "channel_id": resolved_id})


async def _discord_dm_handler(args: Dict[str, Any], **_kw) -> str:
    user = (args.get("user") or "").strip()
    message = (args.get("message") or "").strip() or None
    file_path = (args.get("file_path") or "").strip() or None

    if not user:
        return json.dumps({"error": "user is required"})
    if not message and not file_path:
        return json.dumps({"error": "At least one of 'message' or 'file_path' must be provided"})

    if file_path:
        err = _validate_file(file_path)
        if err:
            return json.dumps({"error": err})

    adapter = _get_discord_adapter()
    if adapter is None:
        return json.dumps({"error": "Discord adapter is not running. Is the gateway connected to Discord?"})

    # Resolve user → user_id
    user_id = _resolve_discord_user_id(user)
    if user_id is None:
        # Fall back to guild member search
        from tools.discord_tool import _get_bot_token, _discord_request, DiscordAPIError
        token = _get_bot_token()
        guild_id = _resolve_guild_id_for_search()
        if token and guild_id:
            try:
                members = _discord_request("GET", f"/guilds/{guild_id}/members/search", token, params={"query": user, "limit": "5"})
                if members:
                    user_id = members[0].get("user", {}).get("id")
            except (DiscordAPIError, Exception):
                pass
    if not user_id:
        return json.dumps({"error": f"Could not resolve '{user}' to a Discord user. Try adding them as a contact first."})

    dm_channel_id = await adapter.open_dm_channel(user_id)
    if not dm_channel_id:
        return json.dumps({"error": f"Could not open DM channel with user {user_id}"})

    await adapter.send_typing(dm_channel_id)
    try:
        if file_path:
            result = await adapter.send_document(dm_channel_id, file_path, caption=message)
        else:
            result = await adapter.send(chat_id=dm_channel_id, content=message)
    except Exception as e:
        return json.dumps({"error": f"DM send failed: {e}"})
    finally:
        await adapter.stop_typing(dm_channel_id)

    if not result.success:
        return json.dumps({"error": result.error or "DM send failed"})

    inject_content = message or f"[file: {Path(file_path).name}]"
    _inject_sent_into_channel(dm_channel_id, inject_content)
    return json.dumps({"success": True, "message_id": result.message_id, "dm_channel_id": dm_channel_id, "user_id": user_id})


async def _discord_react_handler(args: Dict[str, Any], **_kw) -> str:
    channel_id = (args.get("channel_id") or "").strip()
    session_id = (args.get("session_id") or "").strip()
    message_id = (args.get("message_id") or "").strip()
    emoji = (args.get("emoji") or "").strip()

    if not channel_id and session_id:
        resolved = _channel_from_session(session_id)
        if resolved:
            channel_id = resolved

    if not channel_id or not message_id or not emoji:
        return json.dumps({"error": "channel_id (or session_id), message_id, and emoji are all required"})

    adapter = _get_discord_adapter()
    if adapter is None:
        return json.dumps({"error": "Discord adapter is not running. Is the gateway connected to Discord?"})

    ok = await adapter.add_reaction_by_id(channel_id, message_id, emoji)
    if ok:
        return json.dumps({"success": True})
    return json.dumps({"error": f"Failed to add reaction '{emoji}' — check that the channel/message IDs are correct and the bot has access."})


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

_SEND_SCHEMA = {
    "name": "discord_send",
    "description": (
        "Send a message, a file, or both to a Discord channel. "
        "At least one of 'message' or 'file_path' must be provided. "
        "Provide either channel_id (numeric ID or name like '#bot-home') or, during the "
        "autonomy loop, a session_id/ref that resolves to the session's Discord channel."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "channel_id": {
                "type": "string",
                "description": "Numeric channel ID or name like '#bot-home' or 'GuildName/#channel'.",
            },
            "session_id": {
                "type": "string",
                "description": (
                    "Alternative to channel_id: an autonomy session ref or session_id that "
                    "resolves to that session's Discord channel. channel_id wins if both given."
                ),
            },
            "message": {
                "type": "string",
                "description": "Text content to send (or use as caption when sending a file).",
            },
            "file_path": {
                "type": "string",
                "description": "Absolute path to a local file to upload (max 25 MB).",
            },
            "reply_to": {
                "type": "string",
                "description": "Message ID to reply to (optional).",
            },
        },
        "required": [],
    },
}

_DM_SCHEMA = {
    "name": "discord_dm",
    "description": (
        "Send a direct message to a Discord user. "
        "'user' can be a numeric user ID, a contact name, or a Discord username. "
        "At least one of 'message' or 'file_path' must be provided."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "user": {
                "type": "string",
                "description": "Numeric user ID, contact name, or Discord username to DM.",
            },
            "message": {
                "type": "string",
                "description": "Text content to send (or use as caption when sending a file).",
            },
            "file_path": {
                "type": "string",
                "description": "Absolute path to a local file to upload (max 25 MB).",
            },
        },
        "required": ["user"],
    },
}

_REACT_SCHEMA = {
    "name": "discord_react",
    "description": "Add an emoji reaction to a Discord message.",
    "parameters": {
        "type": "object",
        "properties": {
            "channel_id": {"type": "string", "description": "Numeric channel ID containing the message."},
            "session_id": {
                "type": "string",
                "description": "Alternative to channel_id: an autonomy session ref/id resolving to the channel.",
            },
            "message_id": {"type": "string", "description": "ID of the message to react to."},
            "emoji": {"type": "string", "description": "Unicode emoji (e.g. '👍') or custom emoji in 'name:id' format."},
        },
        "required": ["message_id", "emoji"],
    },
}


async def _discord_read_dm_handler(args: Dict[str, Any], **_kw) -> str:
    user = (args.get("user") or "").strip()
    limit = int(args.get("limit") or 20)
    before = (args.get("before") or "").strip() or None
    after = (args.get("after") or "").strip() or None

    if not user:
        return json.dumps({"error": "user is required"})
    limit = max(1, min(limit, 100))

    adapter = _get_discord_adapter()
    if adapter is None:
        return json.dumps({"error": "Discord adapter is not running. Is the gateway connected to Discord?"})

    user_id = _resolve_discord_user_id(user)
    if user_id is None:
        from tools.discord_tool import _get_bot_token, _discord_request, DiscordAPIError
        token = _get_bot_token()
        guild_id = _resolve_guild_id_for_search()
        if token and guild_id:
            try:
                members = _discord_request("GET", f"/guilds/{guild_id}/members/search", token, params={"query": user, "limit": "5"})
                if members:
                    user_id = members[0].get("user", {}).get("id")
            except (DiscordAPIError, Exception):
                pass
    if not user_id:
        return json.dumps({"error": f"Could not resolve '{user}' to a Discord user."})

    dm_channel_id = await adapter.open_dm_channel(user_id)
    if not dm_channel_id:
        return json.dumps({"error": f"Could not open DM channel with user {user_id}"})

    from tools.discord_tool import _get_bot_token, _discord_request, DiscordAPIError
    token = _get_bot_token()
    if not token:
        return json.dumps({"error": "DISCORD_BOT_TOKEN not set"})

    params: Dict[str, str] = {"limit": str(limit)}
    if before:
        params["before"] = before
    if after:
        params["after"] = after

    try:
        messages = _discord_request("GET", f"/channels/{dm_channel_id}/messages", token, params=params)
    except DiscordAPIError as e:
        return json.dumps({"error": f"Discord API error {e.status}: {e.body}"})
    except Exception as e:
        return json.dumps({"error": f"Failed to fetch DM messages: {e}"})

    result = []
    for msg in messages:
        author = msg.get("author", {})
        result.append({
            "id": msg["id"],
            "content": msg.get("content", ""),
            "author": {
                "id": author.get("id"),
                "username": author.get("username"),
                "display_name": author.get("global_name"),
                "bot": author.get("bot", False),
            },
            "timestamp": msg.get("timestamp"),
            "attachments": [
                {"filename": a.get("filename"), "url": a.get("url"), "size": a.get("size")}
                for a in msg.get("attachments", [])
            ],
        })
    return json.dumps({"messages": result, "count": len(result), "dm_channel_id": dm_channel_id, "user_id": user_id})


def _discord_resolve_channel_handler(args: Dict[str, Any], **_kw) -> str:
    channel_name = (args.get("channel_name") or "").strip()
    if not channel_name:
        return json.dumps({"error": "channel_name is required"})

    resolved = _resolve_channel_id(channel_name)
    if not resolved.isdigit():
        return json.dumps({"error": f"Could not resolve '{channel_name}' to a numeric channel ID. Try refreshing the channel directory or providing the numeric ID directly."})
    return json.dumps({"channel_id": resolved, "channel_name": channel_name, "resolved": True})


_READ_DM_SCHEMA = {
    "name": "discord_read_dm",
    "description": (
        "Read recent messages from a DM conversation with a Discord user. "
        "'user' can be a numeric user ID, a contact name, or a Discord username."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "user": {
                "type": "string",
                "description": "Numeric user ID, contact name, or Discord username.",
            },
            "limit": {
                "type": "integer",
                "description": "Number of messages to return (1–100, default 20).",
                "default": 20,
            },
            "before": {
                "type": "string",
                "description": "Return messages before this message snowflake ID (for pagination).",
            },
            "after": {
                "type": "string",
                "description": "Return messages after this message snowflake ID (for pagination).",
            },
        },
        "required": ["user"],
    },
}

_RESOLVE_CHANNEL_SCHEMA = {
    "name": "discord_resolve_channel",
    "description": (
        "Resolve a Discord channel name (like '#bot-home' or 'GuildName/#channel') to its numeric channel ID. "
        "Useful for cron jobs that need to address channels by ID."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "channel_name": {
                "type": "string",
                "description": "Channel name to resolve, e.g. '#bot-home', 'GuildName/#alerts', or a numeric ID (passed through unchanged).",
            },
        },
        "required": ["channel_name"],
    },
}


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

from tools.discord_tool import check_discord_tool_requirements  # noqa: E402

registry.register(
    name="discord_send",
    toolset="discord",
    schema=_SEND_SCHEMA,
    handler=lambda args, **kw: _discord_send_handler(args, **kw),
    check_fn=check_discord_tool_requirements,
    requires_env=["DISCORD_BOT_TOKEN"],
    is_async=True,
    emoji="📤",
)

registry.register(
    name="discord_dm",
    toolset="discord",
    schema=_DM_SCHEMA,
    handler=lambda args, **kw: _discord_dm_handler(args, **kw),
    check_fn=check_discord_tool_requirements,
    requires_env=["DISCORD_BOT_TOKEN"],
    is_async=True,
    emoji="💬",
)

registry.register(
    name="discord_react",
    toolset="discord",
    schema=_REACT_SCHEMA,
    handler=lambda args, **kw: _discord_react_handler(args, **kw),
    check_fn=check_discord_tool_requirements,
    requires_env=["DISCORD_BOT_TOKEN"],
    is_async=True,
    emoji="👍",
)

registry.register(
    name="discord_read_dm",
    toolset="discord",
    schema=_READ_DM_SCHEMA,
    handler=lambda args, **kw: _discord_read_dm_handler(args, **kw),
    check_fn=check_discord_tool_requirements,
    requires_env=["DISCORD_BOT_TOKEN"],
    is_async=True,
    emoji="📥",
)

registry.register(
    name="discord_resolve_channel",
    toolset="discord",
    schema=_RESOLVE_CHANNEL_SCHEMA,
    handler=_discord_resolve_channel_handler,
    check_fn=check_discord_tool_requirements,
    requires_env=["DISCORD_BOT_TOKEN"],
    emoji="🔍",
)
