"""Channel store tool.

Manages ~/.hermes/channels.json — a persistent, platform-agnostic registry of
channels Sou cares about. Each entry records a channel's ID, name, description
(auto-populated from the Discord topic when adding a Discord channel), member
list, and free-form tags for retrieval.

Mirrors the contacts_tool.py pattern: single dispatcher, thread-safe file I/O,
fuzzy search on name / guild / description / tags.

Registered as toolset "channels" so it's always available.
"""

import json
import logging
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from tools.registry import registry

logger = logging.getLogger(__name__)

_CHANNELS_PATH = Path.home() / ".hermes" / "channels.json"
_CHANNELS_LOCK = threading.Lock()


# ---------------------------------------------------------------------------
# File I/O helpers
# ---------------------------------------------------------------------------

def _load_data() -> Dict[str, Any]:
    try:
        if _CHANNELS_PATH.exists():
            return json.loads(_CHANNELS_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        pass
    return {"version": 1, "channels": []}


def _save_data(data: Dict[str, Any]) -> None:
    _CHANNELS_PATH.parent.mkdir(parents=True, exist_ok=True)
    _CHANNELS_PATH.write_text(json.dumps(data, indent=2))


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


# ---------------------------------------------------------------------------
# Fuzzy matching
# ---------------------------------------------------------------------------

def fuzzy_match_channels(channels: List[Dict], query: str) -> List[Dict]:
    """Return channels ranked by fuzzy match quality.

    Score priority (channel-specific fields):
      name exact=4, name prefix=3, name contains=2,
      guild contains=2, description contains=1, tag exact=1.
    Strips leading # before comparing; case-insensitive.
    """
    q = query.lstrip("#").strip().lower()
    if not q:
        return []
    results = []
    for c in channels:
        score = 0
        name = c.get("name", "").lstrip("#").lower()
        if name == q:
            score = 4
        elif name.startswith(q):
            score = 3
        elif q in name:
            score = 2
        if score == 0:
            guild = c.get("guild", "").lower()
            if q in guild:
                score = 2
        if score == 0:
            desc = c.get("description", "").lower()
            if q in desc:
                score = 1
        if score == 0:
            for tag in c.get("tags", []):
                if q in tag.lower():
                    score = 1
                    break
        if score > 0:
            results.append((score, c))
    results.sort(key=lambda x: -x[0])
    return [c for _, c in results]


# ---------------------------------------------------------------------------
# Discord topic auto-fetch helper
# ---------------------------------------------------------------------------

def _try_fetch_discord_topic(channel_id: str) -> Optional[str]:
    """Best-effort: fetch the Discord channel topic via REST. Returns None on any failure."""
    try:
        from tools.discord_tool import _get_bot_token, _discord_request
        token = _get_bot_token()
        if not token:
            return None
        data = _discord_request("GET", f"/channels/{channel_id}", token)
        return data.get("topic") or None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Action implementations
# ---------------------------------------------------------------------------

def _channel_list(args: Dict, **_) -> str:
    platform = (args.get("platform") or "").strip().lower() or None
    with _CHANNELS_LOCK:
        data = _load_data()
    channels = data.get("channels", [])
    if platform:
        channels = [c for c in channels if c.get("platform", "").lower() == platform]
    return json.dumps({"channels": channels, "count": len(channels)})


def _channel_find(args: Dict, **_) -> str:
    query = (args.get("query") or "").strip()
    platform = (args.get("platform") or "").strip().lower() or None
    if not query:
        return json.dumps({"error": "query is required"})
    with _CHANNELS_LOCK:
        data = _load_data()
    channels = data.get("channels", [])
    if platform:
        channels = [c for c in channels if c.get("platform", "").lower() == platform]
    matches = fuzzy_match_channels(channels, query)
    return json.dumps({"matches": matches, "count": len(matches)})


def _channel_add(args: Dict, **_) -> str:
    platform = (args.get("platform") or "").strip().lower()
    if not platform:
        return json.dumps({"error": "platform is required (e.g. 'discord', 'slack')"})

    channel_id = (args.get("channel_id") or "").strip()
    if not channel_id:
        return json.dumps({"error": "channel_id is required"})

    name = (args.get("name") or "").strip()
    guild = (args.get("guild") or "").strip()
    description = (args.get("description") or "").strip() or None
    tags = [t.strip() for t in (args.get("tags") or []) if str(t).strip()]

    with _CHANNELS_LOCK:
        data = _load_data()
        channels = data.setdefault("channels", [])

        # Duplicate check: same platform + channel_id
        for existing in channels:
            if existing.get("platform") == platform and existing.get("channel_id") == channel_id:
                return json.dumps({"already_exists": True, "channel": existing})

        # Auto-fetch Discord topic if description not provided
        if not description and platform == "discord":
            description = _try_fetch_discord_topic(channel_id)

        now = _now_iso()
        channel: Dict[str, Any] = {
            "id": str(uuid.uuid4()),
            "platform": platform,
            "channel_id": channel_id,
            "name": name or f"#{channel_id}",
            "guild": guild,
            "description": description or "",
            "members": [],
            "tags": tags,
            "added_at": now,
            "updated_at": now,
        }
        channels.append(channel)
        _save_data(data)
    return json.dumps({"created": True, "channel": channel})


def _channel_update(args: Dict, **_) -> str:
    entry_id = (args.get("id") or "").strip()
    if not entry_id:
        return json.dumps({"error": "id is required"})

    with _CHANNELS_LOCK:
        data = _load_data()
        channels = data.get("channels", [])
        target = next((c for c in channels if c.get("id") == entry_id), None)
        if target is None:
            return json.dumps({"error": f"Channel entry '{entry_id}' not found"})

        if "name" in args and args["name"]:
            target["name"] = args["name"].strip()
        if "description" in args:
            target["description"] = (args["description"] or "").strip()
        if "guild" in args:
            target["guild"] = (args["guild"] or "").strip()

        # Tag operations
        if "tags" in args:
            target["tags"] = [t.strip() for t in (args["tags"] or []) if str(t).strip()]
        if "tag_add" in args:
            current = target.setdefault("tags", [])
            for t in (args["tag_add"] or []):
                t = str(t).strip()
                if t and t not in current:
                    current.append(t)
        if "tag_remove" in args:
            to_remove = {str(t).strip() for t in (args["tag_remove"] or [])}
            target["tags"] = [t for t in target.get("tags", []) if t not in to_remove]

        # Member operations
        if "member_add" in args:
            existing_ids = {m.get("user_id") for m in target.get("members", [])}
            for m in (args["member_add"] or []):
                if isinstance(m, dict) and m.get("user_id") and m["user_id"] not in existing_ids:
                    target.setdefault("members", []).append({
                        "user_id": str(m["user_id"]),
                        "display_name": str(m.get("display_name") or ""),
                        "contact_id": str(m.get("contact_id") or "") or None,
                    })
                    existing_ids.add(m["user_id"])
        if "member_remove" in args:
            to_remove = {str(uid) for uid in (args["member_remove"] or [])}
            target["members"] = [m for m in target.get("members", []) if m.get("user_id") not in to_remove]

        target["updated_at"] = _now_iso()
        _save_data(data)
    return json.dumps({"updated": True, "channel": target})


def _channel_remove(args: Dict, **_) -> str:
    entry_id = (args.get("id") or "").strip()
    if not entry_id:
        return json.dumps({"error": "id is required"})

    with _CHANNELS_LOCK:
        data = _load_data()
        channels = data.get("channels", [])
        original_len = len(channels)
        data["channels"] = [c for c in channels if c.get("id") != entry_id]
        if len(data["channels"]) == original_len:
            return json.dumps({"error": f"Channel entry '{entry_id}' not found"})
        _save_data(data)
    return json.dumps({"removed": True, "id": entry_id})


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

_ACTIONS = {
    "channel_list": _channel_list,
    "channel_find": _channel_find,
    "channel_add": _channel_add,
    "channel_update": _channel_update,
    "channel_remove": _channel_remove,
}


def _channels_handler(args: Dict[str, Any], **kw) -> str:
    action = (args.get("action") or "").strip()
    if not action:
        return json.dumps({"error": "action is required"})
    fn = _ACTIONS.get(action)
    if fn is None:
        return json.dumps({"error": f"Unknown action '{action}'. Valid actions: {sorted(_ACTIONS)}"})
    try:
        return fn(args, **kw)
    except Exception as e:
        logger.exception("channels tool error in action '%s'", action)
        return json.dumps({"error": f"Unexpected error: {e}"})


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_SCHEMA = {
    "name": "channels",
    "description": (
        "Manage the channel store at ~/.hermes/channels.json. "
        "Tracks channels across platforms with description, member list, and tags. "
        "For Discord channels, 'channel_add' auto-fetches the channel topic as description. "
        "Actions: channel_list, channel_find, channel_add, channel_update, channel_remove."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": list(_ACTIONS),
                "description": "Operation to perform.",
            },
            "query": {
                "type": "string",
                "description": "[channel_find] Search term — matched against name, guild, description, and tags. Strips leading #.",
            },
            "platform": {
                "type": "string",
                "description": "[channel_list, channel_find, channel_add] Filter or set platform (e.g. 'discord', 'slack', 'telegram').",
            },
            "id": {
                "type": "string",
                "description": "[channel_update, channel_remove] UUID of the channel entry to modify or delete.",
            },
            "channel_id": {
                "type": "string",
                "description": "[channel_add] Platform-native channel ID (numeric snowflake for Discord).",
            },
            "name": {
                "type": "string",
                "description": "[channel_add, channel_update] Display name, e.g. '#bot-home'.",
            },
            "guild": {
                "type": "string",
                "description": "[channel_add, channel_update] Server/workspace name (Discord guild name, Slack workspace, etc.).",
            },
            "description": {
                "type": "string",
                "description": "[channel_add, channel_update] Free-text description or bio. Auto-fetched from Discord topic if omitted on add.",
            },
            "tags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "[channel_add, channel_update] Full replace of the tags list.",
            },
            "tag_add": {
                "type": "array",
                "items": {"type": "string"},
                "description": "[channel_update] Append tags without replacing existing ones.",
            },
            "tag_remove": {
                "type": "array",
                "items": {"type": "string"},
                "description": "[channel_update] Remove specific tags.",
            },
            "member_add": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "user_id": {"type": "string"},
                        "display_name": {"type": "string"},
                        "contact_id": {"type": "string"},
                    },
                    "required": ["user_id"],
                },
                "description": "[channel_update] Add members to the channel. Each item needs at least 'user_id'; 'display_name' and 'contact_id' (UUID from contacts.json) are optional.",
            },
            "member_remove": {
                "type": "array",
                "items": {"type": "string"},
                "description": "[channel_update] List of user_id strings to remove from the member list.",
            },
        },
        "required": ["action"],
    },
}


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

registry.register(
    name="channels",
    toolset="channels",
    schema=_SCHEMA,
    handler=_channels_handler,
    check_fn=lambda: True,
    emoji="📺",
)
