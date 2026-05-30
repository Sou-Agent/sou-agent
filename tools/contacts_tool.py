"""Contact management tool.

Provides CRUD operations over ~/.hermes/contacts.json — a platform-agnostic
contact store. Each contact may have platform entries for discord, email,
telegram, etc., allowing cross-platform lookup ("email Bailey" → finds the
contact by name and pulls platforms.email.address).

Registered as toolset "contacts" so it's always available regardless of which
messaging platforms are active.
"""

import json
import logging
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from tools.registry import registry

logger = logging.getLogger(__name__)

_CONTACTS_PATH = Path.home() / ".hermes" / "contacts.json"
_CONTACTS_LOCK = threading.Lock()


# ---------------------------------------------------------------------------
# File I/O helpers
# ---------------------------------------------------------------------------

def _load_data() -> Dict[str, Any]:
    try:
        if _CONTACTS_PATH.exists():
            return json.loads(_CONTACTS_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        pass
    return {"version": 1, "contacts": []}


def _save_data(data: Dict[str, Any]) -> None:
    _CONTACTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    _CONTACTS_PATH.write_text(json.dumps(data, indent=2))


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


# ---------------------------------------------------------------------------
# Fuzzy matching (shared with discord_send_tool imports)
# ---------------------------------------------------------------------------

def fuzzy_match_contacts(contacts: List[Dict], query: str, platform: str = None) -> List[Dict]:
    """Return contacts ranked by fuzzy match quality.

    Score priority: name exact=4, name prefix=3, name contains=2,
    alias match=2, any platform field contains query=1.
    Strips leading @ and # before comparing; case-insensitive throughout.
    Optionally filter to contacts that have a given platform entry.
    """
    q = query.lstrip("@#").strip().lower()
    if not q:
        return []
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


# ---------------------------------------------------------------------------
# Action implementations
# ---------------------------------------------------------------------------

def _contact_list(args: Dict, **_) -> str:
    platform = (args.get("platform") or "").strip().lower() or None
    with _CONTACTS_LOCK:
        data = _load_data()
    contacts = data.get("contacts", [])
    if platform:
        contacts = [c for c in contacts if platform in c.get("platforms", {})]
    return json.dumps({"contacts": contacts, "count": len(contacts)})


def _contact_find(args: Dict, **_) -> str:
    query = (args.get("query") or "").strip()
    platform = (args.get("platform") or "").strip().lower() or None
    if not query:
        return json.dumps({"error": "query is required"})
    with _CONTACTS_LOCK:
        data = _load_data()
    matches = fuzzy_match_contacts(data.get("contacts", []), query, platform=platform)
    return json.dumps({"matches": matches, "count": len(matches)})


def _contact_add(args: Dict, **_) -> str:
    name = (args.get("name") or "").strip()
    if not name:
        return json.dumps({"error": "name is required"})
    aliases = [a.strip() for a in (args.get("aliases") or []) if str(a).strip()]
    platform = (args.get("platform") or "").strip().lower() or None
    platform_data = args.get("platform_data") or {}

    with _CONTACTS_LOCK:
        data = _load_data()
        contacts = data.setdefault("contacts", [])

        # --- Dedupe: search by multiple strategies before creating ---

        # 1. Check by platform user_id (e.g. discord user_id already under another contact)
        target = None
        platform_user_id = platform_data.get("user_id", "") if platform_data else ""
        if platform and platform_user_id:
            for c in contacts:
                existing_platform = c.get("platforms", {}).get(platform, {})
                if str(existing_platform.get("user_id", "")) == str(platform_user_id):
                    target = c
                    break

        # 2. Check by exact name match
        if target is None:
            existing = fuzzy_match_contacts(contacts, name)
            if existing and existing[0].get("name", "").lower() == name.lower():
                target = existing[0]

        # 3. Check if any provided alias matches an existing contact's name or aliases
        if target is None and aliases:
            for alias in aliases:
                alias_lower = alias.strip().lower()
                if not alias_lower:
                    continue
                for c in contacts:
                    cname = c.get("name", "").lower()
                    if cname == alias_lower:
                        target = c
                        break
                    for a in c.get("aliases", []):
                        if a.lower() == alias_lower:
                            target = c
                            break
                    if target:
                        break
                if target:
                    break

        # Merge into found contact
        if target:
            # Auto-add the new name as an alias if different from current name
            if name.lower() != target.get("name", "").lower() and name not in target.get("aliases", []):
                target.setdefault("aliases", []).append(name)
            if aliases:
                current = target.setdefault("aliases", [])
                for a in aliases:
                    if a not in current:
                        current.append(a)
            if platform and platform_data:
                # Merge platform data — update existing fields rather than replace
                existing_plat = target.setdefault("platforms", {}).get(platform, {})
                if existing_plat:
                    existing_plat.update(platform_data)
                else:
                    target.setdefault("platforms", {})[platform] = platform_data
            target["updated_at"] = _now_iso()
            _save_data(data)
            return json.dumps({"merged_into": target["id"], "contact": target})

        # New contact (no match found)
        now = _now_iso()
        contact: Dict[str, Any] = {
            "id": str(uuid.uuid4()),
            "name": name,
            "aliases": aliases,
            "platforms": {},
            "added_at": now,
            "updated_at": now,
        }
        if platform and platform_data:
            contact["platforms"][platform] = platform_data
        contacts.append(contact)
        _save_data(data)
    return json.dumps({"created": True, "contact": contact})


def _contact_update(args: Dict, **_) -> str:
    contact_id = (args.get("id") or "").strip()
    if not contact_id:
        return json.dumps({"error": "id is required"})

    with _CONTACTS_LOCK:
        data = _load_data()
        contacts = data.get("contacts", [])
        target = next((c for c in contacts if c.get("id") == contact_id), None)
        if target is None:
            return json.dumps({"error": f"Contact '{contact_id}' not found"})

        if "name" in args and args["name"]:
            target["name"] = args["name"].strip()

        # Alias operations
        if "aliases" in args:
            target["aliases"] = [a.strip() for a in args["aliases"] if str(a).strip()]
        if "alias_add" in args:
            current = target.setdefault("aliases", [])
            for a in (args["alias_add"] or []):
                a = str(a).strip()
                if a and a not in current:
                    current.append(a)
        if "alias_remove" in args:
            to_remove = {str(a).strip() for a in (args["alias_remove"] or [])}
            target["aliases"] = [a for a in target.get("aliases", []) if a not in to_remove]

        # Platform add/overwrite
        platform = (args.get("platform") or "").strip().lower() or None
        platform_data = args.get("platform_data")
        if platform and platform_data is not None:
            target.setdefault("platforms", {})[platform] = platform_data

        target["updated_at"] = _now_iso()
        _save_data(data)
    return json.dumps({"updated": True, "contact": target})


def _contact_remove(args: Dict, **_) -> str:
    contact_id = (args.get("id") or "").strip()
    platform = (args.get("platform") or "").strip().lower() or None
    if not contact_id:
        return json.dumps({"error": "id is required"})

    with _CONTACTS_LOCK:
        data = _load_data()
        contacts = data.get("contacts", [])
        target = next((c for c in contacts if c.get("id") == contact_id), None)
        if target is None:
            return json.dumps({"error": f"Contact '{contact_id}' not found"})

        if platform:
            # Remove just this platform entry
            removed = target.get("platforms", {}).pop(platform, None)
            target["updated_at"] = _now_iso()
            _save_data(data)
            if removed is None:
                return json.dumps({"error": f"Contact has no '{platform}' platform entry"})
            return json.dumps({"removed_platform": platform, "contact": target})

        # Remove the whole contact
        data["contacts"] = [c for c in contacts if c.get("id") != contact_id]
        _save_data(data)
    return json.dumps({"removed": True, "id": contact_id})


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

_ACTIONS = {
    "contact_list": _contact_list,
    "contact_find": _contact_find,
    "contact_add": _contact_add,
    "contact_update": _contact_update,
    "contact_remove": _contact_remove,
}


def _contacts_handler(args: Dict[str, Any], **kw) -> str:
    action = (args.get("action") or "").strip()
    if not action:
        return json.dumps({"error": "action is required"})
    fn = _ACTIONS.get(action)
    if fn is None:
        return json.dumps({"error": f"Unknown action '{action}'. Valid actions: {sorted(_ACTIONS)}"})
    try:
        return fn(args, **kw)
    except Exception as e:
        logger.exception("contacts tool error in action '%s'", action)
        return json.dumps({"error": f"Unexpected error: {e}"})


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_SCHEMA = {
    "name": "contacts",
    "description": (
        "Manage the platform-agnostic contact list at ~/.hermes/contacts.json. "
        "Each contact can have entries for multiple platforms (discord, email, telegram, etc.). "
        "Actions: contact_list, contact_find, contact_add, contact_update, contact_remove."
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
                "description": "[contact_find] Search term — matched against name, aliases, and platform usernames. Strips leading @ and #.",
            },
            "platform": {
                "type": "string",
                "description": "[contact_find, contact_list, contact_add] Filter to contacts with this platform (e.g. 'discord', 'email'). Also used with contact_update/contact_remove to target a specific platform entry.",
            },
            "id": {
                "type": "string",
                "description": "[contact_update, contact_remove] UUID of the contact to modify or delete.",
            },
            "name": {
                "type": "string",
                "description": "[contact_add, contact_update] Display name for the contact.",
            },
            "aliases": {
                "type": "array",
                "items": {"type": "string"},
                "description": "[contact_add, contact_update] Full replace of the aliases list.",
            },
            "alias_add": {
                "type": "array",
                "items": {"type": "string"},
                "description": "[contact_update] Append these aliases without replacing existing ones.",
            },
            "alias_remove": {
                "type": "array",
                "items": {"type": "string"},
                "description": "[contact_update] Remove these specific aliases.",
            },
            "platform_data": {
                "type": "object",
                "description": "[contact_add, contact_update] Platform-specific fields, e.g. {\"user_id\": \"...\", \"username\": \"...\"} for discord or {\"address\": \"...\"} for email.",
            },
        },
        "required": ["action"],
    },
}


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

registry.register(
    name="contacts",
    toolset="contacts",
    schema=_SCHEMA,
    handler=_contacts_handler,
    check_fn=lambda: True,
    emoji="📇",
)
