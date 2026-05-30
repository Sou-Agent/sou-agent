"""Update ``last_seen`` on watched people when they message the agent.

Drives the ``contact:{name} last_seen > {duration}`` intent conditions and the
contact-silence signals — both need an actual clock that advances on every
interaction. Called from the gateway message handler.

Resolution order for an incoming sender:
  1. direct match of the platform display name against a person entry, then
  2. resolve the platform user id via contacts.json to a contact name, and
     match that against a person entry.

Only watched people (``watch: true``) with a matching entry are updated; an
unknown sender is a no-op. Never raises — a tracking failure must not break
message handling.
"""

import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def _now_iso() -> str:
    from hermes_time import now as _hermes_now
    return _hermes_now().isoformat(timespec="seconds")


def _resolve_contact_name(platform: str, user_id: str, display_name: str) -> Optional[str]:
    """Resolve a platform sender to a contact name via ~/.hermes/contacts.json."""
    try:
        from hermes_constants import get_hermes_home
        path = get_hermes_home() / "contacts.json"
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, Exception):
        return None

    uid = str(user_id or "").strip()
    for contact in data.get("contacts", []):
        plat = contact.get("platforms", {}).get(platform, {})
        if not isinstance(plat, dict):
            continue
        # Match any platform field whose value equals the user id (id, user_id, …).
        for v in plat.values():
            if isinstance(v, str) and uid and v.strip() == uid:
                return contact.get("name")
    return None


def update_last_seen(
    platform: str,
    user_id: str = "",
    display_name: str = "",
    when_iso: Optional[str] = None,
) -> bool:
    """Update last_seen for the matching watched person. Returns True if written."""
    try:
        from autonomy import watchlists

        when = when_iso or _now_iso()

        # 1. Direct display-name match against a person entry.
        if display_name:
            found = watchlists.find_person(display_name)
            if found is not None:
                return watchlists.set_last_seen(display_name, when)

        # 2. Resolve via contacts.json, then match by contact name.
        name = _resolve_contact_name(platform, user_id, display_name)
        if name:
            found = watchlists.find_person(name)
            if found is not None:
                return watchlists.set_last_seen(name, when)
    except Exception:
        logger.debug("contact_tracker: update_last_seen failed", exc_info=True)
    return False
