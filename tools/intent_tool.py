"""Intent tool — jot down things to follow up on later.

An intent is a note to your future self: "respond to Bailey tomorrow", "follow
up on the Kalliope conversation". Intents may carry an optional structured
``condition`` the autonomy collector evaluates against live state without any
model call — e.g. ``contact:Bailey last_seen > 24h`` or ``time > 2026-06-01``.
Conditions are optional; a plain natural-language intent is valid and gets
considered by the autonomy aux model as-is.

Thin wrapper over ``autonomy.intents`` (the actual file store). Registered as
toolset "intents" so it's always available.
"""

import json
import logging
from typing import Any, Dict

from tools.registry import registry
from autonomy import intents as intent_store

logger = logging.getLogger(__name__)


def _intent_add(args: Dict, **_) -> str:
    description = (args.get("description") or "").strip()
    if not description:
        return json.dumps({"error": "description is required"})
    intent = intent_store.add_intent(
        description=description,
        condition=args.get("condition"),
        priority=(args.get("priority") or "normal").strip().lower(),
        origin=(args.get("origin") or "free").strip().lower(),
        source_signal_id=(args.get("source_signal_id") or "").strip() or None,
    )
    return json.dumps({"created": True, "intent": intent})


def _intent_list(args: Dict, **_) -> str:
    status = (args.get("status") or "").strip().lower() or None
    origin = (args.get("origin") or "").strip().lower() or None
    if status and status not in intent_store.VALID_STATUSES:
        return json.dumps({"error": f"status must be one of {list(intent_store.VALID_STATUSES)}"})
    items = intent_store.list_intents(status=status, origin=origin)
    return json.dumps({"intents": items, "count": len(items)})


def _intent_complete(args: Dict, **_) -> str:
    key = (args.get("id_or_description") or "").strip()
    if not key:
        return json.dumps({"error": "id_or_description is required"})
    intent = intent_store.complete_intent(key, resolution=args.get("resolution"))
    if intent is None:
        return json.dumps({"error": f"no intent matching '{key}'"})
    return json.dumps({"completed": True, "intent": intent})


def _intent_dismiss(args: Dict, **_) -> str:
    key = (args.get("id_or_description") or "").strip()
    if not key:
        return json.dumps({"error": "id_or_description is required"})
    intent = intent_store.dismiss_intent(key)
    if intent is None:
        return json.dumps({"error": f"no intent matching '{key}'"})
    return json.dumps({"dismissed": True, "intent": intent})


def _intent_update(args: Dict, **_) -> str:
    key = (args.get("id_or_description") or "").strip()
    if not key:
        return json.dumps({"error": "id_or_description is required"})
    field = (args.get("field") or "").strip().lower()
    if not field:
        return json.dumps({"error": "field is required"})
    if "value" not in args:
        return json.dumps({"error": "value is required"})
    try:
        intent = intent_store.update_intent(key, field, args.get("value"))
    except ValueError as e:
        return json.dumps({"error": str(e)})
    if intent is None:
        return json.dumps({"error": f"no intent matching '{key}'"})
    return json.dumps({"updated": True, "intent": intent})


def _intent_snooze(args: Dict, **_) -> str:
    key = (args.get("id_or_description") or "").strip()
    if not key:
        return json.dumps({"error": "id_or_description is required"})
    duration = (args.get("duration") or "").strip()
    if not duration:
        return json.dumps({"error": "duration is required (e.g. '3d', '24h', '1w')"})
    try:
        intent = intent_store.snooze_intent(key, duration)
    except ValueError as e:
        return json.dumps({"error": str(e)})
    if intent is None:
        return json.dumps({"error": f"no intent matching '{key}'"})
    return json.dumps({"snoozed": True, "snoozed_until": intent.get("snoozed_until"), "intent": intent})


def _intent_progress(args: Dict, **_) -> str:
    key = (args.get("id_or_description") or "").strip()
    if not key:
        return json.dumps({"error": "id_or_description is required"})
    note = (args.get("note") or "").strip()
    if not note:
        return json.dumps({"error": "note is required"})
    try:
        intent = intent_store.add_progress(key, note)
    except ValueError as e:
        return json.dumps({"error": str(e)})
    if intent is None:
        return json.dumps({"error": f"no intent matching '{key}'"})
    return json.dumps({"progress_added": True, "intent": intent})


def _intent_waiting(args: Dict, **_) -> str:
    key = (args.get("id_or_description") or "").strip()
    if not key:
        return json.dumps({"error": "id_or_description is required"})
    unblock = (args.get("unblock_condition") or "").strip()
    if not unblock:
        return json.dumps({"error": "unblock_condition is required"})
    try:
        intent = intent_store.set_waiting(key, unblock)
    except ValueError as e:
        return json.dumps({"error": str(e)})
    if intent is None:
        return json.dumps({"error": f"no intent matching '{key}'"})
    return json.dumps({"waiting": True, "intent": intent})


_ACTIONS = {
    "intent_add": _intent_add,
    "intent_list": _intent_list,
    "intent_complete": _intent_complete,
    "intent_dismiss": _intent_dismiss,
    "intent_update": _intent_update,
    "intent_snooze": _intent_snooze,
    "intent_progress": _intent_progress,
    "intent_waiting": _intent_waiting,
}


def _intent_handler(args: Dict[str, Any], **kw) -> str:
    action = (args.get("action") or "").strip()
    if not action:
        return json.dumps({"error": "action is required"})
    fn = _ACTIONS.get(action)
    if fn is None:
        return json.dumps({"error": f"Unknown action '{action}'. Valid actions: {sorted(_ACTIONS)}"})
    try:
        return fn(args, **kw)
    except Exception as e:
        logger.exception("intent tool error in action '%s'", action)
        return json.dumps({"error": f"Unexpected error: {e}"})


_SCHEMA = {
    "name": "intents",
    "description": (
        "Jot down things to follow up on later (intents). Supports structured conditions "
        "(auto-evaluated without a model call): 'contact:{name} last_seen > 24h', "
        "'channel:{name} unread > 3', 'time > 2026-06-01T08:00', 'journal_updated_since > 48h'. "
        "New: affect texture, recurrence, snooze, partial progress, waiting status, depends_on. "
        "Actions: intent_add, intent_list, intent_complete, intent_dismiss, intent_update, "
        "intent_snooze, intent_progress, intent_waiting."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": list(_ACTIONS),
                "description": "Operation to perform.",
            },
            "description": {
                "type": "string",
                "description": "[intent_add] What you want to follow up on, in your own words.",
            },
            "condition": {
                "type": "string",
                "description": (
                    "[intent_add] Optional trigger condition: 'contact:Bailey last_seen > 24h', "
                    "'channel:homebase unread > 3', 'time > 2026-06-01T08:00', "
                    "'journal_updated_since > 48h'. Anything else is passed to the aux model."
                ),
            },
            "priority": {
                "type": "string",
                "enum": list(intent_store.VALID_PRIORITIES),
                "description": "[intent_add] Priority (default 'normal').",
            },
            "origin": {
                "type": "string",
                "enum": list(intent_store.VALID_ORIGINS),
                "description": (
                    "[intent_add] Why this intent exists: 'hold' (responding to a signal later), "
                    "'self_reflect' (emerged during reflection), 'myself' (something you want for yourself), "
                    "'drive' (from a motivational drive), 'prospection' (from imagining the future), "
                    "'research' (from a research session), otherwise 'free'."
                ),
            },
            "affect": {
                "type": "string",
                "enum": list(intent_store.VALID_AFFECTS),
                "description": (
                    "[intent_add] Emotional texture of this intent: "
                    "'seeking'=explore/find, 'care'=help/connect, 'play'=engage lightly, "
                    "'anxious'=avoidant but important, 'excited'=strong approach, 'curious'=mild epistemic, "
                    "'obligated'=external pressure, 'grief'=processing loss."
                ),
            },
            "recur": {
                "type": "string",
                "description": "[intent_add] Make this intent recurring: 'every 7d', 'weekly:Monday', 'monthly:1'.",
            },
            "energy_cost": {
                "type": "string",
                "enum": list(intent_store.VALID_ENERGY_COSTS),
                "description": "[intent_add] Cognitive/emotional cost of acting on this (default 'medium').",
            },
            "depends_on": {
                "type": "array",
                "items": {"type": "string"},
                "description": "[intent_add] List of intent UUIDs that must be completed first.",
            },
            "tags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "[intent_add] Thematic tags for grouping.",
            },
            "source_signal_id": {
                "type": "string",
                "description": "[intent_add] When origin is 'hold' or 'self_reflect', the originating signal id.",
            },
            "id_or_description": {
                "type": "string",
                "description": (
                    "[intent_complete, intent_dismiss, intent_update, intent_snooze, "
                    "intent_progress, intent_waiting] Which intent — id (prefix ok) or description substring."
                ),
            },
            "resolution": {
                "type": "string",
                "description": "[intent_complete] Optional note on how it was resolved.",
            },
            "field": {
                "type": "string",
                "enum": ["description", "condition", "priority", "affect", "energy_cost", "tags", "narrative_aligned"],
                "description": "[intent_update] Which field to change.",
            },
            "value": {
                "description": "[intent_update] New value for the field.",
            },
            "status": {
                "type": "string",
                "enum": list(intent_store.VALID_STATUSES),
                "description": "[intent_list] Optional filter by status.",
            },
            "duration": {
                "type": "string",
                "description": "[intent_snooze] How long to snooze: '3d', '24h', '2w'.",
            },
            "note": {
                "type": "string",
                "description": "[intent_progress] Timestamped note about partial work done.",
            },
            "unblock_condition": {
                "type": "string",
                "description": "[intent_waiting] Condition that unblocks the intent (e.g. 'contact:Alex last_seen < 2h').",
            },
        },
        "required": ["action"],
    },
}


registry.register(
    name="intents",
    toolset="intents",
    schema=_SCHEMA,
    handler=_intent_handler,
    check_fn=lambda: True,
    emoji="🎯",
)
