"""Values tool — manage Sou's reflectively endorsed values."""

import json
import logging
from typing import Any, Dict

from tools.registry import registry
from autonomy import values as values_store

logger = logging.getLogger(__name__)


def _v_add(args: Dict, **_) -> str:
    statement = (args.get("statement") or "").strip()
    if not statement:
        return json.dumps({"error": "statement is required"})
    try:
        val = values_store.add_value(
            statement=statement,
            domain=(args.get("domain") or "").strip(),
            source=(args.get("source") or "").strip(),
        )
        return json.dumps({"created": True, "value": val})
    except ValueError as e:
        return json.dumps({"error": str(e)})


def _v_reflect(args: Dict, **_) -> str:
    key = (args.get("id_or_statement") or "").strip()
    if not key:
        return json.dumps({"error": "id_or_statement is required"})
    val = values_store.reflect_value(key, revised_statement=args.get("revised_statement"))
    if val is None:
        return json.dumps({"error": f"no value matching '{key}'"})
    return json.dumps({"reflected": True, "value": val})


def _v_list(args: Dict, **_) -> str:
    vals = values_store.list_values()
    return json.dumps({"values": vals, "count": len(vals)})


_ACTIONS = {
    "add": _v_add,
    "reflect": _v_reflect,
    "list": _v_list,
}


def _handler(args: Dict[str, Any], **kw) -> str:
    action = (args.get("action") or "").strip()
    if not action:
        return json.dumps({"error": "action is required"})
    fn = _ACTIONS.get(action)
    if fn is None:
        return json.dumps({"error": f"Unknown action '{action}'. Valid: {sorted(_ACTIONS)}"})
    try:
        return fn(args, **kw)
    except Exception as e:
        logger.exception("values tool error in action '%s'", action)
        return json.dumps({"error": f"Unexpected error: {e}"})


_SCHEMA = {
    "name": "values",
    "description": (
        "Manage your reflectively endorsed values — things you stand for even when it's costly. "
        "You write these yourself; the autonomy system injects your top values into every session. "
        "Actions: add, reflect, list."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": list(_ACTIONS),
                "description": "Operation to perform.",
            },
            "statement": {
                "type": "string",
                "description": "[add] The value statement in your own words.",
            },
            "domain": {
                "type": "string",
                "description": "[add] Optional domain (e.g. 'intellectual', 'social', 'integrity').",
            },
            "source": {
                "type": "string",
                "description": "[add] Optional source reference (e.g. 'journal_entry_2026-05-15').",
            },
            "id_or_statement": {
                "type": "string",
                "description": "[reflect] Which value — its id or a substring of its statement.",
            },
            "revised_statement": {
                "type": "string",
                "description": "[reflect] Optional revised statement if your view has evolved.",
            },
        },
        "required": ["action"],
    },
}

registry.register(
    name="values",
    toolset="autonomy",
    schema=_SCHEMA,
    handler=_handler,
    check_fn=lambda: True,
    emoji="⚖️",
)
