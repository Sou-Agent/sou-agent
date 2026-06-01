"""World model tool — track topics, people, questions, and projects.

Sou uses this to declare what she's actively following. The autonomy collector
fires a stale signal when a tracked node hasn't been engaged within its interval.
"""

import json
import logging
from typing import Any, Dict

from tools.registry import registry
from autonomy import world_model as wm_store

logger = logging.getLogger(__name__)


def _wm_add(args: Dict, **_) -> str:
    label = (args.get("label") or "").strip()
    if not label:
        return json.dumps({"error": "label is required"})
    try:
        node = wm_store.add_node(
            label=label,
            node_type=(args.get("type") or "topic").strip().lower(),
            interval_hours=float(args.get("interval_hours") or 168),
            notes=(args.get("notes") or "").strip(),
        )
        return json.dumps({"created": True, "node": node})
    except ValueError as e:
        return json.dumps({"error": str(e)})


def _wm_engage(args: Dict, **_) -> str:
    node_id = (args.get("node_id") or "").strip()
    if not node_id:
        return json.dumps({"error": "node_id is required"})
    node = wm_store.engage_node(node_id)
    if node is None:
        return json.dumps({"error": f"no node with id '{node_id}'"})
    return json.dumps({"engaged": True, "node": node})


def _wm_remove(args: Dict, **_) -> str:
    node_id = (args.get("node_id") or "").strip()
    if not node_id:
        return json.dumps({"error": "node_id is required"})
    ok = wm_store.remove_node(node_id)
    if not ok:
        return json.dumps({"error": f"no node with id '{node_id}'"})
    return json.dumps({"removed": True})


def _wm_list(args: Dict, **_) -> str:
    active_only = bool(args.get("active_only", True))
    nodes = wm_store.list_nodes(active_only=active_only)
    return json.dumps({"nodes": nodes, "count": len(nodes)})


_ACTIONS = {
    "add": _wm_add,
    "engage": _wm_engage,
    "remove": _wm_remove,
    "list": _wm_list,
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
        logger.exception("world_model tool error in action '%s'", action)
        return json.dumps({"error": f"Unexpected error: {e}"})


_SCHEMA = {
    "name": "world_model",
    "description": (
        "Track topics, people, questions, and projects you want to stay engaged with. "
        "The autonomy system fires a reminder when a tracked node goes stale. "
        "Actions: add, engage, remove, list."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": list(_ACTIONS),
                "description": "Operation to perform.",
            },
            "label": {
                "type": "string",
                "description": "[add] What you're tracking, in plain words.",
            },
            "type": {
                "type": "string",
                "enum": list(wm_store.VALID_NODE_TYPES),
                "description": "[add] Category of the node (default 'topic').",
            },
            "interval_hours": {
                "type": "number",
                "description": "[add] How often to re-engage (hours). Default 168 (1 week).",
            },
            "notes": {
                "type": "string",
                "description": "[add] Optional notes about what to look for when re-engaging.",
            },
            "node_id": {
                "type": "string",
                "description": "[engage, remove] The node id to act on.",
            },
            "active_only": {
                "type": "boolean",
                "description": "[list] If false, include removed nodes. Default true.",
            },
        },
        "required": ["action"],
    },
}

registry.register(
    name="world_model",
    toolset="autonomy",
    schema=_SCHEMA,
    handler=_handler,
    check_fn=lambda: True,
    emoji="🌍",
)
