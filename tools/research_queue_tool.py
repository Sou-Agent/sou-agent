"""Research queue tool — defer research questions for focused investigation later."""

import json
import logging
from typing import Any, Dict

from tools.registry import registry
from autonomy import research_queue as rq_store

logger = logging.getLogger(__name__)


def _rq_add(args: Dict, **_) -> str:
    question = (args.get("question") or "").strip()
    if not question:
        return json.dumps({"error": "question is required"})
    try:
        item = rq_store.add_question(
            question=question,
            priority=(args.get("priority") or "normal").strip().lower(),
            tags=args.get("tags") or [],
            source=(args.get("source") or "").strip(),
            min_wait_hours=float(args.get("min_wait_hours") or 1),
        )
        return json.dumps({"queued": True, "item": item})
    except ValueError as e:
        return json.dumps({"error": str(e)})


def _rq_complete(args: Dict, **_) -> str:
    item_id = (args.get("id") or "").strip()
    if not item_id:
        return json.dumps({"error": "id is required"})
    item = rq_store.complete_question(item_id, summary=(args.get("summary") or "").strip())
    if item is None:
        return json.dumps({"error": f"no item with id '{item_id}'"})
    return json.dumps({"completed": True, "item": item})


def _rq_list(args: Dict, **_) -> str:
    status = (args.get("status") or "pending").strip()
    items = rq_store.list_queue(status=status)
    return json.dumps({"items": items, "count": len(items)})


_ACTIONS = {
    "add": _rq_add,
    "complete": _rq_complete,
    "list": _rq_list,
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
        logger.exception("research_queue tool error in action '%s'", action)
        return json.dumps({"error": f"Unexpected error: {e}"})


_SCHEMA = {
    "name": "research_queue",
    "description": (
        "Queue a question or topic for focused research during a future inward session. "
        "Use when you're curious about something but can't investigate now. "
        "Actions: add, complete, list."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": list(_ACTIONS),
                "description": "Operation to perform.",
            },
            "question": {
                "type": "string",
                "description": "[add] The research question or topic.",
            },
            "priority": {
                "type": "string",
                "enum": list(rq_store.VALID_PRIORITIES),
                "description": "[add] Priority (default 'normal').",
            },
            "tags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "[add] Optional tags for grouping.",
            },
            "source": {
                "type": "string",
                "description": "[add] Where this question came from.",
            },
            "min_wait_hours": {
                "type": "number",
                "description": "[add] Minimum hours before surfacing (default 1).",
            },
            "id": {
                "type": "string",
                "description": "[complete] The item id to mark as answered.",
            },
            "summary": {
                "type": "string",
                "description": "[complete] Short summary of what you found.",
            },
            "status": {
                "type": "string",
                "description": "[list] Filter by status: 'pending' (default) or 'completed'.",
            },
        },
        "required": ["action"],
    },
}

registry.register(
    name="research_queue",
    toolset="intents",
    schema=_SCHEMA,
    handler=_handler,
    check_fn=lambda: True,
    emoji="🔬",
)
