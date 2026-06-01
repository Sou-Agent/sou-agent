"""Social model tool — maintain theory-of-mind models of people Sou cares about."""

import json
import logging
from typing import Any, Dict

from tools.registry import registry
from autonomy import social_cognition as sc_store

logger = logging.getLogger(__name__)


def _sm_update(args: Dict, **_) -> str:
    person_key = (args.get("person_key") or "").strip()
    if not person_key:
        return json.dumps({"error": "person_key is required"})
    field = (args.get("field") or "").strip()
    if not field:
        return json.dumps({"error": "field is required"})
    if "value" not in args:
        return json.dumps({"error": "value is required"})
    person = sc_store.update_person(person_key, field, args["value"])
    return json.dumps({"updated": True, "person": person})


def _sm_get(args: Dict, **_) -> str:
    person_key = (args.get("person_key") or "").strip()
    if not person_key:
        return json.dumps({"error": "person_key is required"})
    person = sc_store.get_person(person_key)
    if person is None:
        return json.dumps({"found": False, "person_key": person_key})
    return json.dumps({"found": True, "person": person})


def _sm_add_initiative(args: Dict, **_) -> str:
    person_key = (args.get("person_key") or "").strip()
    if not person_key:
        return json.dumps({"error": "person_key is required"})
    topic = (args.get("topic") or "").strip()
    if not topic:
        return json.dumps({"error": "topic is required"})
    person = sc_store.add_initiative(person_key, topic)
    return json.dumps({"added": True, "person": person})


def _sm_record_outreach(args: Dict, **_) -> str:
    person_key = (args.get("person_key") or "").strip()
    if not person_key:
        return json.dumps({"error": "person_key is required"})
    sc_store.record_outreach(person_key)
    return json.dumps({"recorded": True})


def _sm_list(args: Dict, **_) -> str:
    people = sc_store.list_people()
    return json.dumps({"people": people, "count": len(people)})


_ACTIONS = {
    "update": _sm_update,
    "get": _sm_get,
    "add_initiative": _sm_add_initiative,
    "record_outreach": _sm_record_outreach,
    "list": _sm_list,
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
        logger.exception("social_model tool error in action '%s'", action)
        return json.dumps({"error": f"Unexpected error: {e}"})


_SCHEMA = {
    "name": "social_model",
    "description": (
        "Maintain rich mental models of people you care about — interests, challenges, "
        "communication style, pending topics, theory-of-mind notes. The autonomy system uses "
        "these to generate purposeful outreach signals. "
        "Actions: update, get, add_initiative, record_outreach, list."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": list(_ACTIONS),
                "description": "Operation to perform.",
            },
            "person_key": {
                "type": "string",
                "description": "[update, get, add_initiative, record_outreach] The person's key (short name/handle).",
            },
            "field": {
                "type": "string",
                "description": (
                    "[update] Field to set. Any of: known_interests, known_challenges, "
                    "communication_style, relationship_quality, theory_of_mind_notes, "
                    "desired_outreach_interval_hours, last_good_exchange."
                ),
            },
            "value": {
                "description": "[update] New value for the field.",
            },
            "topic": {
                "type": "string",
                "description": "[add_initiative] Something you want to bring up with this person.",
            },
        },
        "required": ["action"],
    },
}

registry.register(
    name="social_model",
    toolset="outreach",
    schema=_SCHEMA,
    handler=_handler,
    check_fn=lambda: True,
    emoji="🧠",
)
