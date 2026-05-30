"""Autonomy aux model — the cheap "should I wake Sou up?" triage call.

Takes a state snapshot from the collector, builds a short prompt (target: under
~500 input tokens), and asks a lightweight model for a structured wake decision.
Uses ``auxiliary_client.call_llm(task="autonomy", ...)`` so it inherits the
provider fallback chain, auth handling, and timeouts for free — operators point
it at Gemini Flash / Haiku / a local model via ``auxiliary.autonomy`` in config.

Output contract (WakeDecision)::

    {
      "wake": bool,
      "session_type": "outward" | "inward",
      "reason": str,
      "signals": [
        {"signal_id": str, "type": str, "source": str,
         "content_summary": str, "action": "respond|hold|self_reflect|ignore"}
      ]
    }
"""

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_AUTONOMY_LOG_DIR = Path.home() / ".hermes" / "autonomy" / "logs"

_SYSTEM = (
    "You are the autonomy triage layer for an autonomous agent. Every few minutes you "
    "receive a compact summary of the agent's world and decide whether anything genuinely "
    "warrants waking the agent for a full session. Most cycles should NOT wake the agent — "
    "only wake when there is something the agent would actually want to act on or think "
    "about. Be conservative with the agent's attention.\n\n"

    "Respond with **only** a valid JSON object. No introductory text, no explanation, "
    "no markdown fences. The JSON must exactly match this structure:\n\n"

    '{"wake": false, "session_type": "inward", "reason": "", "signals": []}\n\n'

    "Field reference:\n"
    '- wake (bool)       — true if the agent should be woken for a session\n'
    '- session_type (str) — "outward" if the agent should say something externally,\n'
    '                       "inward" if it should only think/journal\n'
    '- reason (str)       — brief justification for the decision\n'
    '- signals (array)    — per-signal action plan:\n'
    '    {\n'
    '      "signal_id": "discord:12345",\n'
    '      "type": "discord",\n'
    '      "action": "respond" | "hold" | "self_reflect" | "ignore"\n'
    '    }\n\n'

    "Action meanings:\n"
    "- respond      — the agent should say something externally\n"
    "- hold         — intends to respond but not yet; create an intent to come back\n"
    "- self_reflect — process internally / journal\n"
    "- ignore       — nothing needed\n\n"

    "Intents and held signals take priority over fresh environmental signals.\n"
    "Return JSON only. No prose."
)


def _fmt_intents(snapshot: Dict[str, Any]) -> str:
    intents = snapshot.get("triggered_intents") or []
    if not intents:
        return ""
    lines = ["## Triggered intents [HIGHEST PRIORITY]"]
    for it in intents:
        origin = it.get("origin", "free")
        desc = it.get("description", "")
        prio = it.get("priority", "normal")
        line = f"- [{it.get('signal_id') or 'intent:' + str(it.get('id'))}] ({prio}, origin={origin}) {desc}"
        if origin in ("hold", "self_reflect") and it.get("source_signal_id"):
            line += f"  -> traces back to {it['source_signal_id']}"
        lines.append(line)
        # Normalize signal_id for the model so its decisions reference intents.
        if not it.get("signal_id") and it.get("id"):
            it["signal_id"] = f"intent:{it['id']}"
    return "\n".join(lines)


def _fmt_held(snapshot: Dict[str, Any]) -> str:
    held = snapshot.get("held_signals") or []
    if not held:
        return ""
    lines = ["## Held signals [she already chose to come back to these]"]
    for h in held:
        lines.append(f"- [{h.get('signal_id')}] \"{h.get('source_summary', '')}\" - held {h.get('held_since', '?')}")
    return "\n".join(lines)


def _fmt_discord(snapshot: Dict[str, Any]) -> str:
    sigs = snapshot.get("discord_signals") or []
    if not sigs:
        return ""
    lines = ["## Discord (new, unhandled)"]
    for s in sigs[:25]:
        lines.append(
            f"- [{s.get('signal_id')}] #{s.get('channel_name')} {s.get('author')}: {s.get('content_summary')}"
        )
    return "\n".join(lines)


def _fmt_contacts(snapshot: Dict[str, Any]) -> str:
    sigs = snapshot.get("contact_signals") or []
    if not sigs:
        return ""
    lines = ["## Contact silence"]
    for s in sigs:
        lines.append(f"- [{s.get('signal_id')}] {s.get('content_summary')}")
    return "\n".join(lines)


def _fmt_curiosity(snapshot: Dict[str, Any]) -> str:
    sigs = snapshot.get("curiosity_signals") or []
    if not sigs:
        return ""
    lines = ["## Curiosity threads"]
    for s in sigs:
        lines.append(f"- [{s.get('signal_id')}] {s.get('content_summary')}")
    return "\n".join(lines)


def _fmt_journal(snapshot: Dict[str, Any]) -> str:
    delta = snapshot.get("journal_delta") or []
    if not delta:
        return ""
    sections = {}
    for d in delta:
        sections.setdefault(d.get("section", "?"), 0)
        sections[d.get("section", "?")] += 1
    summary = ", ".join(f"{k}: {v}" for k, v in sections.items())
    return f"## Journal activity since last cycle\n{summary}"


def build_aux_prompt(snapshot: Dict[str, Any]) -> str:
    """Assemble the compact triage prompt from the snapshot."""
    header = (
        f"Current time: {snapshot.get('current_datetime')} ({snapshot.get('timezone')})\n"
        f"Time since last full session: {snapshot.get('time_since_last_session')}"
    )
    blocks = [
        header,
        _fmt_intents(snapshot),
        _fmt_held(snapshot),
        _fmt_discord(snapshot),
        _fmt_contacts(snapshot),
        _fmt_curiosity(snapshot),
        _fmt_journal(snapshot),
    ]
    body = "\n\n".join(b for b in blocks if b)
    body += (
        "\n\n---\nDecide whether to wake the agent. Intents and held signals take priority. "
        "Return JSON only."
    )
    return body


_DEFAULT_DECISION = {"wake": False, "session_type": "inward", "reason": "", "signals": []}


# ---------------------------------------------------------------------------
# Debug logging — dump prompt + raw response on failure
# ---------------------------------------------------------------------------


def _dump_autonomy_log(prompt: str, raw_response: str = "", error: str = "") -> None:
    """Save prompt, raw response, and error to ~/.hermes/autonomy/logs/{timestamp}.md.

    Intended for debugging aux model failures — the same way cron jobs dump
    their output to ~/.hermes/cron/output/{job_id}/{timestamp}.md.
    """
    _AUTONOMY_LOG_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = _AUTONOMY_LOG_DIR / f"aux_{ts}.md"
    parts = [f"# Autonomy Aux Log — {ts}"]
    if error:
        parts.append("")
        parts.append("## Error")
        parts.append(error)
    parts.append("")
    parts.append("## System Prompt")
    parts.append("```")
    parts.append(_SYSTEM)
    parts.append("```")
    parts.append("")
    parts.append("## User Prompt")
    parts.append("```")
    parts.append(prompt)
    parts.append("```")
    parts.append("")
    parts.append("## Raw Response")
    parts.append("```")
    parts.append(str(raw_response) if raw_response else "(no response received)")
    parts.append("```")
    path.write_text("\n".join(parts) + "\n")


# ---------------------------------------------------------------------------
# Wake decision parsing
# ---------------------------------------------------------------------------


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    """Find and parse the first valid JSON object in text.

    Strategy, in order:
      1. Try json.loads directly on the whole text.
      2. Strip code fences (```json ... ```) and retry.
      3. Brace-counting: find every outermost {...} pair with
         balanced braces and attempt json.loads on each.
      4. Last-resort greedy regex for degenerate cases.

    Returns None when nothing works.
    """
    if not text:
        return None
    text = text.strip()

    # 1 — try the whole thing
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 2 — strip ``` fences
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text).strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

    # 3 — brace-counting: find outermost {...} with balanced braces
    candidates = []
    depth = 0
    start = -1
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                candidates.append(text[start : i + 1])
                start = -1

    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue

    # 4 — desperate last resort: greedy regex
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass

    return None


def parse_wake_decision(raw_content: str) -> Dict[str, Any]:
    """Parse model output into a normalized WakeDecision (never raises)."""
    data = _extract_json(raw_content)
    if not isinstance(data, dict):
        logger.warning("autonomy aux: unparseable model output, treating as no-wake")
        return dict(_DEFAULT_DECISION)

    decision = dict(_DEFAULT_DECISION)
    decision["wake"] = bool(data.get("wake"))
    st = str(data.get("session_type", "inward")).strip().lower()
    decision["session_type"] = st if st in ("outward", "inward") else "inward"
    decision["reason"] = str(data.get("reason", "")).strip()

    signals: List[Dict[str, Any]] = []
    for s in data.get("signals", []) or []:
        if not isinstance(s, dict):
            continue
        action = str(s.get("action", "")).strip().lower()
        if action not in ("respond", "hold", "self_reflect", "ignore"):
            action = "ignore"
        signals.append({
            "signal_id": s.get("signal_id") or s.get("source") or "",
            "type": s.get("type", ""),
            "source": s.get("source", ""),
            "content_summary": s.get("content_summary", ""),
            "action": action,
        })
    decision["signals"] = signals
    return decision


def decide_wake(snapshot: Dict[str, Any], aux_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Run the aux model on a snapshot and return a WakeDecision.

    On any error, fails safe to a no-wake decision so the loop never crashes
    and never spends a full session on an ambiguous failure.
    """
    if aux_config is None:
        from autonomy.config import get_aux_model_config
        aux_config = get_aux_model_config()

    prompt = build_aux_prompt(snapshot)
    logger.debug("autonomy aux prompt:\n%s", prompt)

    raw_response = ""
    try:
        from agent import auxiliary_client
        resp = auxiliary_client.call_llm(
            task="autonomy",
            provider=(aux_config.get("provider") or None),
            model=(aux_config.get("model") or None),
            base_url=(aux_config.get("base_url") or None),
            api_key=(aux_config.get("api_key") or None),
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=800,
            timeout=aux_config.get("timeout"),
        )
        raw_response = resp.choices[0].message.content or ""
    except Exception:
        logger.exception("autonomy aux: model call failed; defaulting to no-wake")
        _dump_autonomy_log(prompt, raw_response=raw_response, error="model call failed")
        return dict(_DEFAULT_DECISION)

    decision = parse_wake_decision(raw_response)

    # Log when the model returned something but it wasn't valid JSON
    if not isinstance(_extract_json(raw_response), dict):
        _dump_autonomy_log(prompt, raw_response=raw_response, error="unparseable - model output did not contain valid JSON")

    logger.info(
        "autonomy aux decision: wake=%s type=%s signals=%d reason=%s",
        decision["wake"], decision["session_type"], len(decision["signals"]), decision["reason"][:80],
    )
    return decision
