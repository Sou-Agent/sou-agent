"""Spawn a full agent session from a wake decision.

This is the bridge from "the aux model said wake" to a real Hermes session —
same primary model, same system prompt, same tools as a user-initiated chat.
The agent is constructed exactly like a cron job (see
``cron/scheduler.py::_run_job_impl``) and run on a worker thread via
``contextvars.copy_context()``, but with two deliberate differences:

  - ``skip_context_files=False`` / ``skip_memory=False`` — Sou wakes with her
    full identity and memory, not the stripped cron context.
  - ``platform="autonomy"`` — so tools can tell this apart from cron/gateway.

Outward delivery is NOT handled here: an outward session uses Sou's own Discord
tools to send messages during the run, exactly as she would in a live chat.

After the run, decisions are persisted to the response log, and any signal the
aux model marked ``hold`` is linked to the intent Sou created for it (matched by
``source_signal_id``), so future cycles can reconstruct the history.
"""

import concurrent.futures
import contextvars
import json
import logging
import os
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from autonomy import response_log
from autonomy import intents as intent_store
from autonomy.config import get_autonomy_config, get_autonomy_state_dir

logger = logging.getLogger(__name__)

# Rough chars-per-token for cost estimation (model-independent heuristic).
_CHARS_PER_TOKEN = 4
# Rolling window over which estimated token spend decays for rate limiting.
_BUDGET_WINDOW = timedelta(hours=1)


def _now():
    from hermes_time import now as _hermes_now
    return _hermes_now()


def _session_state_path() -> Path:
    return get_autonomy_state_dir() / "last_session.json"


def _load_session_state() -> Dict[str, Any]:
    path = _session_state_path()
    try:
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
    except (json.JSONDecodeError, OSError):
        pass
    return {"sessions": []}


def _save_session_state(state: Dict[str, Any]) -> None:
    path = _session_state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2), encoding="utf-8")


def _parse_iso(value):
    if not value:
        return None
    try:
        from datetime import datetime, timezone
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (ValueError, TypeError):
        return None


def _recent_token_spend(state: Dict[str, Any], now) -> int:
    """Sum estimated tokens of sessions within the rolling budget window."""
    cutoff = now - _BUDGET_WINDOW
    total = 0
    for s in state.get("sessions", []):
        ts = _parse_iso(s.get("at"))
        if ts is not None and ts >= cutoff:
            total += int(s.get("estimated_tokens", 0))
    return total


def check_rate_limit(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Return ``{"allowed": bool, "reason": str}``.

    Blocks when the estimated token spend of autonomy sessions in the last hour
    is at or above ``min_tokens_between_full_sessions``. Higher config value =>
    fewer full sessions. Decays over a 1-hour window.
    """
    if config is None:
        config = get_autonomy_config()
    budget = int(config.get("min_tokens_between_full_sessions", 10000) or 0)
    if budget <= 0:
        return {"allowed": True, "reason": "no rate limit configured"}
    now = _now()
    spent = _recent_token_spend(_load_session_state(), now)
    if spent >= budget:
        return {"allowed": False, "reason": f"rate limited: ~{spent} tokens spent in last hour (budget {budget})"}
    return {"allowed": True, "reason": f"~{spent}/{budget} tokens used this hour"}


def _record_session(estimated_tokens: int, session_id: str) -> None:
    state = _load_session_state()
    now = _now()
    state.setdefault("sessions", []).append({
        "at": now.isoformat(timespec="seconds"),
        "session_id": session_id,
        "estimated_tokens": int(estimated_tokens),
    })
    # Keep only the last hour-and-a-bit of history.
    cutoff = now - timedelta(hours=2)
    state["sessions"] = [s for s in state["sessions"] if (_parse_iso(s.get("at")) or now) >= cutoff]
    state["last_session_at"] = now.isoformat(timespec="seconds")
    _save_session_state(state)


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def build_injected_prompt(snapshot: Dict[str, Any], decision: Dict[str, Any],
                           prepared: Optional[Dict[str, Any]] = None) -> str:
    """Turn a wake decision + snapshot context into the session's opening message.

    Tells Sou why she woke, what each signal is about (with full context from
    the collector), and what tool to use to respond. ``prepared`` is the
    {ref: prepared_session} map from outreach pre-creation.

    The key rule: for signals marked ``respond``, she MUST actually call a send
    tool — text output alone reaches no one.
    """
    signal_contexts = _build_signal_context(snapshot, decision)
    lines = [
        "[autonomy wake]",
        "",
        "You've chosen to wake yourself. Act naturally — this is your own initiative.",
        "",
        f"Reason: {decision.get('reason') or '(unspecified)'}",
        f"Session type: {decision.get('session_type', 'inward')} "
        "(outward = you intend to send messages externally; inward = think/journal only)",
        "",
        "--- DELIVERY RULES ---",
        "Your final free-form response is DISCARDED — no one sees it.",
        "The ONLY way to reach anyone is to call a tool directly:",
        "",
        "  discord_send(session_id='...', message='...')  — send to a Discord channel/DM",
        "  discord_dm(user='...', message='...')             — send a Discord DM",
        "  send_into_session(ref='...', message='...')       — send into a prepared session",
        "  journal(section='...', content='...')              — write to your journal",
        "  intents(action='create', ...)                      — record an intent for later",
        "",
        "If you decided a signal should get a 'respond' action, you MUST actually",
        "call one of the send tools above. Writing the response as your final",
        "message discards it.",
        "",
        "--- PREPARED SESSIONS ---",
    ]

    if prepared:
        lines.append("Gateway sessions are already prepared for each respond target. Use")
        lines.append("these refs with send_into_session:")
        for ref, sess in prepared.items():
            label = sess.get("label", ref)
            chat_type = sess.get("chat_type", "?")
            platform = sess.get("platform", "?")
            session_id = sess.get("session_id", "?")
            lines.append(f"  ref '{ref}' → {chat_type} on {platform} (session_id={session_id})")
    else:
        lines.append("  (none — no outward targets were prepared)")

    lines += [
        "",
        "--- SIGNAL CONTEXT ---",
        "The aux model decided each signal's action. Context from the collector:",
        "",
    ]

    for s in decision.get("signals", []):
        action = s.get("action", "ignore")
        sid = s.get("signal_id", "")
        ctx = signal_contexts.get(sid, {})
        lines.append(f"[{action.upper()}] {sid}")
        if ctx:
            if ctx.get("author"):
                lines.append(f"  From: {ctx['author']}")
            if ctx.get("channel_name"):
                lines.append(f"  Channel: #{ctx['channel_name']}")
            if ctx.get("content"):
                lines.append(f"  Content: {ctx['content']}")
        lines.append(f"  Reason: {s.get('content_summary', '')}")
        lines.append("")

    lines += [
        "--- GUIDANCE ---",
        "  respond      — STAYING SILENT IS NOT AN OPTION. You MUST call a send tool.",
        "                 Use send_into_session(ref=...) with one of the refs above, or",
        "                 discord_send/discord_dm with the session_id from the prepared session.",
        "  hold         — Record an intent with origin='hold' and source_signal_id set.",
        "                 You'll come back to this on the next cycle.",
        "  self_reflect — You MUST actually use journal tools. This is NOT optional.",
        "                 Call journal_write(section='thoughts', content=...) for new entries.",
        "                 Call journal_append(section='...', content=...) or journal_read(...)",
        "                 to re-engage with existing entries. Journal pruning is also valid.",
        "                 No journal interaction = you didn't do the task.",
        "  ignore       — No action needed.",
        "",
        "If nothing needs doing, just note that and end.",
        "Privacy: only explicit send tool calls reach anyone. Everything else stays private.",
    ]
    return "\n".join(lines)


def _build_signal_context(snapshot: Dict[str, Any], decision: Dict[str, Any]) -> Dict[str, dict]:
    """Cross-reference decision signal_ids against the original snapshot to
    recover full context (author, channel, content)."""
    signal_ids = {s.get("signal_id", "") for s in decision.get("signals", [])}
    ctx: Dict[str, dict] = {}

    # Match against raw discord_signals from the collector.
    for sig in (snapshot.get("discord_signals") or []):
        sid = sig.get("signal_id", "")
        if sid in signal_ids:
            ctx[sid] = {
                "author": sig.get("author", ""),
                "channel_name": sig.get("channel_name", ""),
                "content": sig.get("content_summary", ""),
            }

    # Match against contact signals.
    for sig in (snapshot.get("contact_signals") or []):
        sid = sig.get("signal_id", "")
        if sid in signal_ids:
            ctx[sid] = {
                "author": sig.get("name", ""),
                "content": sig.get("content_summary", ""),
            }

    return ctx


# ---------------------------------------------------------------------------
# Outreach session pre-creation
# ---------------------------------------------------------------------------

def _prepare_outreach_sessions(decision: Dict[str, Any]) -> Dict[str, Any]:
    """Create hop-in gateway sessions for outward respond-targets, up front.

    Returns a {ref: prepared_session} map (also installed on the outreach
    contextvar). A ref is a short handle Sou uses with send_into_session.
    """
    from autonomy import outreach

    prepared: Dict[str, Any] = {}
    for s in decision.get("signals", []):
        if s.get("action") != "respond":
            continue
        sid = str(s.get("signal_id") or "")

        if sid.startswith("discord:"):
            # discord:{channel_id}/msg:{message_id}
            try:
                channel_id = sid.split("discord:", 1)[1].split("/", 1)[0]
            except Exception:
                continue
            label = s.get("source") or f"#{channel_id}"
            sess = outreach.ensure_hop_in_session(
                platform="discord", chat_id=channel_id, chat_type="channel", label=label,
            )
            if sess:
                prepared[label] = sess

        elif sid.startswith("contact:"):
            name = sid.split("contact:", 1)[1].strip()
            targets = outreach.resolve_targets(name)
            for t in targets:
                sess = outreach.ensure_hop_in_session(
                    platform=t["platform"], chat_id=t["chat_id"], user_id=t.get("user_id"),
                    chat_name=name, chat_type="dm", label=name,
                )
                if sess:
                    # One platform → ref is the name; multiple → name@platform.
                    ref = name if len(targets) == 1 else f"{name}@{t['platform']}"
                    prepared[ref] = sess

    if prepared:
        outreach.set_prepared(prepared)
    return prepared


# ---------------------------------------------------------------------------
# Agent construction (mirrors cron/_run_job_impl, trimmed for autonomy)
# ---------------------------------------------------------------------------

def _build_agent(session_id: str):
    """Construct an AIAgent with the user's primary model, full identity + memory."""
    from run_agent import AIAgent
    from hermes_constants import get_hermes_home, parse_reasoning_effort

    home = get_hermes_home()
    _cfg: Dict[str, Any] = {}
    try:
        import yaml
        cfg_path = home / "config.yaml"
        if cfg_path.exists():
            with open(cfg_path, encoding="utf-8") as f:
                _cfg = yaml.safe_load(f) or {}
    except Exception as e:
        logger.warning("autonomy: failed to load config.yaml, using defaults: %s", e)

    model = os.getenv("HERMES_MODEL") or ""
    model_cfg = _cfg.get("model", {})
    if isinstance(model_cfg, str):
        model = model_cfg
    elif isinstance(model_cfg, dict):
        model = model_cfg.get("default", model)

    effort = str(_cfg.get("agent", {}).get("reasoning_effort", "")).strip()
    reasoning_config = parse_reasoning_effort(effort)
    max_iterations = _cfg.get("agent", {}).get("max_turns") or _cfg.get("max_turns") or 90
    pr = _cfg.get("provider_routing", {}) or {}

    from hermes_cli.runtime_provider import resolve_runtime_provider, format_runtime_provider_error
    try:
        runtime = resolve_runtime_provider(requested=None)
    except Exception as exc:
        raise RuntimeError(format_runtime_provider_error(exc)) from exc

    session_db = None
    try:
        from hermes_state import SessionDB
        session_db = SessionDB()
    except Exception as e:
        logger.debug("autonomy: SessionDB unavailable: %s", e)

    # Make MCP tools available, same as cron.
    try:
        from tools.mcp_tool import discover_mcp_tools
        discover_mcp_tools()
    except Exception as e:
        logger.debug("autonomy: MCP init failed (non-fatal): %s", e)

    agent = AIAgent(
        model=model,
        api_key=runtime.get("api_key"),
        base_url=runtime.get("base_url"),
        provider=runtime.get("provider"),
        api_mode=runtime.get("api_mode"),
        acp_command=runtime.get("command"),
        acp_args=runtime.get("args"),
        max_iterations=max_iterations,
        reasoning_config=reasoning_config,
        providers_allowed=pr.get("only"),
        providers_ignored=pr.get("ignore"),
        providers_order=pr.get("order"),
        provider_sort=pr.get("sort"),
        quiet_mode=True,
        # Unlike cron: wake with full identity, project context, and memory.
        skip_context_files=False,
        load_soul_identity=True,
        skip_memory=False,
        platform="autonomy",
        session_id=session_id,
        session_db=session_db,
    )
    return agent


def _run_agent_with_timeout(agent, prompt: str, inactivity_limit: float = 600.0) -> Dict[str, Any]:
    """Run the agent on a worker thread with an inactivity timeout (cron pattern)."""
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    ctx = contextvars.copy_context()
    future = pool.submit(ctx.run, agent.run_conversation, prompt)
    try:
        if inactivity_limit and inactivity_limit > 0:
            while True:
                done, _ = concurrent.futures.wait({future}, timeout=5.0)
                if done:
                    return future.result()
                idle = 0.0
                if hasattr(agent, "get_activity_summary"):
                    try:
                        idle = agent.get_activity_summary().get("seconds_since_activity", 0.0)
                    except Exception:
                        pass
                if idle >= inactivity_limit:
                    if hasattr(agent, "interrupt"):
                        agent.interrupt("Autonomy session timed out (inactivity)")
                    raise TimeoutError(f"autonomy session idle for {int(idle)}s")
        return future.result()
    finally:
        pool.shutdown(wait=False, cancel_futures=True)


# ---------------------------------------------------------------------------
# Hold → intent linkage
# ---------------------------------------------------------------------------

def _link_holds(decision: Dict[str, Any]) -> None:
    """For each held signal, find the intent Sou created for it and link them."""
    held_ids = {s.get("signal_id") for s in decision.get("signals", []) if s.get("action") == "hold"}
    if not held_ids:
        return
    try:
        for intent in intent_store.list_intents(status="pending", origin="hold"):
            sid = intent.get("source_signal_id")
            if sid in held_ids and intent.get("id"):
                response_log.link_intent(sid, intent["id"])
    except Exception:
        logger.exception("autonomy: failed to link holds to intents")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def spawn_autonomy_session(decision: Dict[str, Any], config: Optional[Dict[str, Any]] = None,
                            snapshot: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Run a full agent session for a wake decision. Returns a result dict.

    ``{"fired": bool, "reason": str, "final_response": str, "session_id": str}``
    """
    if config is None:
        config = get_autonomy_config()

    if not decision.get("wake"):
        return {"fired": False, "reason": "decision.wake is false", "final_response": "", "session_id": ""}

    gate = check_rate_limit(config)
    if not gate["allowed"]:
        logger.info("autonomy: session suppressed — %s", gate["reason"])
        return {"fired": False, "reason": gate["reason"], "final_response": "", "session_id": ""}

    session_id = f"autonomy_{_now().strftime('%Y%m%d_%H%M%S')}"

    # Pre-create hop-in gateway sessions for outward respond-targets BEFORE the
    # agent runs, and install them on the outreach contextvar so send_into_session
    # can resolve refs. copy_context() in _run_agent_with_timeout propagates the
    # contextvar into the worker thread.
    prepared: Dict[str, Any] = {}
    if decision.get("session_type") == "outward":
        try:
            prepared = _prepare_outreach_sessions(decision)
        except Exception:
            logger.exception("autonomy: outreach pre-creation failed")

    prompt = build_injected_prompt(snapshot or {}, decision, prepared=prepared)

    # Mark this as a cron-like internal session so the approval system applies
    # non-interactive auto-approval, and isolate session identity from any live
    # gateway message handlers.
    os.environ["HERMES_CRON_SESSION"] = "1"
    ctx_tokens = None
    try:
        from gateway.session_context import set_session_vars
        ctx_tokens = set_session_vars(platform="", chat_id="", chat_name="")
    except Exception:
        ctx_tokens = None

    final_response = ""
    try:
        agent = _build_agent(session_id)
        result = _run_agent_with_timeout(agent, prompt)
        if isinstance(result, dict):
            final_response = result.get("final_response", "") or ""
    except Exception:
        logger.exception("autonomy: session failed")
        return {"fired": False, "reason": "session error", "final_response": "", "session_id": session_id}
    finally:
        if ctx_tokens is not None:
            try:
                from gateway.session_context import clear_session_vars
                clear_session_vars(ctx_tokens)
            except Exception:
                pass

    # Persist decisions, link holds, and record token spend for rate limiting.
    try:
        response_log.record_decisions(decision.get("signals", []), session_id=session_id)
        _link_holds(decision)
    except Exception:
        logger.exception("autonomy: failed to persist decisions")

    estimated_tokens = (len(prompt) + len(final_response)) // _CHARS_PER_TOKEN
    # Floor the estimate so a near-silent inward session still counts against budget.
    estimated_tokens = max(estimated_tokens, 500)
    _record_session(estimated_tokens, session_id)

    logger.info("autonomy: session %s complete (~%d tokens)", session_id, estimated_tokens)
    return {
        "fired": True,
        "reason": decision.get("reason", ""),
        "final_response": final_response,
        "session_id": session_id,
    }
