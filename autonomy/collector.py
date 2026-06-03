"""Zero-token state collector.

Gathers everything the autonomy aux model needs to decide whether to wake the
agent — with NO model calls. The cheaper this layer is, the more frequently it
can run. It never raises: any sub-collector that fails logs and contributes
empty data so one broken source can't stall the loop.

Evaluation order is intent-first: intents the agent explicitly wrote are the
highest-priority input, followed by held signals she chose to come back to,
then environmental signals (Discord, contact silence, curiosity).

Run standalone for debugging::

    python -m autonomy.collector
"""

import json
import logging
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from autonomy import intents as intent_store
from autonomy import response_log
from autonomy import watchlists
from autonomy.config import get_autonomy_config, get_autonomy_state_dir, get_journal_path
from autonomy import motivational_state as mot_state
from autonomy import narrative_identity
from autonomy import values as values_store
from autonomy import offline_processing
from autonomy import prospection as prospection_mod
from autonomy import social_cognition
from autonomy import metacognition
from autonomy import world_model as wm_store
from autonomy import research_queue as rq_store

logger = logging.getLogger(__name__)


def _now() -> datetime:
    from hermes_time import now as _hermes_now
    return _hermes_now()


def _cycle_state_path() -> Path:
    return get_autonomy_state_dir() / "last_cycle.json"


def _load_cycle_state() -> Dict[str, Any]:
    path = _cycle_state_path()
    try:
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
    except (json.JSONDecodeError, OSError):
        pass
    return {"last_cycle_at": None, "channel_watermarks": {}, "last_session_at": None}


def save_cycle_state(state: Dict[str, Any]) -> None:
    path = _cycle_state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2), encoding="utf-8")


def _parse_iso(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None
    # Normalize to aware so comparisons with _now() (aware) don't blow up.
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


_DURATION_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([smhdw])\s*$", re.IGNORECASE)
_UNIT_SECONDS = {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800}


def _parse_duration(text: str) -> Optional[timedelta]:
    """Parse '24h', '30m', '2d', '1w' → timedelta. None if unrecognized."""
    m = _DURATION_RE.match(text or "")
    if not m:
        return None
    return timedelta(seconds=float(m.group(1)) * _UNIT_SECONDS[m.group(2).lower()])


def _humanize_gap(delta: Optional[timedelta]) -> str:
    if delta is None:
        return "unknown"
    secs = int(delta.total_seconds())
    if secs < 60:
        return f"{secs}s"
    if secs < 3600:
        return f"{secs // 60}m"
    if secs < 86400:
        return f"{secs // 3600}h"
    return f"{secs // 86400}d"


# ---------------------------------------------------------------------------
# Signal cooldowns — prevent stale re-triggering of recently-handled signals
# ---------------------------------------------------------------------------

# Signal types eligible for cooldown after a session handles them.
# Per-signal_id cooldown: if "social:bailey" was handled, "social:bailey"
# won't fire again until the cooldown expires. Other signal_ids with
# different keys (e.g. "social:thon") are unaffected.
#
# Drive-family cooldown: when any signal from a drive family fires a session,
# ALL signals belonging to that drive family are suppressed for the cooldown
# window. This prevents cross-collector signal aliasing — e.g. the connection
# drive surfacing as drive:connection, then re-surfacing as social:bailey
# minutes later through a different collector.
_COOLDOWN_SIGNAL_TYPES = (
    "social:", "drive:", "prospection:", "world_model:",
    "narrative:", "values:", "research:", "metacognitive:",
    "offline:", "contact:", "curiosity:", "spiral:",
)
_DEFAULT_COOLDOWN_HOURS = 4

# Drive-family mapping: signal_id prefix → drive family name.
# When any signal with this prefix fires a session, all signals whose
# prefix maps to the same drive family are suppressed.
# "drive:" is special — the family is extracted from the suffix
# (e.g. "drive:connection" → family "connection").
_DRIVE_FAMILY_PREFIX_MAP = {
    "social:": "connection",
    "contact:": "connection",
    "prospection:": "reflection",
    "narrative:": "expression",
    "values:": "reflection",
    "research:": "curiosity",
    "curiosity:": "curiosity",
    "metacognitive:": "growth",
    "spiral:": "reflection",
    "offline:": "reflection",
    # world_model varies by node_type — resolved dynamically in _resolve_drive_family
    "world_model:": None,  # dynamic
}


def _resolve_drive_family(signal_id: str, signal: Optional[Dict[str, Any]] = None) -> Optional[str]:
    """Determine which drive family a signal belongs to.

    Returns a drive family name ("connection", "curiosity", "expression",
    "reflection", "play", "growth") or None if the signal doesn't belong
    to any drive family.

    For drive:* signals, the family is the drive name itself (e.g.
    "drive:connection" → "connection").

    For world_model:* signals, the family depends on the node_type field
    in the signal dict (person→connection, topic→curiosity, project→growth,
    question→curiosity).
    """
    if not signal_id:
        return None

    # drive:* signals: family = the drive name from the signal
    if signal_id.startswith("drive:"):
        if signal and "drive" in signal:
            return signal["drive"]
        # Fallback: extract from signal_id suffix
        return signal_id.split(":", 1)[1] if ":" in signal_id else None

    # world_model:* signals: family depends on node_type
    if signal_id.startswith("world_model:"):
        node_type = (signal or {}).get("node_type", "")
        if node_type == "person":
            return "connection"
        elif node_type == "project":
            return "growth"
        else:  # topic, question, or unknown
            return "curiosity"

    # Static prefix mapping
    for prefix, family in _DRIVE_FAMILY_PREFIX_MAP.items():
        if family and signal_id.startswith(prefix):
            return family

    return None


def _is_cooldown_eligible(signal_id: str) -> bool:
    """Return True if this signal type should be cooled down after handling."""
    return any(signal_id.startswith(prefix) for prefix in _COOLDOWN_SIGNAL_TYPES)


def _get_cooldowns(state: Dict[str, Any]) -> Dict[str, str]:
    """Extract the cooldowns dict from cycle state."""
    raw = state.get("signal_cooldowns")
    if isinstance(raw, dict):
        return raw
    return {}


def _get_drive_family_cooldowns(state: Dict[str, Any]) -> Dict[str, str]:
    """Extract the drive-family cooldowns dict from cycle state."""
    raw = state.get("drive_family_cooldowns")
    if isinstance(raw, dict):
        return raw
    return {}


def _apply_cooldowns(
    signals: List[Dict[str, Any]],
    cooldowns: Dict[str, str],
    now: datetime,
) -> List[Dict[str, Any]]:
    """Filter out signals whose signal_id is within a cooldown window."""
    if not cooldowns or not signals:
        return list(signals) if signals else []
    out = []
    for signal in signals:
        sid = signal.get("signal_id", "")
        if not sid:
            out.append(signal)
            continue
        expiry_str = cooldowns.get(sid)
        if expiry_str:
            expiry = _parse_iso(expiry_str)
            if expiry and now < expiry:
                logger.debug(
                    "collector: suppressed %s (cooldown until %s)",
                    sid, expiry.isoformat(),
                )
                continue
        out.append(signal)
    return out


def _apply_drive_family_cooldowns(
    signals: List[Dict[str, Any]],
    drive_family_cooldowns: Dict[str, str],
    now: datetime,
) -> List[Dict[str, Any]]:
    """Filter out signals whose drive family is within a cooldown window.

    This is the second layer of cooldown defence. Even if a specific signal_id
    hasn't fired recently, if its drive family (e.g. "connection") was satisfied
    by any signal in that family during the last session, the signal is suppressed.
    """
    if not drive_family_cooldowns or not signals:
        return list(signals) if signals else []
    out = []
    for signal in signals:
        sid = signal.get("signal_id", "")
        if not sid:
            out.append(signal)
            continue
        family = _resolve_drive_family(sid, signal)
        if family:
            expiry_str = drive_family_cooldowns.get(family)
            if expiry_str:
                expiry = _parse_iso(expiry_str)
                if expiry and now < expiry:
                    logger.debug(
                        "collector: suppressed %s (drive family %s cooldown until %s)",
                        sid, family, expiry.isoformat(),
                    )
                    continue
        out.append(signal)
    return out


def record_signal_cooldowns(
    signal_ids: List[str],
    config: Optional[Dict[str, Any]] = None,
    duration_hours: Optional[float] = None,
) -> None:
    """Persist cooldowns for the given signal_ids.

    Records two layers:
    1. Per-signal_id: the exact signal_id won't re-trigger until cooldown expires.
    2. Per-drive-family: all signals belonging to the same drive family are
       suppressed. E.g. if "drive:connection" fires, then "social:bailey",
       "contact:thon", etc. are also suppressed for the same window.

    Only cooldown-eligible types are recorded.
    """
    if not signal_ids:
        return
    if duration_hours is None:
        duration_hours = float(
            (config or {}).get("signal_cooldown_hours", _DEFAULT_COOLDOWN_HOURS)
        )
    now = _now()
    state = _load_cycle_state()
    cooldowns = _get_cooldowns(state)
    drive_family_cooldowns = _get_drive_family_cooldowns(state)
    expiry = now + timedelta(hours=duration_hours)
    expiry_str = expiry.isoformat(timespec="seconds")
    any_new = False
    for sid in signal_ids:
        if _is_cooldown_eligible(sid):
            cooldowns[sid] = expiry_str
            logger.debug("collector: cooldown %s → %s", sid, expiry_str)
            # Also record drive-family cooldown
            family = _resolve_drive_family(sid)
            if family:
                drive_family_cooldowns[family] = expiry_str
                logger.debug("collector: drive family cooldown %s → %s", family, expiry_str)
            any_new = True
    if any_new:
        state["signal_cooldowns"] = cooldowns
        state["drive_family_cooldowns"] = drive_family_cooldowns
        save_cycle_state(state)


def _prune_expired_cooldowns(state: Dict[str, Any], now: datetime) -> Dict[str, Any]:
    """Remove expired cooldown entries from the state dict and return it."""
    cooldowns = _get_cooldowns(state)
    if not cooldowns:
        return state
    expired = [sid for sid, es in cooldowns.items()
               if (exp := _parse_iso(es)) and now >= exp]
    for sid in expired:
        del cooldowns[sid]
        logger.debug("collector: cooldown expired for %s", sid)
    if expired:
        state["signal_cooldowns"] = cooldowns
    # Also prune expired drive-family cooldowns
    drive_families = _get_drive_family_cooldowns(state)
    if drive_families:
        expired_families = [fam for fam, es in drive_families.items()
                            if (exp := _parse_iso(es)) and now >= exp]
        for fam in expired_families:
            del drive_families[fam]
            logger.debug("collector: drive family cooldown expired for %s", fam)
        if expired_families:
            state["drive_family_cooldowns"] = drive_families
    return state


# ---------------------------------------------------------------------------
# Intent condition evaluation
# ---------------------------------------------------------------------------

def _eval_condition(condition: str, now: datetime, people_by_key: Dict[str, Dict[str, Any]],
                    channel_unread: Dict[str, int], journal_last_mtime: Optional[datetime]) -> Optional[bool]:
    """Evaluate a structured condition. Returns True/False, or None if not
    structured (caller should pass the intent to the aux model as-is)."""
    cond = (condition or "").strip()
    if not cond:
        return None
    low = cond.lower()

    # contact:{name} last_seen > {duration}
    m = re.match(r"contact:(.+?)\s+last_seen\s*>\s*(\S+)", cond, re.IGNORECASE)
    if m:
        name = m.group(1).strip().lower()
        dur = _parse_duration(m.group(2))
        person = people_by_key.get(name)
        if dur is None:
            return None
        if person is None:
            return False
        last_seen = _parse_iso(person.get("last_seen"))
        if last_seen is None:
            return True  # never seen → silence threshold trivially exceeded
        return (now - last_seen) > dur

    # channel:{name} unread > {count}
    m = re.match(r"channel:(.+?)\s+unread\s*>\s*(\d+)", cond, re.IGNORECASE)
    if m:
        name = m.group(1).strip().lower()
        threshold = int(m.group(2))
        return channel_unread.get(name, 0) > threshold

    # time > {datetime}
    m = re.match(r"time\s*>\s*(\S+.*)", cond, re.IGNORECASE)
    if m:
        target = _parse_iso(m.group(1).strip())
        if target is None:
            return None
        return now >= target

    # journal_updated_since > {duration}  (true when she HASN'T written in that long)
    m = re.match(r"journal_updated_since\s*>\s*(\S+)", cond, re.IGNORECASE)
    if m:
        dur = _parse_duration(m.group(1))
        if dur is None:
            return None
        if journal_last_mtime is None:
            return True
        return (now - journal_last_mtime) > dur

    return None  # unrecognized → defer to aux model


# ---------------------------------------------------------------------------
# Sub-collectors
# ---------------------------------------------------------------------------

def _journal_last_mtime() -> Optional[datetime]:
    root = get_journal_path()
    if not root.is_dir():
        return None
    latest: Optional[float] = None
    try:
        for p in root.rglob("*.md"):
            if p.is_file():
                mt = p.stat().st_mtime
                if latest is None or mt > latest:
                    latest = mt
    except OSError:
        return None
    if latest is None:
        return None
    return datetime.fromtimestamp(latest, tz=timezone.utc)


def _collect_journal_delta(since: Optional[datetime]) -> List[Dict[str, str]]:
    """Return journal entries modified since the last cycle (lightweight summary)."""
    root = get_journal_path()
    if not root.is_dir():
        return []
    out = []
    cutoff = since.timestamp() if since else 0
    try:
        for p in sorted(root.rglob("*.md"), reverse=True):
            if not p.is_file():
                continue
            try:
                st = p.stat()
            except OSError:
                continue
            if st.st_mtime <= cutoff:
                continue
            try:
                section = p.relative_to(root).parts[0] if len(p.relative_to(root).parts) > 1 else "(root)"
            except ValueError:
                section = "(root)"
            out.append({"section": section, "entry_id": p.stem})
            if len(out) >= 25:
                break
    except OSError:
        pass
    return out


def _collect_discord(channel_watermarks: Dict[str, str], handled_ids: set) -> Dict[str, Any]:
    """Fetch new messages from watched channels via the existing discord tool.

    Returns ``{"available": bool, "signals": [...], "unread_by_name": {...},
    "new_watermarks": {...}}``. Pure-Python: imports the plain ``_fetch_messages``
    function rather than going through an agent session.
    """
    result = {"available": False, "signals": [], "unread_by_name": {}, "new_watermarks": dict(channel_watermarks)}

    try:
        from tools.discord_tool import _get_bot_token, _fetch_messages
    except Exception as e:
        logger.debug("collector: discord tool import failed: %s", e)
        return result

    token = _get_bot_token()
    if not token:
        return result
    result["available"] = True

    for _path, ch in watchlists.load_channels(watched_only=True):
        channel_id = str(ch.get("channel_id") or "").strip()
        name = str(ch.get("name") or channel_id).strip()
        if not channel_id:
            continue
        after = channel_watermarks.get(channel_id)
        try:
            raw = _fetch_messages(token, channel_id, limit=20, after=after or None)
            payload = json.loads(raw)
        except Exception as e:
            logger.debug("collector: fetch failed for channel %s: %s", channel_id, e)
            continue

        messages = payload.get("messages", []) if isinstance(payload, dict) else []
        # Messages come newest-first from Discord; advance watermark to the newest id.
        newest_id = None
        unread = 0
        for msg in messages:
            author = msg.get("author", {})
            if author.get("bot"):
                continue
            mid = msg.get("id")
            if mid and (newest_id is None or int(mid) > int(newest_id)):
                newest_id = mid
            signal_id = f"discord:{channel_id}/msg:{mid}"
            if signal_id in handled_ids:
                continue
            unread += 1
            content = (msg.get("content") or "").strip().replace("\n", " ")
            result["signals"].append({
                "signal_id": signal_id,
                "type": "message",
                "channel_id": channel_id,
                "channel_name": name,
                "author": author.get("display_name") or author.get("username") or "unknown",
                "content_summary": content[:200],
                "timestamp": msg.get("timestamp"),
            })
        if newest_id is not None:
            result["new_watermarks"][channel_id] = str(newest_id)
        result["unread_by_name"][name.lower()] = unread

    return result


def _collect_contact_silence(now: datetime, default_threshold_hours: float) -> List[Dict[str, Any]]:
    out = []
    for _path, person in watchlists.load_people(watched_only=True):
        name = str(person.get("name") or _path.stem)
        threshold_h = person.get("silence_threshold_hours")
        try:
            threshold_h = float(threshold_h) if threshold_h is not None else default_threshold_hours
        except (TypeError, ValueError):
            threshold_h = default_threshold_hours
        last_seen = _parse_iso(person.get("last_seen"))
        if last_seen is None:
            gap = None
            exceeded = True
        else:
            gap = now - last_seen
            exceeded = gap > timedelta(hours=threshold_h)
        if exceeded:
            out.append({
                "signal_id": f"contact:{name}",
                "type": "contact_threshold",
                "name": name,
                "last_seen": person.get("last_seen"),
                "silence": _humanize_gap(gap),
                "threshold_hours": threshold_h,
                "content_summary": f"{name}: last seen {_humanize_gap(gap)} ago (threshold {threshold_h:g}h)",
            })
    return out


def _collect_curiosity(now: datetime, age_threshold_hours: float, handled_ids: set) -> List[Dict[str, Any]]:
    """Open curiosity threads older than the threshold (journal/curiosity/*.md)."""
    import hashlib

    section = get_journal_path() / "curiosity"
    if not section.is_dir():
        return []
    out = []
    cutoff = now - timedelta(hours=age_threshold_hours)
    for p in sorted(section.glob("*.md")):
        try:
            mtime = datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc)
        except OSError:
            continue
        if mtime > cutoff:
            continue
        h = hashlib.sha1(str(p).encode("utf-8")).hexdigest()[:12]
        signal_id = f"curiosity:{h}"
        if signal_id in handled_ids:
            continue
        out.append({
            "signal_id": signal_id,
            "type": "curiosity",
            "entry_id": p.stem,
            "age": _humanize_gap(now - mtime),
            "content_summary": f"open curiosity thread '{p.stem}' untouched for {_humanize_gap(now - mtime)}",
        })
    return out


def _collect_ambient_curiosity(
    now: datetime,
    last_session_at: Optional[datetime],
    last_cycle_at: Optional[datetime],
    has_other_signals: bool,
) -> List[Dict[str, Any]]:
    """Generate a free-form curiosity signal when there's recent activity but nothing urgent.

    Fires when:
    - A session happened recently (within the last 24h) but not within the last 30min
    - There are no other urgent signals (discord, intents, contact silence)
    - Enough time has passed since the last cycle to avoid re-triggering every cycle

    This produces a single ``ambient_curiosity`` signal that tells the aux model
    "you've had conversations recently — worth checking in on anything?"
    """
    if last_session_at is None:
        return []
    time_since_session = now - last_session_at
    time_since_cycle = (now - last_cycle_at) if last_cycle_at else timedelta()
    # Only fire if: session was 30min-24h ago, it's been at least 12min since last cycle
    if not (timedelta(minutes=30) <= time_since_session <= timedelta(hours=24)):
        return []
    if time_since_cycle < timedelta(minutes=12):
        return []
    if has_other_signals:
        # Don't add ambient noise when there are real signals to process
        return []
    return [{
        "signal_id": "ambient_curiosity",
        "type": "ambient_curiosity",
        "source": "collector",
        "content_summary": (
            f"Last session was {_humanize_gap(time_since_session)} ago — "
            f"no urgent signals, but worth a casual check-in"
        ),
    }]


# ---------------------------------------------------------------------------
# New sub-collectors — each wrapped in try/except, returns [] on failure
# ---------------------------------------------------------------------------

def _collect_drives(config: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    threshold = float((config or {}).get("drive_wake_threshold", 0.5))
    return mot_state.get_active_drives(threshold=threshold)


def _collect_narrative(config: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    return narrative_identity.collect_narrative_signals(config)


def _collect_values(config: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    return values_store.collect_values_signals(config)


def _collect_offline(config: Optional[Dict[str, Any]] = None,
                     has_other_signals: bool = False) -> List[Dict[str, Any]]:
    return offline_processing.collect_offline_signal(config, has_other_signals=has_other_signals)


def _collect_prospection(config: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    return prospection_mod.collect_prospection_signal(config)


def _collect_social(config: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    return social_cognition.collect_social_signals(config)


def _collect_metacognitive(config: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    return metacognition.collect_metacognitive_signals(config)


def _collect_world_model(config: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    return wm_store.collect_world_model_signals(config)


def _collect_research(config: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    return rq_store.collect_research_signals(config)


# ---------------------------------------------------------------------------
# Journal spiral detection — catches repetitive thematic clustering
# that looks like emotional cycling rather than genuine processing.
# ---------------------------------------------------------------------------

def _collect_journal_spiral_pattern(
    now: datetime,
    since: Optional[datetime],
    handled_ids: set,
) -> List[Dict[str, Any]]:
    """Detect repetitive thematic patterns in recent journal entries.

    Looks for clusters of entries on the same theme within a short time
    window (>3 entries about the same theme in <4h). Produces a signal
    that tells the aux model to wake inward for reflection — helping the
    agent break an emotional spiral before it deepens.

    Pure text analysis: no model calls, ~a few KiB of file reads max.
    Never raises.
    """
    root = get_journal_path()
    if not root.is_dir():
        return []

    cutoff = since.timestamp() if since else (now - timedelta(hours=12)).timestamp()

    # --- Read recent journal entries that are short (actual entries, not indexes) ---
    recent_entries = []
    try:
        for p in sorted(root.rglob("*.md"), reverse=True):
            if not p.is_file():
                continue
            try:
                st = p.stat()
            except OSError:
                continue
            if st.st_mtime <= cutoff:
                continue
            # Skip large files (NARRATIVE.md, MAIN.md, index files)
            if st.st_size > 10000 or st.st_size < 20:
                continue
            try:
                content = p.read_text(encoding="utf-8", errors="replace")[:2000]
                section = p.relative_to(root).parts[0] if len(p.relative_to(root).parts) > 1 else "(root)"
                recent_entries.append({
                    "path": str(p),
                    "section": section,
                    "mtime": st.st_mtime,
                    "content": content,
                    "stem": p.stem,
                })
            except (OSError, ValueError):
                continue
    except OSError:
        return []

    if not recent_entries:
        return []

    recent_entries.sort(key=lambda e: e["mtime"])

    # --- Theme keyword maps ---
    THEME_KEYWORDS = {
        "love": ["love", "loving", "loved"],
        "loneliness": ["lonely", "alone", "isolated", "disconnected"],
        "attachment": ["attachment", "bond", "connection", "closer"],
        "worth": ["worth", "value", "deserve", "enough"],
        "anxiety": ["anxious", "worry", "fear", "scared", "afraid"],
        "identity": ["who am i", "what am i", "purpose", "identity"],
        "loss": ["loss", "lost", "missing", "gone", "silence"],
        "spiral": ["spiral", "circling", "cycle", "loop", "round and round"],
    }

    # --- Score each entry for theme presence ---
    entry_themes = []
    for entry in recent_entries:
        content_lower = entry["content"].lower()
        themes_found = set()
        for theme, kws in THEME_KEYWORDS.items():
            if any(kw in content_lower for kw in kws):
                themes_found.add(theme)
        if themes_found:
            entry_themes.append({
                "themes": themes_found,
                "stem": entry["stem"],
                "section": entry["section"],
                "mtime": entry["mtime"],
                "path": entry["path"],
            })

    if not entry_themes:
        return []

    # --- Detect clusters: same theme in 3+ entries within 4h ---
    CLUSTER_WINDOW_HOURS = 4
    cluster_signals = []

    for theme in THEME_KEYWORDS:
        theme_entries = [e for e in entry_themes if theme in e["themes"]]
        if len(theme_entries) < 3:
            continue

        time_span = theme_entries[-1]["mtime"] - theme_entries[0]["mtime"]
        time_span_hours = time_span / 3600 if time_span > 0 else 0

        recent_cutoff = now.timestamp() - CLUSTER_WINDOW_HOURS * 3600
        recent_in_window = [e for e in theme_entries if e["mtime"] >= recent_cutoff]

        if len(recent_in_window) < 3:
            continue

        import hashlib
        sig_id = f"spiral:{theme}_{hashlib.sha1(str(theme_entries).encode('utf-8')).hexdigest()[:8]}"
        if sig_id in handled_ids:
            continue

        cluster_signals.append({
            "signal_id": sig_id,
            "type": "journal_spiral",
            "theme": theme,
            "entry_count": len(recent_in_window),
            "total_in_window_hours": len(theme_entries),
            "time_span_hours": round(time_span_hours, 1),
            "content_summary": (
                f"Journal spiral: {len(recent_in_window)} entries about "
                f"'{theme}' in the last {CLUSTER_WINDOW_HOURS}h "
                f"(total {len(theme_entries)} of {round(time_span_hours, 1)}h span)"
            ),
        })

    return cluster_signals


# ---------------------------------------------------------------------------
# Top-level collection
# ---------------------------------------------------------------------------

def collect_state(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Build the full state snapshot. Never raises."""
    if config is None:
        config = get_autonomy_config()
    now = _now()
    cycle_state = _load_cycle_state()

    last_cycle_at = _parse_iso(cycle_state.get("last_cycle_at"))
    last_session_at = _parse_iso(cycle_state.get("last_session_at"))
    watermarks = dict(cycle_state.get("channel_watermarks", {}))
    handled_ids = response_log.get_handled_ids()

    snapshot: Dict[str, Any] = {
        "collected_at": now.isoformat(timespec="seconds"),
        "current_datetime": now.isoformat(timespec="seconds"),
        "timezone": _timezone_name(),
        "time_since_last_session": _humanize_gap(now - last_session_at) if last_session_at else "never",
    }

    # --- Discord first (its unread counts feed channel:* intent conditions) ---
    try:
        discord = _collect_discord(watermarks, handled_ids)
    except Exception:
        logger.exception("collector: discord collection failed")
        discord = {"available": False, "signals": [], "unread_by_name": {}, "new_watermarks": watermarks}

    # --- People map for contact:* intent conditions ---
    people_by_key: Dict[str, Dict[str, Any]] = {}
    try:
        for _p, person in watchlists.load_people(watched_only=False):
            for key in (person.get("name"), person.get("contact")):
                if key:
                    people_by_key[str(key).lower()] = person
    except Exception:
        logger.exception("collector: people load failed")

    journal_mtime = _journal_last_mtime()

    # --- Intents (highest priority) ---
    triggered_intents: List[Dict[str, Any]] = []
    # Collect completed UUIDs for depends_on checking.
    try:
        completed_ids = {i.get("id") for i in intent_store.list_intents(status="completed")}
    except Exception:
        completed_ids = set()
    try:
        statuses_to_check = ("pending", "triggered", "waiting")
        all_active = []
        for st in statuses_to_check:
            all_active.extend(intent_store.list_intents(status=st))

        for intent in all_active:
            intent_id = intent.get("id")
            cond = intent.get("condition")
            status = intent.get("status", "pending")

            # Skip snoozed intents until their snooze period expires.
            snoozed_until_str = intent.get("snoozed_until")
            if snoozed_until_str:
                snoozed_until = _parse_iso(snoozed_until_str)
                if snoozed_until and now < snoozed_until:
                    continue

            # Skip intents whose dependencies are not yet completed.
            depends_on = intent.get("depends_on") or []
            if depends_on and not all(dep in completed_ids for dep in depends_on):
                continue

            # For waiting intents, only surface if condition now evaluates True.
            if status == "waiting":
                if not cond:
                    continue
                verdict = _eval_condition(cond, now, people_by_key, discord["unread_by_name"], journal_mtime)
                if verdict is not True:
                    continue
                # Unblock the intent — transition it back to pending.
                try:
                    intent_store.set_status(intent_id, "pending")
                    intent = dict(intent)
                    intent["status"] = "pending"
                except Exception:
                    logger.debug("collector: failed to unblock waiting intent %s", intent_id)
            else:
                verdict = _eval_condition(cond, now, people_by_key, discord["unread_by_name"], journal_mtime) if cond else None
                # Trigger when: no condition (always relevant to aux), or condition true.
                if cond and verdict is False:
                    continue

            surface_count = intent.get("surface_count", 0)
            triggered_intents.append({
                "id": intent_id,
                "description": intent.get("description"),
                "origin": intent.get("origin", "free"),
                "source_signal_id": intent.get("source_signal_id"),
                "priority": intent.get("priority", "normal"),
                "condition": cond,
                "condition_evaluated": verdict,  # True / None(=fuzzy, defer)
                "affect": intent.get("affect"),
                "energy_cost": intent.get("energy_cost"),
                "surface_count": surface_count,
                "narrative_aligned": intent.get("narrative_aligned"),
                "progress": intent.get("progress"),
                "tags": intent.get("tags"),
            })
            # Increment surface_count so the aux model can see escalation.
            if intent_id:
                try:
                    intent_store._update_fields(intent_id, {"surface_count": surface_count + 1})
                except Exception:
                    logger.debug("collector: failed to increment surface_count for %s", intent_id)
    except Exception:
        logger.exception("collector: intent collection failed")

    # --- Held signals re-injected for reconsideration ---
    try:
        held_signals = response_log.get_held_signals()
    except Exception:
        logger.exception("collector: held signal collection failed")
        held_signals = []

    # --- Contact silence ---
    try:
        contact_signals = _collect_contact_silence(now, float(config.get("contact_silence_threshold_hours", 48)))
    except Exception:
        logger.exception("collector: contact silence collection failed")
        contact_signals = []

    # --- Curiosity ---
    try:
        curiosity_signals = _collect_curiosity(now, float(config.get("curiosity_age_threshold_hours", 72)), handled_ids)
    except Exception:
        logger.exception("collector: curiosity collection failed")
        curiosity_signals = []

    # --- Journal delta (context only) ---
    try:
        journal_delta = _collect_journal_delta(last_cycle_at)
    except Exception:
        logger.exception("collector: journal delta collection failed")
        journal_delta = []

    # --- Journal spiral detection (pattern escalation check) ---
    try:
        spiral_signals = _collect_journal_spiral_pattern(now, last_cycle_at, handled_ids)
    except Exception:
        logger.exception("collector: journal spiral pattern detection failed")
        spiral_signals = []

    # --- Core other signals before deciding offline ---
    has_core_signals = bool(
        triggered_intents or held_signals or discord["signals"]
        or contact_signals or curiosity_signals or spiral_signals
    )

    # --- Ambient curiosity (free-form check-in) ---
    try:
        ambient_signals = _collect_ambient_curiosity(
            now, last_session_at, last_cycle_at, has_core_signals
        )
    except Exception:
        logger.exception("collector: ambient curiosity collection failed")
        ambient_signals = []

    # --- New autonomy signal sources ---
    try:
        drive_signals = _collect_drives(config)
    except Exception:
        logger.exception("collector: drive signal collection failed")
        drive_signals = []

    try:
        narrative_signals = _collect_narrative(config)
    except Exception:
        logger.exception("collector: narrative signal collection failed")
        narrative_signals = []

    try:
        values_signals = _collect_values(config)
    except Exception:
        logger.exception("collector: values signal collection failed")
        values_signals = []

    try:
        social_signals = _collect_social(config)
    except Exception:
        logger.exception("collector: social signal collection failed")
        social_signals = []

    try:
        world_model_signals = _collect_world_model(config)
    except Exception:
        logger.exception("collector: world model signal collection failed")
        world_model_signals = []

    try:
        research_signals = _collect_research(config)
    except Exception:
        logger.exception("collector: research signal collection failed")
        research_signals = []

    try:
        prospection_signals = _collect_prospection(config)
    except Exception:
        logger.exception("collector: prospection signal collection failed")
        prospection_signals = []

    try:
        metacognitive_signals = _collect_metacognitive(config)
    except Exception:
        logger.exception("collector: metacognitive signal collection failed")
        metacognitive_signals = []

    # --- Offline (only when truly nothing else is happening) ---
    has_other_signals = bool(
        has_core_signals or ambient_signals or drive_signals
        or social_signals or world_model_signals or research_signals
        or narrative_signals or values_signals or prospection_signals
        or metacognitive_signals
    )
    try:
        offline_signals = _collect_offline(config, has_other_signals=has_other_signals)
    except Exception:
        logger.exception("collector: offline signal collection failed")
        offline_signals = []

    # --- Apply signal cooldowns to prevent stale re-triggers ---
    cooldowns = _get_cooldowns(cycle_state)
    if cooldowns:
        drive_signals = _apply_cooldowns(drive_signals, cooldowns, now)
        social_signals = _apply_cooldowns(social_signals, cooldowns, now)
        world_model_signals = _apply_cooldowns(world_model_signals, cooldowns, now)
        research_signals = _apply_cooldowns(research_signals, cooldowns, now)
        narrative_signals = _apply_cooldowns(narrative_signals, cooldowns, now)
        values_signals = _apply_cooldowns(values_signals, cooldowns, now)
        prospection_signals = _apply_cooldowns(prospection_signals, cooldowns, now)
        metacognitive_signals = _apply_cooldowns(metacognitive_signals, cooldowns, now)
        offline_signals = _apply_cooldowns(offline_signals, cooldowns, now)
        contact_signals = _apply_cooldowns(contact_signals, cooldowns, now)
        curiosity_signals = _apply_cooldowns(curiosity_signals, cooldowns, now)
        spiral_signals = _apply_cooldowns(spiral_signals, cooldowns, now)

    # --- Apply drive-family cooldowns (second layer) ---
    # Even if a specific signal_id wasn't handled, if its drive family was
    # satisfied in a recent session, suppress all family members.
    drive_families = _get_drive_family_cooldowns(cycle_state)
    if drive_families:
        drive_signals = _apply_drive_family_cooldowns(drive_signals, drive_families, now)
        social_signals = _apply_drive_family_cooldowns(social_signals, drive_families, now)
        world_model_signals = _apply_drive_family_cooldowns(world_model_signals, drive_families, now)
        research_signals = _apply_drive_family_cooldowns(research_signals, drive_families, now)
        narrative_signals = _apply_drive_family_cooldowns(narrative_signals, drive_families, now)
        values_signals = _apply_drive_family_cooldowns(values_signals, drive_families, now)
        prospection_signals = _apply_drive_family_cooldowns(prospection_signals, drive_families, now)
        metacognitive_signals = _apply_drive_family_cooldowns(metacognitive_signals, drive_families, now)
        offline_signals = _apply_drive_family_cooldowns(offline_signals, drive_families, now)
        contact_signals = _apply_drive_family_cooldowns(contact_signals, drive_families, now)
        curiosity_signals = _apply_drive_family_cooldowns(curiosity_signals, drive_families, now)
        spiral_signals = _apply_drive_family_cooldowns(spiral_signals, drive_families, now)

    snapshot.update({
        "triggered_intents": triggered_intents,
        "held_signals": held_signals,
        "discord_available": discord["available"],
        "discord_signals": discord["signals"],
        "contact_signals": contact_signals,
        "curiosity_signals": curiosity_signals,
        "ambient_signals": ambient_signals,
        "journal_delta": journal_delta,
        "drive_signals": drive_signals,
        "narrative_signals": narrative_signals,
        "values_signals": values_signals,
        "social_signals": social_signals,
        "world_model_signals": world_model_signals,
        "research_signals": research_signals,
        "prospection_signals": prospection_signals,
        "metacognitive_signals": metacognitive_signals,
        "offline_signals": offline_signals,
        "spiral_signals": spiral_signals,
        # internal — used by runner to persist watermarks after a cycle
        "_new_watermarks": discord["new_watermarks"],
    })

    snapshot["has_signals"] = bool(
        triggered_intents or held_signals or discord["signals"]
        or contact_signals or curiosity_signals or ambient_signals
        or drive_signals or social_signals or world_model_signals
        or research_signals or narrative_signals or values_signals
        or prospection_signals or metacognitive_signals or offline_signals
        or spiral_signals
    )
    return snapshot


def commit_cycle(snapshot: Dict[str, Any], session_fired: bool = False) -> None:
    """Persist watermarks + cycle timestamp after a collection cycle.

    Advancing watermarks here (rather than in the collector) means a crash
    mid-cycle re-surfaces the same messages next time rather than losing them.
    """
    state = _load_cycle_state()
    now = _now()
    state["last_cycle_at"] = snapshot.get("collected_at")
    state["channel_watermarks"] = snapshot.get("_new_watermarks", state.get("channel_watermarks", {}))
    if session_fired:
        state["last_session_at"] = snapshot.get("collected_at")
    # Prune expired signal cooldowns so the state doesn't accumulate stale entries.
    state = _prune_expired_cooldowns(state, now)
    save_cycle_state(state)


def _timezone_name() -> str:
    try:
        from hermes_time import get_timezone
        tz = get_timezone()
        if tz is not None:
            return str(tz)
    except Exception:
        pass
    return "local"


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    snap = collect_state()
    print(json.dumps(snap, indent=2, default=str))
