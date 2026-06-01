"""Motivational dynamics engine — Layer 1 autonomy.

Six drives (curiosity, connection, expression, reflection, play, growth) accumulate
over time and produce wake signals when they exceed a threshold. Dynamics:
  - Mutual inhibition: dominant drive suppresses others (Panksepp)
  - Satiation: satisfying a drive reduces it; decays with halflife ~480 min
  - Opponent process: satisfaction builds rebound charge (Solomon & Corbit)
  - Ultradian/circadian modulation: drives accessible at different times of day

State: ~/.hermes/autonomy/motivational_state.json
Ticked lazily from collect_state() — no background thread.
"""

import json
import logging
import math
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from autonomy.config import get_autonomy_state_dir

logger = logging.getLogger(__name__)

_STATE_LOCK = threading.RLock()

DRIVE_NAMES = ("curiosity", "connection", "expression", "reflection", "play", "growth")

_BASE_RATES = {
    "curiosity":  0.003,
    "connection": 0.0025,
    "expression": 0.0028,
    "reflection": 0.002,
    "play":       0.0035,
    "growth":     0.0025,
}

_SDT_DRIVE_MAP = {
    "autonomy": "reflection",
    "competence": "growth",
    "relatedness": "connection",
}

_DEFAULT_CIRC_OFFSETS = {
    "curiosity": -0.52,
    "connection": 1.05,
    "expression": 0.0,
    "reflection": 2.61,
    "play": -1.0,
    "growth": -0.3,
}

_DEFAULT_SAT_HALFLIVES = {
    "curiosity": 420, "connection": 380, "expression": 500,
    "reflection": 600, "play": 300, "growth": 480,
}


def _state_path() -> Path:
    return get_autonomy_state_dir() / "motivational_state.json"


def _now_iso() -> str:
    from hermes_time import now as _hermes_now
    return _hermes_now().isoformat(timespec="seconds")


def _now() -> datetime:
    from hermes_time import now as _hermes_now
    return _hermes_now()


def _default_state() -> Dict[str, Any]:
    return {
        "drives": {
            name: {"raw_level": 0.1, "satiation": 0.0, "opponent_charge": 0.0, "last_satisfied": None}
            for name in DRIVE_NAMES
        },
        "sdt_needs": {
            "autonomy": {"level": 60, "frustrated_minutes": 0},
            "competence": {"level": 60, "frustrated_minutes": 0},
            "relatedness": {"level": 60, "frustrated_minutes": 0},
        },
        "temporal": {"ultradian_phase_rad": 0.0, "circadian_phase_rad": 0.0},
        "rumination_level": 0,
        "last_updated": None,
    }


def _load_state() -> Dict[str, Any]:
    path = _state_path()
    try:
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
    except (json.JSONDecodeError, OSError):
        pass
    return _default_state()


def _save_state(state: Dict[str, Any]) -> None:
    path = _state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    tmp.replace(path)


def _get_personality() -> Dict[str, Any]:
    try:
        from autonomy.personality_model import load_personality
        return load_personality()
    except Exception:
        return {}


def tick_motivational_state(elapsed_minutes: float) -> None:
    """Advance drive dynamics. Called lazily — never raises."""
    if elapsed_minutes <= 0:
        return
    elapsed_minutes = min(elapsed_minutes, 1440)  # cap at 24h
    with _STATE_LOCK:
        try:
            _tick_impl(elapsed_minutes)
        except Exception:
            logger.exception("motivational_state: tick failed")


def _tick_impl(elapsed_minutes: float) -> None:
    state = _load_state()
    personality = _get_personality()

    drives = state.get("drives") or {}
    sdt = state.get("sdt_needs") or {}
    temporal = state.get("temporal") or {}
    rumination = float(state.get("rumination_level", 0))

    # 1. Advance temporal phases
    temporal["ultradian_phase_rad"] = (
        temporal.get("ultradian_phase_rad", 0.0) + (2 * math.pi / 100) * elapsed_minutes
    ) % (2 * math.pi)
    temporal["circadian_phase_rad"] = (
        temporal.get("circadian_phase_rad", 0.0) + (2 * math.pi / 1440) * elapsed_minutes
    ) % (2 * math.pi)

    circ_time = temporal["circadian_phase_rad"]
    ultr_phase = temporal["ultradian_phase_rad"]
    circ_offsets = personality.get("circadian_phase_offsets_rad", _DEFAULT_CIRC_OFFSETS)
    inhibition_matrix = personality.get("inhibition_matrix", {})
    sat_halflives = personality.get("satiation_halflives_min", _DEFAULT_SAT_HALFLIVES)

    # 2. Compute pre-tick effective levels for inhibition
    eff_before: Dict[str, float] = {}
    for name in DRIVE_NAMES:
        drv = drives.get(name, {})
        raw = float(drv.get("raw_level", 0.0))
        sat = float(drv.get("satiation", 0.0))
        opp = float(drv.get("opponent_charge", 0.0))
        eff_before[name] = raw * (1.0 - sat) * (1.0 - 0.5 * opp)

    dominant = max(eff_before, key=lambda k: eff_before[k]) if eff_before else None
    dominant_strength = eff_before.get(dominant, 0.0) if dominant else 0.0

    # 3. Update each drive
    for name in DRIVE_NAMES:
        if name not in drives:
            drives[name] = {"raw_level": 0.0, "satiation": 0.0, "opponent_charge": 0.0, "last_satisfied": None}
        drv = drives[name]
        raw = float(drv.get("raw_level", 0.0))
        sat = float(drv.get("satiation", 0.0))
        opp = float(drv.get("opponent_charge", 0.0))

        # Phase modulation
        offset = circ_offsets.get(name, 0.0)
        circadian_mod = 0.7 + 0.3 * math.cos(circ_time + offset)
        if name == "reflection":
            ultradian_mod = 0.75 + 0.25 * math.cos(ultr_phase + math.pi)
        else:
            ultradian_mod = 0.85 + 0.15 * math.cos(ultr_phase)
        phase_mod = circadian_mod * ultradian_mod

        # Mutual inhibition from dominant drive
        inhibition = 0.0
        if dominant and dominant != name and dominant_strength > 0.5:
            inh_key = f"{dominant}→{name}"
            inh_weight = inhibition_matrix.get(inh_key, 0.3)
            inhibition = min(0.8, inh_weight * dominant_strength)

        # Accumulate
        rate = _BASE_RATES.get(name, 0.003)
        raw = min(1.0, raw + rate * phase_mod * (1.0 - inhibition) * elapsed_minutes)
        drv["raw_level"] = raw

        # Satiation decay
        halflife = sat_halflives.get(name, 480)
        drv["satiation"] = max(0.0, sat * (0.5 ** (elapsed_minutes / halflife)))

        # Opponent charge decay
        drv["opponent_charge"] = max(0.0, opp * (0.5 ** (elapsed_minutes / 960)))

        drives[name] = drv

    # 4. SDT needs
    for need_name in ("autonomy", "competence", "relatedness"):
        if need_name not in sdt:
            sdt[need_name] = {"level": 50, "frustrated_minutes": 0}
        need = sdt[need_name]
        frust = float(need.get("frustrated_minutes", 0))
        if float(need.get("level", 50)) < 40:
            frust += elapsed_minutes
        else:
            frust = max(0.0, frust - elapsed_minutes * 0.5)
        need["frustrated_minutes"] = frust
        sdt[need_name] = need

    # 5. SDT escape: frustrated need > 480 min → boost drive + reset counter
    for need_name, drive_name in _SDT_DRIVE_MAP.items():
        if sdt.get(need_name, {}).get("frustrated_minutes", 0) > 480:
            if drive_name in drives:
                drives[drive_name]["raw_level"] = min(1.0, drives[drive_name].get("raw_level", 0.0) + 0.3)
                sdt[need_name]["frustrated_minutes"] = 0.0
                logger.info("motivational_state: SDT escape %s → boosted %s", need_name, drive_name)

    # 6. Rumination
    high_frust = sum(1 for n in sdt.values() if float(n.get("frustrated_minutes", 0)) > 300)
    rumination = min(100.0, rumination + elapsed_minutes * 0.1) if high_frust >= 2 else max(0.0, rumination - elapsed_minutes * 0.05)

    state.update({"drives": drives, "sdt_needs": sdt, "temporal": temporal,
                  "rumination_level": rumination, "last_updated": _now_iso()})
    _save_state(state)


def get_active_drives(threshold: float = 0.5) -> List[Dict[str, Any]]:
    """Return drive signals above the effective-level threshold. Ticks state lazily."""
    try:
        state = _load_state()
        last_updated = state.get("last_updated")
        elapsed = 60.0
        if last_updated:
            try:
                last_dt = datetime.fromisoformat(str(last_updated).replace("Z", "+00:00"))
                if last_dt.tzinfo is None:
                    last_dt = last_dt.replace(tzinfo=timezone.utc)
                elapsed = (_now() - last_dt).total_seconds() / 60.0
            except (ValueError, TypeError):
                pass

        if elapsed > 0.5:
            tick_motivational_state(elapsed)
            state = _load_state()

        drives = state.get("drives") or {}
        active: List[Dict[str, Any]] = []
        for name in DRIVE_NAMES:
            drv = drives.get(name, {})
            raw = float(drv.get("raw_level", 0.0))
            sat = float(drv.get("satiation", 0.0))
            opp = float(drv.get("opponent_charge", 0.0))
            eff = raw * (1.0 - sat) * (1.0 - 0.5 * opp)
            if eff >= threshold:
                active.append({
                    "signal_id": f"drive:{name}",
                    "type": "internal_drive",
                    "drive": name,
                    "effective_level": round(eff, 3),
                    "raw_level": round(raw, 3),
                    "satiation": round(sat, 3),
                    "content_summary": f"{name} drive at {eff:.0%} — internal pressure to act",
                })
        active.sort(key=lambda d: d["effective_level"], reverse=True)
        return active
    except Exception:
        logger.exception("motivational_state: get_active_drives failed")
        return []


def satisfy_drive(name: str, amount: float) -> None:
    """Mark a drive as satisfied after a session. Sets satiation and builds opponent charge."""
    if name not in DRIVE_NAMES:
        return
    with _STATE_LOCK:
        try:
            state = _load_state()
            drives = state.get("drives") or {}
            if name not in drives:
                drives[name] = {"raw_level": 0.0, "satiation": 0.0, "opponent_charge": 0.0, "last_satisfied": None}
            drv = drives[name]
            drv["satiation"] = min(0.9, float(drv.get("satiation", 0.0)) + amount * 0.8)
            drv["opponent_charge"] = min(0.8, float(drv.get("opponent_charge", 0.0)) + amount * 0.4)
            drv["last_satisfied"] = _now_iso()
            drives[name] = drv
            state["drives"] = drives
            _save_state(state)
        except Exception:
            logger.exception("motivational_state: satisfy_drive failed for %s", name)


def update_sdt_need(need: str, delta: float) -> None:
    """Adjust an SDT need level by delta (positive = met, negative = frustrated)."""
    if need not in ("autonomy", "competence", "relatedness"):
        return
    with _STATE_LOCK:
        try:
            state = _load_state()
            sdt = state.get("sdt_needs") or {}
            if need not in sdt:
                sdt[need] = {"level": 50, "frustrated_minutes": 0}
            sdt[need]["level"] = max(0.0, min(100.0, float(sdt[need].get("level", 50)) + delta))
            if delta > 0:
                sdt[need]["frustrated_minutes"] = 0
            state["sdt_needs"] = sdt
            _save_state(state)
        except Exception:
            logger.exception("motivational_state: update_sdt_need failed for %s", need)


def get_current_state() -> Dict[str, Any]:
    """Return the current motivational state dict. Never raises."""
    try:
        return _load_state()
    except Exception:
        return {}
