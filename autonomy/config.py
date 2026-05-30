"""Config + path helpers for the autonomy layer.

Single source of truth for:
  - the journal path (config key ``journal.path``, default ~/.hermes/journal)
  - the ``autonomy`` config block (intervals, thresholds, rate limits)
  - the autonomy state directory (~/.hermes/autonomy)

All journal-reading code (collector, intent/journal tools, contact tracker)
resolves the journal path from here rather than hardcoding ~/.hermes/journal,
so an operator can relocate it via config without touching code.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict

from hermes_constants import get_config_path, get_hermes_home

logger = logging.getLogger(__name__)

_DEFAULT_AUTONOMY: Dict[str, Any] = {
    "enabled": False,
    "aux_interval_minutes": 7,
    "min_tokens_between_full_sessions": 10000,
    "contact_silence_threshold_hours": 48,
    "curiosity_age_threshold_hours": 72,
    "hold_reevaluate_hours": 6,
}


def _load_config() -> Dict[str, Any]:
    """Read config.yaml fresh. Returns {} on any failure (never raises)."""
    try:
        import yaml

        config_path = get_config_path()
        if config_path.exists():
            with open(config_path, encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
    except Exception as e:
        logger.debug("autonomy: failed to read config.yaml: %s", e)
    return {}


def get_journal_path() -> Path:
    """Return the configured journal directory (``journal.path``).

    Default: ``<HERMES_HOME>/journal``. Expands ``~`` and env vars. The
    directory is NOT created here — writers (journal/intent tools) create it
    lazily so read-only callers don't litter the filesystem.
    """
    cfg = _load_config()
    journal_cfg = cfg.get("journal")
    raw = ""
    if isinstance(journal_cfg, dict):
        raw = str(journal_cfg.get("path") or "").strip()
    elif isinstance(journal_cfg, str):
        raw = journal_cfg.strip()

    if raw:
        return Path(os.path.expandvars(os.path.expanduser(raw)))
    return get_hermes_home() / "journal"


def get_autonomy_config() -> Dict[str, Any]:
    """Return the ``autonomy`` config block merged over built-in defaults."""
    cfg = _load_config()
    merged = dict(_DEFAULT_AUTONOMY)
    block = cfg.get("autonomy")
    if isinstance(block, dict):
        for key, value in block.items():
            if value is not None:
                merged[key] = value
    return merged


def get_aux_model_config() -> Dict[str, Any]:
    """Return the ``auxiliary.autonomy`` provider/model block.

    Empty/missing values mean "auto-detect" — ``auxiliary_client.call_llm``
    handles the fallback chain.
    """
    cfg = _load_config()
    aux = cfg.get("auxiliary")
    if isinstance(aux, dict):
        block = aux.get("autonomy")
        if isinstance(block, dict):
            return block
    return {}


def get_autonomy_state_dir() -> Path:
    """Return ``<HERMES_HOME>/autonomy`` (created on demand by writers)."""
    return get_hermes_home() / "autonomy"
