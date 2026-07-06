"""Behavioral knobs for the ``deep_research`` pipeline.

Resolution order for every knob (``get_cfg``):

1. ``state["metadata"]["deep_research"][key]`` — explicit per-run override
   (set by the CLI runner or the Hermes tool).
2. The depth preset selected by ``metadata["deep_research"]["depth"]``
   (``quick`` / ``standard`` / ``deep``, default ``standard``).
3. The hardcoded default in ``_DEFAULTS``.

Model/sampling configuration is NOT here — that lives in
``profiles/*.yaml`` (see ``profile.py``); this module only owns loop
bounds, parallelism, and budgets.
"""

from __future__ import annotations

from typing import Any

_DEFAULTS: dict[str, Any] = {
    "max_sub_questions": 5,
    "max_research_iterations": 2,
    "max_revisions": 1,
    "max_parallel_subagents": 3,
    "researcher_max_turns": 12,
    "fact_checker_max_turns": 8,
    "max_cards_to_check": 8,
    "cards_per_checker": 4,
    "subagent_timeout_s": 900,
    "verifier_spot_check": False,
    "verifier_spot_check_turns": 4,
}

DEPTH_PRESETS: dict[str, dict[str, Any]] = {
    "quick": {
        "max_sub_questions": 3,
        "max_research_iterations": 1,
        "max_revisions": 0,
        "researcher_max_turns": 6,
        "fact_checker_max_turns": 4,
        "max_cards_to_check": 4,
        "subagent_timeout_s": 480,
    },
    "standard": {},
    "deep": {
        "max_sub_questions": 6,
        "max_research_iterations": 3,
        "max_revisions": 2,
        "researcher_max_turns": 20,
        "fact_checker_max_turns": 12,
        "max_cards_to_check": 16,
        "subagent_timeout_s": 1500,
        "verifier_spot_check": True,
    },
}


def _dr_meta(state: dict[str, Any]) -> dict[str, Any]:
    meta = state.get("metadata") or {}
    dr = meta.get("deep_research") or {}
    return dr if isinstance(dr, dict) else {}


def get_depth(state: dict[str, Any]) -> str:
    depth = str(_dr_meta(state).get("depth", "standard")).lower()
    return depth if depth in DEPTH_PRESETS else "standard"


def get_cfg(state: dict[str, Any], key: str) -> Any:
    """Resolve a behavioral knob: explicit override > depth preset > default."""
    dr = _dr_meta(state)
    if key in dr:
        return dr[key]
    preset = DEPTH_PRESETS[get_depth(state)]
    if key in preset:
        return preset[key]
    return _DEFAULTS[key]


__all__ = ["DEPTH_PRESETS", "get_cfg", "get_depth"]
