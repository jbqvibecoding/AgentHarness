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
    # hyperresearch-inspired enhancements (standard default):
    "use_vault": True,               # mirror fetched pages into an FTS vault
    "build_contradiction_graph": True,  # conflict_check builds graph + loci
    "loci_max": 3,                   # max contested loci to deep-dive
    "num_critics": 4,                # adversarial critics (dialectic/depth/width/instruction)
    "patch_revision": True,          # revise via surgical edit hunks, not full regen
    "enable_polish": True,           # terminal filler/scaffold cleanup
    # Council peer review (anonymized cross-ranking among member models).
    # Off by default: it adds one LLM call per member, so the default
    # council path keeps its current cost. Never enabled by a depth preset
    # — only an explicit --peer-review / tool argument turns it on.
    "peer_review": False,
    "peer_review_excerpt_chars": 6000,
    "peer_review_parallel": 3,
    # Citation integrity (FrontierAgent's deterministic citation contract).
    # ``citation_contract`` pins the writer to a numbered whitelist and
    # makes the system, not the model, render the References block.
    # ``numeric_grounding`` additionally classifies every specific number
    # in the report against the text behind its citation.
    # ``citation_repair_rounds`` is the repair budget: one round of
    # marker-only repair, then whatever remains is disclosed in the
    # report's Verification appendix rather than blocking delivery.
    "citation_contract": True,
    "numeric_grounding": True,
    "citation_repair_rounds": 1,
    # Cumulative token budget for the whole run, shared by every sub-agent
    # branch and node call. 0 disables budgeting entirely. A branch is also
    # held to max_run_tokens / max_parallel_subagents so one runaway
    # researcher cannot drain the pool before its siblings spend anything.
    "max_run_tokens": 2_000_000,
    "budget_warn_ratio": 0.8,
    # How the research node gathers evidence.
    #   "fanout" — one researcher per pending sub-question (predictable,
    #              cheapest, and what every earlier run used).
    #   "swarm"  — a coordinator opens and assigns researchers itself, so
    #              effort follows what the question turns out to need.
    # The seven-role DAG is identical either way; only this node changes.
    "research_mode": "fanout",
    "coordinator_max_turns": 30,
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
        # Lean path — keep quick smoke fast and cheap.
        "build_contradiction_graph": False,
        "loci_max": 0,
        "num_critics": 1,
        "patch_revision": False,
        "enable_polish": False,
        # The contract itself is free (prompt-only) so it stays on, but
        # the repair round costs an LLM call — quick discloses instead.
        "citation_repair_rounds": 0,
        "max_run_tokens": 400_000,
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
        "loci_max": 6,
        "max_run_tokens": 6_000_000,
        # Deep runs are where a fixed one-per-sub-question fan-out is most
        # obviously wrong: the questions are broad enough that effort
        # genuinely should be uneven.
        "research_mode": "swarm",
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
