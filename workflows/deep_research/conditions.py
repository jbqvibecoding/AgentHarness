"""Routing conditions for the ``deep_research`` DAG.

Contract (``DynamicGraphBuilder``): all transitions out of a node share
ONE condition function, and the route map is keyed by the literal
``to_phase`` strings — so these functions must return the target node id
exactly as written in ``spec.py``.

Both functions are pure: the counters they read (``research_iteration``,
``revision_count``) are incremented inside the nodes.
"""

from __future__ import annotations

from typing import Any

from workflows.deep_research.config import get_cfg


def route_after_conflict_check(state: dict[str, Any]) -> str:
    """``research_fanout`` while gaps remain and iterations are left, else ``draft``."""
    gap_questions = state.get("gap_questions") or []
    iterations = int(state.get("research_iteration", 0))
    max_iterations = int(get_cfg(state, "max_research_iterations"))
    if gap_questions and iterations < max_iterations:
        return "research_fanout"
    return "draft"


def route_after_review(state: dict[str, Any]) -> str:
    """``draft`` while the reviewer demands revision within budget, else ``final_verify``."""
    verdict = state.get("review_verdict", "approve")
    revisions = int(state.get("revision_count", 0))
    max_revisions = int(get_cfg(state, "max_revisions"))
    if verdict == "revise" and revisions <= max_revisions:
        return "draft"
    return "final_verify"


__all__ = ["route_after_conflict_check", "route_after_review"]
