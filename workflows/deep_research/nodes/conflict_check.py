"""``conflict_check`` node — contradiction + coverage audit (role ``dr_conflict_checker``).

Two paths, selected by the ``build_contradiction_graph`` depth knob:

* Full (standard/deep): the hyperresearch contradiction-graph + loci
  method — cluster contested claims, extract consensus, score loci on
  four dimensions with source budgets, and derive the re-research
  ``gap_questions`` from the budgeted loci (each carrying a
  ``flavor="locus"`` so researchers give it a committed position).
* Lean (quick): the original single-shot conflicts + gap-questions
  (deep-searcher REFLECT contract).

Both fail open (no conflicts/gaps) so the pipeline always reaches a
report.
"""

from __future__ import annotations

import logging
from typing import Any

from agent_harness.models.node_context import NodeContext

from workflows.deep_research.config import get_cfg
from workflows.deep_research.hyper_prompts import (
    CONFLICT_GRAPH_SYSTEM,
    build_conflict_graph_prompt,
)
from workflows.deep_research.nodes.common import call_role_llm, claim_table
from workflows.deep_research.prompts import (
    CONFLICT_SYSTEM,
    build_conflict_prompt,
    extract_json_block,
)

logger = logging.getLogger(__name__)

_MAX_GAPS = 3


def _sub_question_view(state: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {"id": s.get("id"), "question": s.get("question"),
         "status": s.get("status")}
        for s in state.get("sub_questions") or []
    ]


async def _graph_path(
    state: dict[str, Any], ctx: NodeContext, table: list[dict[str, Any]],
) -> dict[str, Any]:
    """hyperresearch contradiction-graph + loci."""
    loci_max = int(get_cfg(state, "loci_max"))
    graph: list[dict[str, Any]] = []
    consensus: list[dict[str, Any]] = []
    loci: list[dict[str, Any]] = []
    errors: list[str] = []
    try:
        raw = await call_role_llm(
            role_id="dr_conflict_checker",
            state=state,
            system_prompt=CONFLICT_GRAPH_SYSTEM.replace(
                "{loci_max}", str(loci_max),
            ),
            user_prompt=build_conflict_graph_prompt(
                state.get("research_brief", ""),
                _sub_question_view(state), table, loci_max,
            ),
        )
        data = extract_json_block(raw)
        if isinstance(data, dict):
            graph = [c for c in (data.get("contradiction_graph") or [])
                     if isinstance(c, dict) and c.get("fight")][:12]
            consensus = [c for c in (data.get("consensus_claims") or [])
                         if isinstance(c, dict) and c.get("claim")][:20]
            for item in (data.get("loci") or []):
                if not isinstance(item, dict) or not str(
                    item.get("question", ""),
                ).strip():
                    continue
                try:
                    budget = int(item.get("source_budget", 0))
                except (TypeError, ValueError):
                    budget = 0
                loci.append({
                    "id": str(item.get("id", f"L{len(loci) + 1}")),
                    "question": str(item["question"]).strip(),
                    "flavor": str(item.get("flavor", "factual")),
                    "scores": item.get("scores", {}),
                    "composite": item.get("composite", 0),
                    "source_budget": budget,
                    "rationale": str(item.get("rationale", ""))[:300],
                })
            loci = loci[:loci_max]
        else:
            errors.append("conflict_check: unparseable graph output — failing open")
    except RuntimeError as exc:
        errors.append(f"conflict_check: graph LLM failed ({exc}) — failing open")

    # Budgeted loci become the re-research targets (committed-position dives).
    gap_questions = [
        {
            "question": lo["question"],
            "rationale": lo.get("rationale", ""),
            "flavor": "locus",
            "source_budget": lo["source_budget"],
        }
        for lo in loci if lo["source_budget"] > 0
    ]
    # Conflicts view (for draft/verify) derived from the graph clusters.
    conflicts = [
        {
            "card_ids": (c.get("side_a", {}).get("claims", [])
                         + c.get("side_b", {}).get("claims", [])),
            "description": c.get("fight", ""),
            "severity": ("high" if str(c.get("decision_relevance", "")).lower()
                         == "high" else "low"),
            "resolution_hint": c.get("evidence_quality_delta", ""),
        }
        for c in graph
    ]

    logger.info(
        "deep_research conflict_check/graph (task=%s): %d clusters, "
        "%d consensus, %d loci (%d budgeted)",
        ctx.task_id, len(graph), len(consensus), len(loci), len(gap_questions),
    )
    return {
        "contradiction_graph": graph,
        "consensus_claims": consensus,
        "loci": loci,
        "conflicts": conflicts,
        "gap_questions": gap_questions,
        "errors": errors,
        "current_phase": "conflict_check",
    }


async def _lean_path(
    state: dict[str, Any], ctx: NodeContext, table: list[dict[str, Any]],
) -> dict[str, Any]:
    """Original single-shot conflicts + gap questions (quick depth)."""
    conflicts: list[dict[str, Any]] = []
    gap_questions: list[dict[str, Any]] = []
    errors: list[str] = []
    try:
        raw = await call_role_llm(
            role_id="dr_conflict_checker",
            state=state,
            system_prompt=CONFLICT_SYSTEM,
            user_prompt=build_conflict_prompt(
                state.get("research_brief", ""),
                _sub_question_view(state), table,
            ),
        )
        data = extract_json_block(raw)
        if isinstance(data, dict):
            for item in data.get("conflicts") or []:
                if isinstance(item, dict) and item.get("description"):
                    conflicts.append({
                        "card_ids": [str(c) for c in item.get("card_ids") or []],
                        "description": str(item["description"])[:600],
                        "severity": ("high" if str(item.get("severity", "")).lower()
                                     == "high" else "low"),
                        "resolution_hint": str(item.get("resolution_hint", ""))[:300],
                    })
            for item in (data.get("gap_questions") or [])[:_MAX_GAPS]:
                if isinstance(item, dict) and str(item.get("question", "")).strip():
                    gap_questions.append({
                        "question": str(item["question"]).strip(),
                        "rationale": str(item.get("rationale", ""))[:300],
                    })
                elif isinstance(item, str) and item.strip():
                    gap_questions.append({"question": item.strip(), "rationale": ""})
        else:
            errors.append("conflict_check: unparseable output — failing open")
    except RuntimeError as exc:
        errors.append(f"conflict_check: LLM failed ({exc}) — failing open")

    logger.info(
        "deep_research conflict_check/lean (task=%s): %d conflicts, %d gaps",
        ctx.task_id, len(conflicts), len(gap_questions),
    )
    return {
        "contradiction_graph": [],
        "consensus_claims": [],
        "loci": [],
        "conflicts": conflicts,
        "gap_questions": gap_questions,
        "errors": errors,
        "current_phase": "conflict_check",
    }


async def conflict_check_node(
    state: dict[str, Any], ctx: NodeContext,
) -> dict[str, Any]:
    table = claim_table(state)
    if bool(get_cfg(state, "build_contradiction_graph")):
        return await _graph_path(state, ctx, table)
    return await _lean_path(state, ctx, table)


__all__ = ["conflict_check_node"]
