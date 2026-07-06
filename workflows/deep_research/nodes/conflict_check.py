"""``conflict_check`` node — contradiction + gap audit (role ``dr_conflict_checker``).

Single LLM call over the compact claim table (never raw cards). Output
follows deep-searcher's REFLECT contract: an empty ``gap_questions`` list
signals convergence; the routing condition then proceeds to drafting.
Parse/LLM failure fails open (no conflicts, no gaps) so the pipeline
always reaches a report.
"""

from __future__ import annotations

import logging
from typing import Any

from agent_harness.models.node_context import NodeContext

from workflows.deep_research.nodes.common import call_role_llm, claim_table
from workflows.deep_research.prompts import (
    CONFLICT_SYSTEM,
    build_conflict_prompt,
    extract_json_block,
)

logger = logging.getLogger(__name__)

_MAX_GAPS = 3


async def conflict_check_node(
    state: dict[str, Any], ctx: NodeContext,
) -> dict[str, Any]:
    table = claim_table(state)
    sub_questions = [
        {
            "id": s.get("id"),
            "question": s.get("question"),
            "status": s.get("status"),
        }
        for s in state.get("sub_questions") or []
    ]

    conflicts: list[dict[str, Any]] = []
    gap_questions: list[dict[str, Any]] = []
    errors: list[str] = []
    try:
        raw = await call_role_llm(
            role_id="dr_conflict_checker",
            state=state,
            system_prompt=CONFLICT_SYSTEM,
            user_prompt=build_conflict_prompt(
                state.get("research_brief", ""), sub_questions, table,
            ),
        )
        data = extract_json_block(raw)
        if isinstance(data, dict):
            for item in data.get("conflicts") or []:
                if isinstance(item, dict) and item.get("description"):
                    conflicts.append({
                        "card_ids": [
                            str(c) for c in item.get("card_ids") or []
                        ],
                        "description": str(item["description"])[:600],
                        "severity": (
                            "high"
                            if str(item.get("severity", "")).lower() == "high"
                            else "low"
                        ),
                        "resolution_hint": str(
                            item.get("resolution_hint", ""),
                        )[:300],
                    })
            for item in (data.get("gap_questions") or [])[:_MAX_GAPS]:
                if isinstance(item, dict) and str(
                    item.get("question", ""),
                ).strip():
                    gap_questions.append({
                        "question": str(item["question"]).strip(),
                        "rationale": str(item.get("rationale", ""))[:300],
                    })
                elif isinstance(item, str) and item.strip():
                    gap_questions.append(
                        {"question": item.strip(), "rationale": ""},
                    )
        else:
            errors.append("conflict_check: unparseable output — failing open")
    except RuntimeError as exc:
        # Fail open: an unavailable auditor must not block the report.
        errors.append(f"conflict_check: LLM failed ({exc}) — failing open")

    logger.info(
        "deep_research conflict_check (task=%s): %d conflicts, %d gaps",
        ctx.task_id, len(conflicts), len(gap_questions),
    )
    return {
        "conflicts": conflicts,
        "gap_questions": gap_questions,
        "errors": errors,
        "current_phase": "conflict_check",
    }


__all__ = ["conflict_check_node"]
