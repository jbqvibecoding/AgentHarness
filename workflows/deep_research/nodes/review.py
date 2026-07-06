"""``review`` node — critic pass over the draft (role ``dr_reviewer``).

Hermes autoreason Critic contract: find problems only, no rewritten text;
an explicit approve is a valid outcome. Unparseable/failed review degrades
to approve so the loop can never wedge.
"""

from __future__ import annotations

import logging
from typing import Any

from agent_harness.models.node_context import NodeContext

from workflows.deep_research.nodes.common import call_role_llm, claim_table
from workflows.deep_research.prompts import (
    REVIEWER_SYSTEM,
    build_review_prompt,
    extract_json_block,
)

logger = logging.getLogger(__name__)


async def review_node(state: dict[str, Any], ctx: NodeContext) -> dict[str, Any]:
    verdict = "approve"
    feedback: list[dict[str, Any]] = []
    errors: list[str] = []

    try:
        raw = await call_role_llm(
            role_id="dr_reviewer",
            state=state,
            system_prompt=REVIEWER_SYSTEM,
            user_prompt=build_review_prompt(
                draft=state.get("draft_report", ""),
                brief=state.get("research_brief", ""),
                sub_questions=[
                    {"id": s.get("id"), "question": s.get("question")}
                    for s in state.get("sub_questions") or []
                ],
                claim_table=claim_table(state),
            ),
        )
        data = extract_json_block(raw)
        if isinstance(data, dict):
            if str(data.get("verdict", "")).lower() == "revise":
                verdict = "revise"
            for item in data.get("feedback") or []:
                if isinstance(item, dict) and item.get("issue"):
                    feedback.append({
                        "issue": str(item["issue"])[:600],
                        "severity": (
                            "high"
                            if str(item.get("severity", "")).lower() == "high"
                            else "low"
                        ),
                        "suggestion": str(item.get("suggestion", ""))[:400],
                    })
        else:
            errors.append("review: unparseable output — treating as approve")
    except RuntimeError as exc:
        errors.append(f"review: LLM failed ({exc}) — treating as approve")

    # "revise" without at least one high-severity issue downgrades to
    # approve — the reviewer contract says revise is for substantive
    # problems, and this keeps a chatty reviewer from burning the budget.
    if verdict == "revise" and not any(
        f["severity"] == "high" for f in feedback
    ):
        verdict = "approve"

    revision_count = int(state.get("revision_count", 0))
    if verdict == "revise":
        revision_count += 1

    logger.info(
        "deep_research review (task=%s): %s (%d feedback items)",
        ctx.task_id, verdict, len(feedback),
    )
    return {
        "review_verdict": verdict,
        "review_feedback": feedback,
        "revision_count": revision_count,
        "errors": errors,
        "current_phase": "review",
    }


__all__ = ["review_node"]
