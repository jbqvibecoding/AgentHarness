"""``plan`` node — brief + sub-question decomposition (role ``dr_planner``)."""

from __future__ import annotations

import logging
from typing import Any

from agent_harness.models.node_context import NodeContext

from workflows.deep_research.config import get_cfg
from workflows.deep_research.nodes.common import call_role_llm, today
from workflows.deep_research.prompts import (
    PLANNER_SYSTEM,
    build_plan_prompt,
    extract_json_block,
    with_date,
)

logger = logging.getLogger(__name__)


def _parse_plan(raw: str) -> tuple[str, list[dict[str, Any]]]:
    data = extract_json_block(raw)
    if not isinstance(data, dict):
        return "", []
    brief = str(data.get("research_brief") or "").strip()
    subs: list[dict[str, Any]] = []
    for item in data.get("sub_questions") or []:
        if isinstance(item, dict) and str(item.get("question", "")).strip():
            subs.append({
                "question": str(item["question"]).strip(),
                "rationale": str(item.get("rationale", "")).strip(),
            })
        elif isinstance(item, str) and item.strip():
            subs.append({"question": item.strip(), "rationale": ""})
    return brief, subs


async def plan_node(state: dict[str, Any], ctx: NodeContext) -> dict[str, Any]:
    question = (state.get("original_question") or "").strip()
    if not question:
        raise ValueError("plan_node requires 'original_question' in state")

    max_subs = int(get_cfg(state, "max_sub_questions"))
    system = with_date(PLANNER_SYSTEM, today())
    user = build_plan_prompt(question, max_subs)

    brief, subs = "", []
    errors: list[str] = []
    for attempt in range(2):
        try:
            raw = await call_role_llm(
                role_id="dr_planner", state=state,
                system_prompt=system,
                user_prompt=user if attempt == 0 else (
                    user + "\n\nYour previous output was not valid JSON. "
                    "Output ONLY the JSON object."
                ),
            )
        except RuntimeError as exc:
            errors.append(f"plan: LLM failed: {exc}")
            break
        brief, subs = _parse_plan(raw)
        if subs:
            break

    if not subs:
        # Fallback: research the original question as a single sub-question.
        errors.append("plan: decomposition failed — using original question")
        subs = [{"question": question, "rationale": "fallback"}]
    if not brief:
        brief = question

    sub_questions = [
        {
            "id": f"sq{i + 1}",
            "question": s["question"],
            "rationale": s.get("rationale", ""),
            "status": "pending",
            "iteration": 0,
        }
        for i, s in enumerate(subs[:max_subs])
    ]

    logger.info(
        "deep_research plan (task=%s): %d sub-questions",
        ctx.task_id, len(sub_questions),
    )
    return {
        "research_brief": brief,
        "sub_questions": sub_questions,
        "research_iteration": 0,
        "revision_count": 0,
        "citation_mapping": {},
        "conflicts": [],
        "gap_questions": [],
        "errors": errors,
        "current_phase": "plan",
    }


__all__ = ["plan_node"]
