"""``review`` node — adversarial critics over the draft (role ``dr_reviewer``).

Upgraded from a single reviewer to the hyperresearch four-critic panel
(dialectic / depth / width / instruction) run in parallel, each finding
problems only (never rewriting). Findings merge into ``review_feedback``
which drives the patch-based revision loop. ``num_critics`` (depth-gated)
selects how many critics run — quick uses just the instruction critic,
the highest-variance dimension. Any critic failing is non-fatal; an
all-clear panel approves.
"""

from __future__ import annotations

import logging
from typing import Any

from agent_harness.models.node_context import NodeContext

from workflows.deep_research.config import get_cfg
from workflows.deep_research.hyper_prompts import build_critic_prompt, critic_system
from workflows.deep_research.nodes.common import call_role_llm, claim_table
from workflows.deep_research.prompts import extract_json_block
from workflows.deep_research.subagent import gather_with_limit

logger = logging.getLogger(__name__)

# Priority order — instruction first (widest quality variance, always on).
_CRITIC_ORDER = ["instruction", "dialectic", "width", "depth"]


def _select_critics(n: int) -> list[str]:
    n = max(1, min(len(_CRITIC_ORDER), n))
    return _CRITIC_ORDER[:n]


async def review_node(state: dict[str, Any], ctx: NodeContext) -> dict[str, Any]:
    critics = _select_critics(int(get_cfg(state, "num_critics")))
    draft = state.get("draft_report", "")
    brief = state.get("research_brief", "")
    sub_qs = [
        {"id": s.get("id"), "question": s.get("question")}
        for s in state.get("sub_questions") or []
    ]
    table = claim_table(state)
    graph = state.get("contradiction_graph") or []

    async def run_critic(ctype: str) -> tuple[str, Any]:
        raw = await call_role_llm(
            role_id="dr_reviewer",
            state=state,
            system_prompt=critic_system(ctype),
            user_prompt=build_critic_prompt(
                ctype, draft, brief, sub_qs, table, graph,
            ),
        )
        return ctype, extract_json_block(raw)

    results = await gather_with_limit(
        [run_critic(c) for c in critics],
        limit=int(get_cfg(state, "max_parallel_subagents")),
    )

    feedback: list[dict[str, Any]] = []
    errors: list[str] = []
    for res in results:
        if isinstance(res, BaseException):
            errors.append(f"review: critic failed: {res!r}"[:300])
            continue
        ctype, data = res
        if not isinstance(data, dict):
            errors.append(f"review: {ctype} critic unparseable")
            continue
        for item in data.get("findings") or []:
            if isinstance(item, dict) and item.get("issue"):
                feedback.append({
                    "critic": ctype,
                    "issue": str(item["issue"])[:600],
                    "severity": ("high"
                                 if str(item.get("severity", "")).lower() == "high"
                                 else "low"),
                    "failure_mode": str(item.get("failure_mode", "other"))[:40],
                    "suggestion": str(item.get("suggestion", ""))[:400],
                })

    # Revise only when a substantive (high-severity) problem exists.
    verdict = "revise" if any(f["severity"] == "high" for f in feedback) else "approve"
    revision_count = int(state.get("revision_count", 0))
    if verdict == "revise":
        revision_count += 1

    logger.info(
        "deep_research review (task=%s): %d critics → %s (%d findings)",
        ctx.task_id, len(critics), verdict, len(feedback),
    )
    return {
        "review_verdict": verdict,
        "review_feedback": feedback,
        "revision_count": revision_count,
        "errors": errors,
        "current_phase": "review",
    }


__all__ = ["review_node"]
