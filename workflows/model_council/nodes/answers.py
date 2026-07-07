"""``member_answers`` node — one direct completion per council member.

Lightweight council: no tools, no research loop — each member model
answers the question in parallel; failures are isolated per member and
only an all-member failure raises.
"""

from __future__ import annotations

import logging
from typing import Any

from agent_harness.models.node_context import NodeContext

from workflows.deep_research.council import (
    chat_with_retries,
    get_llm_for_member,
    load_council_members,
)
from workflows.deep_research.council_prompts import (
    COUNCIL_MEMBER_SYSTEM,
    build_member_prompt,
)
from workflows.deep_research.nodes.common import profile_name, today
from workflows.deep_research.profile import load_profile
from workflows.deep_research.prompts import with_date
from workflows.deep_research.subagent import gather_with_limit

logger = logging.getLogger(__name__)

_MEMBER_PARALLELISM = 3


async def member_answers_node(
    state: dict[str, Any], ctx: NodeContext,
) -> dict[str, Any]:
    question = (state.get("original_question") or "").strip()
    if not question:
        raise ValueError("member_answers_node requires 'original_question'")

    profile = load_profile(profile_name(state))
    members = load_council_members(profile)
    system = with_date(COUNCIL_MEMBER_SYSTEM, today())

    async def ask(member: dict[str, Any]) -> str:
        return await chat_with_retries(
            get_llm_for_member(member),
            system_prompt=system,
            user_prompt=build_member_prompt(question),
            timeout_s=600.0,
        )

    logger.info(
        "model_council (task=%s): asking %d members in parallel",
        ctx.task_id, len(members),
    )
    results = await gather_with_limit(
        [ask(m) for m in members],
        limit=min(_MEMBER_PARALLELISM, len(members)),
    )

    member_results: list[dict[str, Any]] = []
    errors: list[str] = []
    ok = 0
    for member, result in zip(members, results):
        if isinstance(result, BaseException):
            errors.append(
                f"council member '{member['name']}' failed: {result!r}"[:500],
            )
            member_results.append({
                "name": member["name"],
                "slug": member["slug"],
                "status": "failed",
                "report": "",
                "error": str(result)[:300],
            })
            continue
        ok += 1
        member_results.append({
            "name": member["name"],
            "slug": member["slug"],
            "status": "ok",
            "report": result,
            "error": "",
        })

    if ok == 0:
        raise RuntimeError(
            f"model_council: all {len(members)} members failed: {errors[:3]}",
        )

    return {
        "members": [
            {"name": m["name"], "slug": m["slug"], "model": m["model"]}
            for m in members
        ],
        "member_results": member_results,
        "errors": errors,
        "current_phase": "member_answers",
    }


__all__ = ["member_answers_node"]
