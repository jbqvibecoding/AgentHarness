"""``council_research_fanout`` node — one full deep_research run per member.

Each council member model gets its own complete deep_research sub-pipeline
(all seven roles forced onto that member's model via a runtime profile),
executed by compiling ``DEEP_RESEARCH_SPEC`` directly on the MiniDAG —
member runs are context-isolated by thread_id and run in parallel with a
concurrency cap. A failed member is recorded and excluded; only an
all-member failure sinks the pipeline.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from agent_harness.core.runtime.dag.graph_builder import DynamicGraphBuilder

from agent_harness.models.node_context import NodeContext

from workflows.deep_research.council import (
    build_member_profile,
    load_council_members,
)
from workflows.deep_research.nodes.common import claim_table, profile_name
from workflows.deep_research.profile import (
    load_profile,
    register_runtime_profile,
)
from workflows.deep_research.subagent import gather_with_limit

logger = logging.getLogger(__name__)

_MEMBER_PARALLELISM = 3


def _member_seed(
    state: dict[str, Any], member: dict[str, Any], profile_id: str,
) -> dict[str, Any]:
    """Seed dict for one member's sub-run (mirrors run.py's seed shape —
    list fields must pre-exist for the reducers to fire)."""
    meta = dict(state.get("metadata") or {})
    meta["profile"] = profile_id
    meta["deep_research"] = dict(meta.get("deep_research") or {})
    seed = {
        "task_id": f"{state.get('task_id', 'council')}-{member['slug']}",
        "original_question": state.get("original_question", ""),
        "metadata": meta,
        "sub_questions": [],
        "evidence_cards": [],
        "research_notes": [],
        "fact_check_results": [],
        "conflicts": [],
        "gap_questions": [],
        "contradiction_graph": [],
        "consensus_claims": [],
        "loci": [],
        "review_feedback": [],
        "citation_mapping": {},
        "research_iteration": 0,
        "revision_count": 0,
        "patch_log": [],
        "polish_log": [],
        "report": None,
        "final_content": "",
        "current_phase": "",
        "errors": [],
    }
    # Members share one evidence vault (sources are model-agnostic facts),
    # so a page fetched by one member is reusable by the others. Each
    # member writes into a per-member subdir to avoid sqlite write races.
    parent_vault = state.get("vault_dir")
    if parent_vault:
        seed["vault_dir"] = str(Path(parent_vault) / member["slug"])
    return seed


async def council_research_fanout_node(
    state: dict[str, Any], ctx: NodeContext,
) -> dict[str, Any]:
    from workflows.deep_research.spec import DEEP_RESEARCH_SPEC
    from workflows.deep_research.state import DeepResearchState

    base_profile = load_profile(profile_name(state))
    members = load_council_members(base_profile)

    for member in members:
        register_runtime_profile(
            f"council/{member['slug']}",
            build_member_profile(base_profile, member),
        )

    runner = DynamicGraphBuilder().compile(
        DEEP_RESEARCH_SPEC, DeepResearchState,
    )

    async def run_member(member: dict[str, Any]) -> dict[str, Any]:
        seed = _member_seed(state, member, f"council/{member['slug']}")
        thread_id = seed["task_id"]
        logger.info(
            "council: member '%s' (%s) starting deep_research sub-run",
            member["name"], member["model"],
        )
        async for _chunk in runner.astream(
            seed, config={"configurable": {"thread_id": thread_id}},
        ):
            pass
        snapshot = await runner.aget_state(
            config={"configurable": {"thread_id": thread_id}},
        )
        return dict(getattr(snapshot, "values", None) or {})

    results = await gather_with_limit(
        [run_member(m) for m in members],
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
                "claim_table": [],
                "verification_summary": "",
                "error": str(result)[:300],
            })
            continue
        ok += 1
        member_results.append({
            "name": member["name"],
            "slug": member["slug"],
            "status": "ok",
            "report": result.get("report")
            or result.get("final_content") or "",
            "claim_table": claim_table(result),
            "verification_summary": result.get("verification_summary", ""),
            "error": "",
        })

    if ok == 0:
        raise RuntimeError(
            f"deep_council_research: all {len(members)} member sub-runs "
            f"failed: {errors[:3]}",
        )

    logger.info(
        "council research fan-out done (task=%s): %d/%d members ok",
        ctx.task_id, ok, len(members),
    )
    return {
        "members": [
            {"name": m["name"], "slug": m["slug"], "model": m["model"]}
            for m in members
        ],
        "member_results": member_results,
        "errors": errors,
        "current_phase": "council_research",
    }


__all__ = ["council_research_fanout_node"]
