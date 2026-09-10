"""``fact_check`` node — parallel claim verification (role ``dr_fact_checker``).

Checks this iteration's evidence cards in small batches, low-confidence
and flagged cards first (local-deep-research cross-reference discipline;
hermes rule: always re-fetch the cited source). Cards beyond the budget
get an explicit ``unverified`` verdict — silence is not a verdict.
Checker failures are never fatal.
"""

from __future__ import annotations

import logging
from typing import Any

from agent_harness.models.node_context import NodeContext

from workflows.deep_research.config import get_cfg
from workflows.deep_research.nodes.common import profile_name, today
from workflows.deep_research.prompts import (
    FACT_CHECKER_SYSTEM,
    build_fact_check_prompt,
    extract_json_block,
    with_date,
)
from workflows.deep_research.subagent import gather_with_limit, run_subagent

logger = logging.getLogger(__name__)

_VALID_VERDICTS = {
    "supported",
    "partially_supported",
    "unsupported",
    "contradicted",
    "unverified",
}
_CONF_ORDER = {"low": 0, "medium": 1, "high": 2}


def _select_cards(state: dict[str, Any]) -> tuple[list[dict], list[dict]]:
    """Return (cards_to_check, cards_over_budget) for this iteration."""
    checked = {
        str(r.get("card_id")) for r in state.get("fact_check_results") or []
    }
    iteration = int(state.get("research_iteration", 0))
    candidates = [
        c for c in state.get("evidence_cards") or []
        if str(c.get("card_id")) not in checked
        and int(c.get("iteration", 0)) == iteration
        and c.get("sources")  # sourceless cards can't be re-fetched
    ]
    # Low confidence and flagged first — highest hallucination risk.
    candidates.sort(key=lambda c: (
        _CONF_ORDER.get(str(c.get("confidence")), 1),
        -len(c.get("flags") or []),
    ))
    budget = int(get_cfg(state, "max_cards_to_check"))
    return candidates[:budget], candidates[budget:]


def _parse_verdicts(
    raw: str, batch: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    expected = {str(c["card_id"]) for c in batch}
    results: list[dict[str, Any]] = []
    data = extract_json_block(raw)
    if isinstance(data, list):
        for item in data:
            if not isinstance(item, dict):
                continue
            card_id = str(item.get("card_id", ""))
            if card_id not in expected:
                continue
            verdict = str(item.get("verdict", "")).lower()
            if verdict not in _VALID_VERDICTS:
                verdict = "unverified"
            results.append({
                "card_id": card_id,
                "verdict": verdict,
                "corrected_claim": str(item.get("corrected_claim", ""))[:400],
                "notes": str(item.get("notes", ""))[:800],
            })
            expected.discard(card_id)
    for card_id in expected:  # checker skipped it → explicit unverified
        results.append({
            "card_id": card_id,
            "verdict": "unverified",
            "corrected_claim": "",
            "notes": "checker returned no verdict for this card",
        })
    return results


async def fact_check_node(
    state: dict[str, Any], ctx: NodeContext,
) -> dict[str, Any]:
    to_check, over_budget = _select_cards(state)
    new_results: list[dict[str, Any]] = []
    errors: list[str] = []

    for card in over_budget:
        new_results.append({
            "card_id": str(card["card_id"]),
            "verdict": "unverified",
            "corrected_claim": "",
            "notes": "not checked (verification budget)",
        })

    if not to_check:
        logger.info("deep_research fact_check: nothing to check this round")
        return {
            "fact_check_results": new_results,
            "current_phase": "fact_check",
        }

    batch_size = max(1, int(get_cfg(state, "cards_per_checker")))
    batches = [
        to_check[i : i + batch_size]
        for i in range(0, len(to_check), batch_size)
    ]
    system = with_date(FACT_CHECKER_SYSTEM, today())
    max_turns = int(get_cfg(state, "fact_checker_max_turns"))
    timeout_s = float(get_cfg(state, "subagent_timeout_s"))
    prof = profile_name(state)

    vault_dir = state.get("vault_dir")
    scope_meta = (
        {"vault_dir": vault_dir}
        if vault_dir and bool(get_cfg(state, "use_vault"))
        else None
    )

    logger.info(
        "deep_research fact_check (task=%s): %d cards in %d batches",
        ctx.task_id, len(to_check), len(batches),
    )

    results = await gather_with_limit(
        [
            run_subagent(
                role_id="dr_fact_checker",
                system_prompt=system,
                user_message=build_fact_check_prompt(batch),
                max_turns=max_turns,
                task_id=ctx.task_id,
                profile_name=prof,
                timeout_s=timeout_s,
                scope_metadata=scope_meta,
                state=state,
            )
            for batch in batches
        ],
        limit=int(get_cfg(state, "max_parallel_subagents")),
    )

    for batch, result in zip(batches, results):
        if isinstance(result, BaseException):
            errors.append(f"fact_check batch failed: {result!r}"[:500])
            for card in batch:
                new_results.append({
                    "card_id": str(card["card_id"]),
                    "verdict": "unverified",
                    "corrected_claim": "",
                    "notes": "checker sub-agent failed",
                })
            continue
        new_results.extend(_parse_verdicts(result.final_content, batch))

    return {
        "fact_check_results": new_results,
        "errors": errors,
        "current_phase": "fact_check",
    }


__all__ = ["fact_check_node"]
