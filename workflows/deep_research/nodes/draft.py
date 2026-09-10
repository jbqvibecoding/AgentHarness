"""``draft`` node — report writing/revision (role ``dr_writer``).

Evidence is injected as labeled per-sub-question blocks (hermes MoA
aggregation style) with GLOBAL citation numbers, so the writer never
invents numbering. On context-length failures the evidence budget shrinks
progressively (open_deep_research final-report retry pattern).
"""

from __future__ import annotations

import logging
from typing import Any

from agent_harness.models.node_context import NodeContext

from workflows._shared.citation_contract import (
    compose_citation_contract,
    strip_references_section_from_prompt,
)
from workflows.deep_research.citations import citation_listing, normalize_url
from workflows.deep_research.config import get_cfg
from workflows.deep_research.references import references_from_mapping
from workflows.deep_research.hyper_prompts import (
    PATCH_REVISE_SYSTEM,
    build_patch_revise_prompt,
)
from workflows.deep_research.nodes.common import call_role_llm, today, usable_cards
from workflows.deep_research.patch import apply_edit_hunks
from workflows.deep_research.prompts import (
    WRITER_SYSTEM,
    build_draft_prompt,
    extract_json_block,
    with_date,
)

logger = logging.getLogger(__name__)

_EVIDENCE_BUDGETS = (120_000, 80_000, 50_000)
_CONTEXT_ERR_MARKERS = ("longer than the model", "context length", "maximum context")

# The one WRITER_SYSTEM bullet that tells the model to author its own
# References section. Under the citation contract the system appends the
# canonical block instead, and leaving both instructions in place is what
# produces two reference lists — one of them mis-numbered.
_OWN_REFERENCES_BULLET = (
    '- End with a "## References" section listing each cited number once, '
    'sequentially without gaps, as "[n] Title: URL".\n'
)


def _writer_system(state: dict[str, Any]) -> str:
    """WRITER_SYSTEM with the deterministic citation contract applied.

    Returns the prompt unchanged when the contract is disabled or when the
    run produced no citable sources — telling a model to "use one of these
    indices" with an empty index list makes it either refuse to cite or
    invent indices.
    """
    system = with_date(WRITER_SYSTEM, today())
    if not bool(get_cfg(state, "citation_contract")):
        return system
    references = references_from_mapping(state.get("citation_mapping") or {})
    contract = compose_citation_contract(references)
    if not contract:
        return system
    system = system.replace(_OWN_REFERENCES_BULLET, "")
    # Defence in depth for the heading-shaped form the shared helper knows.
    system = strip_references_section_from_prompt(system)
    return system + contract


def _citation_nums_for(card: dict[str, Any], url_to_num: dict[str, int]) -> str:
    nums = sorted({
        url_to_num[normalize_url(s.get("url", ""))]
        for s in card.get("sources") or []
        if normalize_url(s.get("url", "")) in url_to_num
    })
    return "".join(f"[{n}]" for n in nums)


def _evidence_blocks(state: dict[str, Any], budget: int) -> str:
    """Group usable cards by sub-question into labeled evidence blocks."""
    mapping = state.get("citation_mapping") or {}
    url_to_num = {
        normalize_url(doc.get("url", "")): int(num)
        for num, doc in mapping.items()
    }
    questions = {
        s["id"]: s.get("question", "") for s in state.get("sub_questions") or []
    }

    by_sq: dict[str, list[str]] = {}
    for card in usable_cards(state):
        cites = _citation_nums_for(card, url_to_num)
        line = (
            f"- ({card.get('verdict')}) {card.get('claim', '')} {cites}"
        )
        quote = card.get("quote") or ""
        if quote:
            line += f'\n  quote: "{quote}"'
        if card.get("flags"):
            line += f"\n  flags: {', '.join(card['flags'])}"
        by_sq.setdefault(str(card.get("sub_question_id", "?")), []).append(line)

    blocks: list[str] = []
    for sq_id, lines in by_sq.items():
        blocks.append(
            f"### Evidence — {sq_id}: {questions.get(sq_id, '')}\n"
            + "\n".join(lines)
        )
    text = "\n\n".join(blocks)
    if len(text) > budget:
        text = text[:budget] + "\n…[evidence truncated]"
    return text


async def _patch_revise(
    state: dict[str, Any], ctx: NodeContext,
    feedback: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Revise the existing draft via surgical edit hunks (patch, never
    regenerate). Returns a node delta, or None to fall back to full regen
    when no hunk could be applied."""
    draft = state.get("draft_report", "")
    if not draft:
        return None
    try:
        raw = await call_role_llm(
            role_id="dr_writer", state=state,
            system_prompt=PATCH_REVISE_SYSTEM,
            user_prompt=build_patch_revise_prompt(draft, feedback),
            timeout_s=900.0,
        )
    except RuntimeError as exc:
        logger.warning("draft patch-revise LLM failed: %s — full regen", exc)
        return None
    data = extract_json_block(raw)
    hunks = data.get("hunks") if isinstance(data, dict) else None
    if not hunks:
        return None
    new_text, log = apply_edit_hunks(
        draft, hunks, citation_mapping=state.get("citation_mapping") or {},
    )
    applied = [e for e in log if e["applied"]]
    if not applied:
        logger.info("draft patch-revise: no hunk applied — full regen")
        return None
    escalated = data.get("escalated") if isinstance(data, dict) else []
    logger.info(
        "deep_research draft/patch (task=%s): %d/%d hunks applied",
        ctx.task_id, len(applied), len(hunks),
    )
    return {
        "draft_report": new_text,
        "patch_log": [{
            "phase": "revision", "applied": len(applied),
            "total": len(hunks), "log": log,
            "escalated": list(escalated or []),
        }],
        "current_phase": "draft",
    }


async def draft_node(state: dict[str, Any], ctx: NodeContext) -> dict[str, Any]:
    feedback = (
        state.get("review_feedback")
        if state.get("review_verdict") == "revise"
        else None
    )
    # Revision path: prefer surgical patches over regenerating the report.
    if feedback and bool(get_cfg(state, "patch_revision")):
        patched = await _patch_revise(state, ctx, feedback)
        if patched is not None:
            return patched

    listing = citation_listing(state.get("citation_mapping") or {})
    system = _writer_system(state)

    last_exc: Exception | None = None
    for budget in _EVIDENCE_BUDGETS:
        user = build_draft_prompt(
            question=state.get("original_question", ""),
            brief=state.get("research_brief", ""),
            evidence_blocks=_evidence_blocks(state, budget),
            conflicts=state.get("conflicts") or [],
            citation_listing=listing,
            feedback=feedback,
            previous_draft=state.get("draft_report", "") if feedback else "",
        )
        try:
            draft = await call_role_llm(
                role_id="dr_writer", state=state,
                system_prompt=system, user_prompt=user,
                timeout_s=900.0,
            )
            logger.info(
                "deep_research draft (task=%s): %d chars%s",
                ctx.task_id, len(draft),
                " (revision)" if feedback else "",
            )
            return {"draft_report": draft, "current_phase": "draft"}
        except RuntimeError as exc:
            last_exc = exc
            if any(m in str(exc) for m in _CONTEXT_ERR_MARKERS):
                logger.warning(
                    "draft: context overflow at budget %d — shrinking", budget,
                )
                continue
            raise
    raise RuntimeError(
        f"draft_node failed at all evidence budgets: {last_exc}",
    ) from last_exc


__all__ = ["draft_node"]
