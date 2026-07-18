"""``research_fanout`` node — parallel evidence-gathering sub-agents.

Fans one ``dr_researcher`` ReAct sub-agent out per pending sub-question
(open_deep_research supervisor pattern: bounded ``asyncio.gather``, each
sub-agent context-isolated with a self-contained instruction). Branch
failures/timeouts mark that sub-question ``failed`` and never sink the
pipeline (onyx timeout-placeholder pattern); only a fully-failed round
raises.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from agent_harness.models.node_context import NodeContext

from workflows.deep_research.citations import merge_sources
from workflows.deep_research.config import get_cfg
from workflows.deep_research.nodes.common import profile_name, today
from workflows.deep_research.prompts import (
    RESEARCHER_SYSTEM,
    build_research_prompt,
    extract_json_block,
    with_date,
)
from workflows.deep_research.subagent import gather_with_limit, run_subagent

logger = logging.getLogger(__name__)

_MAX_CLAIM_CHARS = 400
_MAX_QUOTE_CHARS = 600
_MAX_CARDS_PER_SUBAGENT = 12


def _merge_gap_questions(state: dict[str, Any]) -> list[dict[str, Any]]:
    """Fold pending ``gap_questions`` into ``sub_questions`` with fresh ids."""
    sub_questions = [dict(s) for s in state.get("sub_questions") or []]
    existing = {s["id"] for s in sub_questions}
    iteration = int(state.get("research_iteration", 0))
    n = len(sub_questions)
    for gap in state.get("gap_questions") or []:
        question = str(gap.get("question", "")).strip()
        if not question:
            continue
        n += 1
        new_id = f"sq{n}"
        while new_id in existing:
            n += 1
            new_id = f"sq{n}"
        existing.add(new_id)
        sub_questions.append({
            "id": new_id,
            "question": question,
            "rationale": str(gap.get("rationale", "")),
            "status": "pending",
            "iteration": iteration,
        })
    return sub_questions


def _parse_evidence(
    raw: str, sub_q_id: str, iteration: int,
) -> list[dict[str, Any]]:
    """Parse the researcher's EVIDENCE JSON into capped evidence cards.

    Parse failure degrades to one low-confidence card wrapping the raw
    findings so no researcher output is silently dropped.
    """
    data = extract_json_block(raw)
    cards: list[dict[str, Any]] = []
    if isinstance(data, list):
        for i, item in enumerate(data[:_MAX_CARDS_PER_SUBAGENT]):
            if not isinstance(item, dict):
                continue
            claim = str(item.get("claim", "")).strip()
            if not claim:
                continue
            sources = [
                {
                    "url": str(s.get("url", "")).strip(),
                    "title": str(s.get("title", "")).strip()[:300],
                }
                for s in item.get("sources") or []
                if isinstance(s, dict) and s.get("url")
            ]
            confidence = str(item.get("confidence", "medium")).lower()
            if confidence not in {"high", "medium", "low"}:
                confidence = "medium"
            flags = [
                str(f)[:120] for f in (item.get("flags") or [])
                if isinstance(f, str) and f.strip()
            ]
            cards.append({
                "card_id": f"{sub_q_id}-i{iteration}-c{i + 1}",
                "sub_question_id": sub_q_id,
                "claim": claim[:_MAX_CLAIM_CHARS],
                "quote": str(item.get("quote", ""))[:_MAX_QUOTE_CHARS],
                "sources": sources,
                "confidence": confidence,
                "flags": flags,
                "iteration": iteration,
            })
    if not cards and raw.strip():
        cards.append({
            "card_id": f"{sub_q_id}-i{iteration}-raw",
            "sub_question_id": sub_q_id,
            "claim": raw.strip()[:_MAX_CLAIM_CHARS],
            "quote": "",
            "sources": [],
            "confidence": "low",
            "flags": ["unstructured researcher output"],
            "iteration": iteration,
        })
    return cards


async def research_fanout_node(
    state: dict[str, Any], ctx: NodeContext,
) -> dict[str, Any]:
    sub_questions = _merge_gap_questions(state)
    targets = [s for s in sub_questions if s.get("status") == "pending"]
    iteration = int(state.get("research_iteration", 0)) + 1

    if not targets:
        logger.info("deep_research research: no pending sub-questions")
        return {
            "research_iteration": iteration,
            "sub_questions": sub_questions,
            "gap_questions": [],
            "current_phase": "research",
        }

    brief = state.get("research_brief", "")
    system = with_date(RESEARCHER_SYSTEM, today())
    max_turns = int(get_cfg(state, "researcher_max_turns"))
    timeout_s = float(get_cfg(state, "subagent_timeout_s"))
    parallel = int(get_cfg(state, "max_parallel_subagents"))
    prof = profile_name(state)

    # Optional evidence vault: fetched pages are mirrored here so
    # fact-checkers can re-read them without re-fetching. No-ops when the
    # vault is disabled or hyperresearch is not installed.
    vault_dir = state.get("vault_dir")
    vault = None
    scope_meta = None
    if vault_dir and bool(get_cfg(state, "use_vault")):
        from workflows.deep_research.vault import open_vault

        vault = open_vault(vault_dir, enabled=True)
        if vault is not None:
            scope_meta = {"vault_dir": vault_dir}

    logger.info(
        "deep_research research iteration %d (task=%s): %d sub-agents, "
        "parallel=%d, max_turns=%d, vault=%s",
        iteration, ctx.task_id, len(targets), parallel, max_turns,
        "on" if vault is not None else "off",
    )

    def _observers_for(sq_id: str) -> list[Any] | None:
        if vault is None:
            return None
        from workflows.deep_research.vault import VaultWriterObserver

        return [VaultWriterObserver(vault, suggested_by=sq_id)]

    results = await gather_with_limit(
        [
            run_subagent(
                role_id="dr_researcher",
                system_prompt=system,
                user_message=build_research_prompt(
                    t["question"], brief,
                    locus=t if t.get("flavor") == "locus" else None,
                ),
                max_turns=max_turns,
                task_id=ctx.task_id,
                profile_name=prof,
                timeout_s=timeout_s,
                extra_observers=_observers_for(t["id"]),
                scope_metadata=scope_meta,
            )
            for t in targets
        ],
        limit=parallel,
    )
    if vault is not None:
        vault.close()

    new_cards: list[dict[str, Any]] = []
    new_notes: list[dict[str, Any]] = []
    errors: list[str] = []
    citation_mapping = dict(state.get("citation_mapping") or {})
    by_id = {s["id"]: s for s in sub_questions}

    for target, result in zip(targets, results):
        sq = by_id[target["id"]]
        if isinstance(result, BaseException):
            kind = (
                "timeout" if isinstance(result, asyncio.TimeoutError)
                else "error"
            )
            sq["status"] = "failed"
            errors.append(
                f"research[{sq['id']}] {kind}: {result!r}"[:500],
            )
            new_notes.append({
                "sub_question_id": sq["id"],
                "question": sq["question"],
                "summary": f"researcher {kind} — no evidence collected",
                "status": "failed",
                "turns_used": 0,
            })
            continue

        cards = _parse_evidence(result.final_content, sq["id"], iteration)
        for card in cards:
            citation_mapping, _ = merge_sources(
                citation_mapping, card["sources"],
            )
        sq["status"] = "researched" if cards else "failed"
        new_cards.extend(cards)
        new_notes.append({
            "sub_question_id": sq["id"],
            "question": sq["question"],
            "summary": (result.final_content or "")[:2000],
            "status": sq["status"],
            "turns_used": result.turns_used,
        })

    if not new_cards and all(
        isinstance(r, BaseException) for r in results
    ):
        raise RuntimeError(
            f"deep_research: all {len(targets)} researcher sub-agents "
            f"failed in iteration {iteration}: {errors[:3]}",
        )

    return {
        "evidence_cards": new_cards,
        "research_notes": new_notes,
        "sub_questions": sub_questions,
        "gap_questions": [],  # consumed by the merge above
        "research_iteration": iteration,
        "citation_mapping": citation_mapping,
        "errors": errors,
        "current_phase": "research",
    }


__all__ = ["research_fanout_node"]
