"""``final_verify`` node — global verifier, terminal gate (role ``dr_verifier``).

Judge-style per-claim audit against the fact-check verdicts, surgical
corrections only, and an honest verification summary (hermes rule: state
what was NOT verified instead of claiming it was). On ``deep`` depth a
short web-search spot-check loop precedes the audit.

Terminal-output guarantee: this node ALWAYS returns non-empty ``report``
and ``final_content`` (Scheduler requirement) — verifier failure falls
back to the draft plus an explicit note; a missing draft falls back to a
minimal report synthesized from research notes.
"""

from __future__ import annotations

import logging
from typing import Any

from agent_harness.models.node_context import NodeContext

from workflows.deep_research.citations import citation_listing
from workflows.deep_research.config import get_cfg
from workflows.deep_research.nodes.common import (
    call_role_llm,
    claim_table,
    profile_name,
    today,
)
from workflows.deep_research.prompts import (
    VERIFIER_SYSTEM,
    build_verify_prompt,
    extract_json_block,
    with_date,
)

logger = logging.getLogger(__name__)


def _fallback_report(state: dict[str, Any]) -> str:
    """Minimal report from research notes — last-resort terminal output."""
    lines = [
        f"# Research notes: {state.get('original_question', '')}",
        "",
        "*Report generation degraded — raw research summaries below.*",
        "",
    ]
    for note in state.get("research_notes") or []:
        lines.append(f"## {note.get('question', note.get('sub_question_id'))}")
        lines.append(note.get("summary", "(no findings)"))
        lines.append("")
    listing = citation_listing(state.get("citation_mapping") or {})
    if listing:
        lines.extend(["## References", listing])
    return "\n".join(lines)


async def _spot_check(state: dict[str, Any], ctx: NodeContext) -> str:
    """Optional web spot-check of the highest-stakes claims (deep depth)."""
    from workflows.deep_research.prompts import FACT_CHECKER_SYSTEM
    from workflows.deep_research.subagent import run_subagent

    table = [
        row for row in claim_table(state)
        if row["verdict"] in {"supported", "partially_supported"}
    ][:3]
    if not table:
        return ""
    try:
        result = await run_subagent(
            role_id="dr_verifier",
            system_prompt=with_date(FACT_CHECKER_SYSTEM, today()),
            user_message=(
                "Spot-check these already-verified claims one more time "
                "with independent searches; report anything that looks "
                "wrong (plain text, brief):\n"
                + "\n".join(f"- {r['card_id']}: {r['claim']}" for r in table)
            ),
            max_turns=int(get_cfg(state, "verifier_spot_check_turns")),
            task_id=ctx.task_id,
            profile_name=profile_name(state),
            timeout_s=float(get_cfg(state, "subagent_timeout_s")),
            state=state,
        )
        return (result.final_content or "")[:3000]
    except Exception as exc:  # noqa: BLE001 — spot-check is best-effort
        logger.warning("verifier spot-check failed (non-fatal): %s", exc)
        return ""


def _coverage_caveats(state: dict[str, Any]) -> str:
    """Disclose which sub-questions were researched only partially.

    A branch cut short by a token, turn, loop or wall limit still returns
    its evidence, and that evidence still reaches the report. Saying so is
    the difference between a partial answer and a partial answer the reader
    mistakes for a complete one.
    """
    from workflows.deep_research.stop_reason import describe

    capped = [
        (note.get("question") or note.get("sub_question_id") or "?",
         describe(note.get("stop_reason")))
        for note in state.get("research_notes") or []
        if note.get("stop_reason")
    ]
    if not capped:
        return ""
    lines = "\n".join(f"- {question} — {why}" for question, why in capped)
    return (
        "\n\n**Coverage caveats.** These sub-questions were researched only "
        "partially; their findings are included but are less complete than "
        f"the rest:\n{lines}\n"
    )


async def final_verify_node(
    state: dict[str, Any], ctx: NodeContext,
) -> dict[str, Any]:
    draft = state.get("draft_report", "")
    errors: list[str] = []

    if not draft:
        report = _fallback_report(state)
        summary = (
            "Verification skipped: no draft was produced. The output is a "
            "degraded compilation of raw research notes and must not be "
            "treated as a verified report."
        )
        return {
            "report": report,
            "final_content": report,
            "verification_summary": summary,
            "errors": ["final_verify: empty draft — emitted fallback report"],
            "current_phase": "final_verify",
        }

    spot_notes = ""
    if bool(get_cfg(state, "verifier_spot_check")):
        spot_notes = await _spot_check(state, ctx)

    report = draft
    summary = ""
    patch_log: list[dict[str, Any]] = []
    try:
        user_prompt = build_verify_prompt(
            report=draft,
            claim_table=claim_table(state),
            conflicts=state.get("conflicts") or [],
            revision_count=int(state.get("revision_count", 0)),
        )
        if spot_notes:
            user_prompt += f"\n\nSpot-check notes from live searches:\n{spot_notes}"
        raw = await call_role_llm(
            role_id="dr_verifier",
            state=state,
            system_prompt=with_date(VERIFIER_SYSTEM, today()),
            user_prompt=user_prompt,
            timeout_s=900.0,
        )
        data = extract_json_block(raw)
        if isinstance(data, dict):
            summary = str(data.get("verification_summary") or "").strip()
            hunks = data.get("hunks") or []
            if hunks:
                from workflows.deep_research.patch import apply_edit_hunks

                report, plog = apply_edit_hunks(
                    draft, hunks,
                    citation_mapping=state.get("citation_mapping") or {},
                )
                applied = sum(1 for e in plog if e["applied"])
                patch_log = [{
                    "phase": "verify", "applied": applied,
                    "total": len(hunks), "log": plog,
                }]
                logger.info(
                    "final_verify: %d/%d correction hunks applied",
                    applied, len(hunks),
                )
        else:
            errors.append("final_verify: unparseable verifier output")
    except RuntimeError as exc:
        errors.append(f"final_verify: verifier LLM failed: {exc}")

    if not summary:
        summary = (
            "Verifier unavailable — the report is the reviewed draft. "
            "Claims were fact-checked during the pipeline (see per-claim "
            "verdicts in state), but no final global audit was completed."
        )

    final_report = f"{report}\n\n## Verification\n\n{summary}{_coverage_caveats(state)}"

    logger.info(
        "deep_research final_verify (task=%s): report %d chars",
        ctx.task_id, len(final_report),
    )
    return {
        "report": final_report,
        "final_content": final_report,
        "verification_summary": summary,
        "patch_log": patch_log,
        "errors": errors,
        "current_phase": "final_verify",
    }


__all__ = ["final_verify_node"]
