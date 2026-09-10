"""``citation_audit`` node — deterministic citation and numeric integrity.

Runs after ``final_verify`` and before ``polish``. Everything upstream of
here judges the report with an LLM; this node is the one place that checks
it *mechanically*:

1. repair citation URLs against the spellings the tools actually returned;
2. validate every inline ``[N]`` against the reference whitelist
   (membership mode, so an index pointing at a dropped source fails rather
   than silently retargeting after renumber);
3. audit numeric grounding — classify every specific number as
   verbatim-grounded, derivation-grounded, suspect, or uncited;
4. on any finding, run **one** LLM repair round restricted to citation
   markers (never content);
5. whatever survives is disclosed in the report's Verification appendix and
   summarised in ``result.json`` — the report still ships.

The repair-once-then-disclose policy is deliberate: a hard gate could spend
the whole revision budget and still hand back a flawed report, and the
scheduler requires a non-empty terminal ``report``. Silence would be worse
than either — a reader cannot tell a checked report from an unchecked one.

Every step is wrapped: a failure here passes the incoming report through
untouched rather than costing the run its deliverable.
"""

from __future__ import annotations

import logging
from typing import Any

from agent_harness.models.node_context import NodeContext
from agent_harness.utils.language import resolve_language

from workflows._shared.cited_report_finalizer import (
    build_citation_maps,
    render_references_block,
    repair_citations_once,
    strip_trailing_references,
    validate_citation_body,
    validate_numeric_grounding,
)
from workflows._shared.citation_contract import (
    finalize_report_with_canonical_references,
)
from workflows._shared.citation_url_repair import repair_citation_urls
from workflows._shared.report_clean import clean_report_v3
from workflows.deep_research.config import get_cfg
from workflows.deep_research.nodes.common import profile_name, usable_cards
from workflows.deep_research.references import (
    cards_for_url_repair,
    mapping_from_references,
    references_from_mapping,
    renumbered_references,
    snippet_lookup_for,
)

logger = logging.getLogger(__name__)

# A number the writer never cited is a much weaker signal than a cited
# number that contradicts its own source: uncited figures are routinely
# legitimate (counts of items in the report's own tables, years in prose).
# Only suspects and citation errors are worth spending a repair call on.
_REPAIR_TIMEOUT_S = 600.0


def _appendix(
    issues: list[str],
    grounding: Any,
    repaired: bool,
) -> str:
    """Render the Verification appendix disclosing what survived repair."""
    lines: list[str] = []
    if issues:
        lines.append("**Citation issues:**")
        lines.extend(f"- {issue}" for issue in issues)
    suspects = list(getattr(grounding, "suspect", ()) or ())
    if suspects:
        lines.append(
            "**Numbers not found in their cited source** "
            "(treat these as unverified):",
        )
        lines.extend(
            f"- `{token}` cited as [{idx}]" for token, idx in suspects[:20]
        )
        if len(suspects) > 20:
            lines.append(f"- …and {len(suspects) - 20} more")
    if not lines:
        return ""
    prefix = (
        "One automated repair pass ran and these findings remained."
        if repaired
        else "Automated checks found the following."
    )
    return (
        "\n\n## Verification appendix — automated citation audit\n\n"
        f"{prefix}\n\n" + "\n".join(lines) + "\n"
    )


def _audit(body: str, references: list[dict[str, str]], cards: list[dict[str, Any]]):
    """Run both deterministic checks over ``body``. Returns (issues, grounding)."""
    citation_map, _titles = build_citation_maps(references)
    valid_indices = [int(k) for k in citation_map]
    issues = validate_citation_body(
        body, max_ref=len(citation_map), valid_indices=valid_indices,
    )
    # "no inline [N] citations were emitted" is a real finding for a
    # research report, so it is kept rather than filtered.
    grounding = validate_numeric_grounding(
        body, snippet_lookup=snippet_lookup_for(references, cards),
    )
    return issues, grounding


async def citation_audit_node(
    state: dict[str, Any], ctx: NodeContext,
) -> dict[str, Any]:
    report = state.get("report") or state.get("final_content") or ""
    if not report.strip():
        return {"current_phase": "citation_audit"}
    if not bool(get_cfg(state, "citation_contract")):
        return {"current_phase": "citation_audit"}

    try:
        return await _run(state, ctx, report)
    except Exception as exc:  # noqa: BLE001 — terminal-output invariant
        logger.warning(
            "deep_research citation_audit (task=%s) failed, passing report "
            "through unchanged: %s", ctx.task_id, exc,
        )
        return {
            "current_phase": "citation_audit",
            "errors": [f"citation_audit: {exc}"],
        }


async def _run(
    state: dict[str, Any], ctx: NodeContext, report: str,
) -> dict[str, Any]:
    cards = usable_cards(state)
    references = references_from_mapping(state.get("citation_mapping") or {})
    if not references:
        logger.info(
            "deep_research citation_audit (task=%s): no references — skipped",
            ctx.task_id,
        )
        return {"current_phase": "citation_audit"}

    # 1. URL repair, before any validation reads the URLs.
    body, url_stats = repair_citation_urls(report, cards_for_url_repair(cards))

    # 2 + 3. Deterministic checks on the body without its references footer.
    clean = strip_trailing_references(body).strip()
    issues, grounding = _audit(clean, references, cards)

    # 4. One repair round, if configured and if there is something to fix.
    rounds = int(get_cfg(state, "citation_repair_rounds") or 0)
    audit_numbers = bool(get_cfg(state, "numeric_grounding"))
    needs_repair = bool(issues) or (audit_numbers and bool(grounding.suspect))
    repaired = False
    if rounds > 0 and needs_repair:
        repair_issues = list(issues)
        if audit_numbers and grounding.suspect:
            repair_issues.append(
                "these numbers are not supported by the source they cite — "
                "re-cite them to a source that does support them, or drop the "
                "citation and hedge the claim: "
                + ", ".join(
                    f"{token} [{idx}]" for token, idx in grounding.suspect[:20]
                ),
            )
        from workflows.deep_research.profile import get_llm_for_role

        try:
            candidate = await repair_citations_once(
                llm=get_llm_for_role("dr_verifier", profile_name(state)),
                question=state.get("original_question", ""),
                draft_answer=state.get("draft_report", ""),
                body=clean,
                references_block=render_references_block(references),
                issues=repair_issues,
                timeout_s=_REPAIR_TIMEOUT_S,
            )
        except Exception as exc:  # noqa: BLE001 — advisory pass, never fatal
            logger.warning("citation repair round failed: %s", exc)
            candidate = ""
        if candidate and candidate.strip():
            new_issues, new_grounding = _audit(candidate, references, cards)
            # Only accept the repair if it actually improved things — a
            # repair that trades three orphan citations for four is worse
            # than the draft it replaced.
            before = len(issues) + len(grounding.suspect)
            after = len(new_issues) + len(new_grounding.suspect)
            if after <= before:
                clean, issues, grounding = candidate, new_issues, new_grounding
                repaired = True
            else:
                logger.info(
                    "citation repair rejected (%d → %d findings)", before, after,
                )

    # 5. Scrub internal ids, disclose what remains, append canonical refs.
    card_ids = {
        str(c.get("card_id")) for c in cards if c.get("card_id")
    }
    clean, clean_stats = clean_report_v3(clean, internal_node_ids=card_ids)
    if audit_numbers or issues:
        clean = clean.rstrip() + _appendix(issues, grounding, repaired)
    # The finalizer renumbers by first appearance and drops uncited sources,
    # so the state mapping has to follow or it stops describing the report
    # that shipped. Predict the new order from the pre-finalize body rather
    # than parsing it back out — the footer parser ignores a References
    # section landing in the first 30% of the text, which is what a short
    # report produces.
    shipped = renumbered_references(clean, references)
    final = finalize_report_with_canonical_references(
        clean, references=references, language=resolve_language(state),
    )
    mapping = mapping_from_references(shipped or references)

    audit = {
        "issues": issues,
        "repaired": repaired,
        "reference_count": len(references),
        "url_repair": url_stats,
        "clean": {k: v for k, v in clean_stats.items() if v},
    }
    numeric = {
        "grounded": len(grounding.grounded),
        "grounded_derived": len(grounding.grounded_derived),
        "suspect": len(grounding.suspect),
        "uncited": len(grounding.uncited),
        "total": grounding.total,
        "suspect_ratio": round(grounding.suspect_ratio, 4),
        "suspect_tokens": [
            {"token": t, "citation": i} for t, i in grounding.suspect[:50]
        ],
    }
    logger.info(
        "deep_research citation_audit (task=%s): %d issues, %d/%d numbers "
        "suspect, repaired=%s",
        ctx.task_id, len(issues), numeric["suspect"], numeric["total"], repaired,
    )
    return {
        "report": final,
        "final_content": final,
        "references": references,
        "citation_mapping": mapping,
        "citation_audit": audit,
        "numeric_grounding": numeric,
        "current_phase": "citation_audit",
    }
