"""``council_synthesis`` node — analyst tables + synthesizer, shared by
``deep_council_research`` (mode "deep") and ``model_council`` (mode
"light"); the mode comes from ``metadata["council_mode"]``.

Fail-open design: an unavailable analyst degrades to per-member headline
tables; an unavailable synthesizer degrades to concatenated member
summaries — the terminal report is always non-empty (Scheduler contract).
"""

from __future__ import annotations

import logging
from typing import Any

from agent_harness.models.node_context import NodeContext

from workflows.deep_research.council import (
    chat_with_retries,
    get_synthesizer_llm,
)
from workflows.deep_research.council_prompts import (
    COUNCIL_ANALYST_SYSTEM,
    COUNCIL_SYNTH_DEEP_SYSTEM,
    COUNCIL_SYNTH_LIGHT_SYSTEM,
    build_analyst_prompt,
    build_synth_prompt,
)
from workflows.deep_research.council_tables import (
    degraded_council,
    normalize_council,
    render_markdown_tables,
    render_ranking_table,
)
from workflows.deep_research.nodes.common import profile_name
from workflows.deep_research.profile import load_profile
from workflows.deep_research.prompts import extract_json_block

logger = logging.getLogger(__name__)

_DEEP_EXCERPT_CHARS = 15_000
_LIGHT_EXCERPT_CHARS = 8_000


def _member_blocks(
    member_results: list[dict[str, Any]], mode: str,
) -> list[dict[str, Any]]:
    limit = _DEEP_EXCERPT_CHARS if mode == "deep" else _LIGHT_EXCERPT_CHARS
    blocks: list[dict[str, Any]] = []
    for res in member_results:
        if res.get("status") != "ok":
            continue
        content = (res.get("report") or "").strip()
        if len(content) > limit:
            content = content[:limit] + "\n…[truncated]"
        block: dict[str, Any] = {"name": res.get("name", "?"), "content": content}
        if mode == "deep" and res.get("claim_table"):
            block["claim_table"] = res["claim_table"]
        blocks.append(block)
    return blocks


def _assemble_report(
    *,
    question: str,
    mode: str,
    synthesis: str,
    tables_md: str,
    member_results: list[dict[str, Any]],
) -> str:
    title = (
        "Deep Council Research Report" if mode == "deep"
        else "Model Council Report"
    )
    artifact_dir = "papers" if mode == "deep" else "answers"
    parts = [
        f"# {title}",
        "",
        f"**Question:** {question}",
        "",
        "## Council Synthesis",
        "",
        synthesis,
        "",
        tables_md,
        "",
        "## Member Outputs",
        "",
    ]
    for res in member_results:
        if res.get("status") == "ok":
            parts.append(
                f"- **{res['name']}** — `{artifact_dir}/{res['slug']}.md`",
            )
            if mode == "deep" and res.get("verification_summary"):
                parts.append(
                    f"  - verification: "
                    f"{res['verification_summary'][:400]}",
                )
        else:
            parts.append(
                f"- **{res.get('name', '?')}** — FAILED "
                f"({res.get('error', 'unknown error')[:200]})",
            )
    return "\n".join(parts)


async def council_synthesis_node(
    state: dict[str, Any], ctx: NodeContext,
) -> dict[str, Any]:
    mode = (
        "deep"
        if (state.get("metadata") or {}).get("council_mode") == "deep"
        else "light"
    )
    question = state.get("original_question", "")
    member_results = state.get("member_results") or []
    member_names = [
        r["name"] for r in member_results if r.get("status") == "ok"
    ]
    blocks = _member_blocks(member_results, mode)
    errors: list[str] = []
    # Peer-review signal (empty when that stage is disabled or skipped).
    peer_ranking = state.get("peer_ranking") or []
    peer_reviews = state.get("peer_reviews") or []

    profile = load_profile(profile_name(state))
    llm = get_synthesizer_llm(profile)

    # 1) Analyst — the three comparison tables.
    council: dict[str, Any] | None = None
    if len(blocks) >= 2:
        try:
            raw = await chat_with_retries(
                llm,
                system_prompt=COUNCIL_ANALYST_SYSTEM,
                user_prompt=build_analyst_prompt(
                    question, blocks, mode, peer_ranking, peer_reviews,
                ),
                timeout_s=900.0,
            )
            parsed = extract_json_block(raw)
            if isinstance(parsed, dict):
                council = normalize_council(parsed, member_names)
            else:
                errors.append("council analyst: unparseable output")
        except RuntimeError as exc:
            errors.append(f"council analyst failed: {exc}")
    else:
        errors.append(
            f"council analyst skipped: only {len(blocks)} member(s) "
            "succeeded — no cross-analysis possible",
        )
    if council is None or not any(council.values()):
        council = degraded_council(member_results)
    tables_md = render_markdown_tables(council, member_names)
    # Fourth table — only rendered when peer review actually ran.
    ranking_md = render_ranking_table(peer_ranking, peer_reviews)
    if ranking_md:
        tables_md = f"{tables_md}\n\n{ranking_md}"

    # 2) Synthesizer — combined answer / executive synthesis.
    synth_system = (
        COUNCIL_SYNTH_DEEP_SYSTEM if mode == "deep"
        else COUNCIL_SYNTH_LIGHT_SYSTEM
    )
    try:
        synthesis = await chat_with_retries(
            llm,
            system_prompt=synth_system,
            user_prompt=build_synth_prompt(
                question, council, blocks, mode, peer_ranking, peer_reviews,
            ),
            timeout_s=900.0,
        )
    except RuntimeError as exc:
        errors.append(f"council synthesizer failed: {exc}")
        summaries = [
            f"**{b['name']}**: {b['content'][:800]}" for b in blocks
        ]
        synthesis = (
            "*Synthesizer unavailable — per-member summaries below.*\n\n"
            + "\n\n".join(summaries)
        )

    report = _assemble_report(
        question=question,
        mode=mode,
        synthesis=synthesis,
        tables_md=tables_md,
        member_results=member_results,
    )

    logger.info(
        "council synthesis (task=%s, mode=%s): %d agreements, "
        "%d disagreements, %d unique",
        ctx.task_id, mode,
        len(council["agreements"]), len(council["disagreements"]),
        len(council["unique"]),
    )
    return {
        "council": council,
        "council_tables_md": tables_md,
        "synthesis": synthesis,
        "report": report,
        "final_content": report,
        "errors": errors,
        "current_phase": "council_synthesis",
    }


__all__ = ["council_synthesis_node"]
