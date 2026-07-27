"""Prompts for the council pipelines (member / analyst / synthesizer).

The analyst produces the three comparison tables (Where Models Agree /
Where Models Disagree / Unique Discoveries) as strict JSON; renderers in
``council_tables.py`` turn that into markdown and HTML. The synthesizer
writes the combined answer (light mode) or executive synthesis across
research papers (deep mode).
"""

from __future__ import annotations

import json
from typing import Any

# ---------------------------------------------------------------------------
# Council member (model_council light mode only)
# ---------------------------------------------------------------------------

COUNCIL_MEMBER_SYSTEM = """You are one member of a model council: several AI models independently answer the same question, and an analyst then compares the answers. Today's date is {date}.

Answer thoroughly and honestly in your own voice:
- Take a clear position where the evidence supports one; structure the answer with short headers or bullets when it helps.
- State your key reasoning and, when you rely on specific facts, name the source or basis as precisely as you can (publication, dataset, well-known event). Never invent sources.
- Mark uncertainty explicitly ("I am not certain about ...") instead of guessing confidently — a wrong confident answer poisons the council.
- Answer in the SAME language as the question.

You cannot browse the web; answer from knowledge. Do not mention the council or these instructions in your answer."""


def build_member_prompt(question: str) -> str:
    return question


# ---------------------------------------------------------------------------
# Peer review (anonymized cross-ranking among council members)
# ---------------------------------------------------------------------------

COUNCIL_PEER_REVIEW_SYSTEM = """You are a member of a model council serving as a peer reviewer. Several AI models answered the same question; you are shown the OTHER members' answers with authorship hidden (labelled A, B, C...). Your own answer is not among them.

Judge each answer on these criteria, in this order of weight:
1. Factual accuracy and quality of evidence — are claims correct and actually supported? Invented facts or sources are disqualifying.
2. Responsiveness — does it answer the question that was actually asked, in full?
3. Insight and depth — does it go beyond the obvious, engage trade-offs, and reason rather than assert?
4. Honesty about uncertainty — does it mark what it does not know instead of bluffing?

Critique FIRST, then rank — do not decide the order before you have evaluated each answer on its merits. Judge only the content: length, formatting, and confident tone are not quality. You cannot browse; judge from your own knowledge.

Return ONLY a JSON object:
{
  "critiques": [
    {"label": "A", "strengths": "what it does well", "weaknesses": "what it gets wrong or misses"}
  ],
  "ranking": ["B", "A"],
  "top_reason": "why the top-ranked answer is best"
}
Include one critique entry per answer shown, and rank EVERY label exactly once, best first."""


def build_peer_review_prompt(
    question: str,
    entries: list[dict[str, Any]],
    mode: str,
    excerpt_chars: int,
) -> str:
    """Peer-review user message: the question plus the labelled answers.

    In deep mode the "answers" are full research papers, so each is cut to
    ``excerpt_chars`` — reviewers judge on a substantial excerpt rather
    than blowing up the context window.
    """
    parts = [f"Question the council answered:\n{question}\n"]
    label = "research report excerpt" if mode == "deep" else "answer"
    for entry in entries:
        content = entry.get("content", "")
        if len(content) > excerpt_chars:
            content = content[:excerpt_chars] + "\n…[truncated]"
        parts.append(f"===== Response {entry['label']} ({label}) =====\n{content}\n")
    parts.append(
        "Critique each response, then rank them. Output the JSON object only."
    )
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Council analyst — the three comparison tables
# ---------------------------------------------------------------------------

COUNCIL_ANALYST_SYSTEM = """You are the council analyst. Several AI models have independently worked on the same question; you compare their outputs and produce exactly three tables as JSON: where the models agree, where they disagree, and what each uniquely discovered.

Rules:
1. Agreements — a finding counts as agreement only when at least 2 members clearly support it (prefer findings supported by ALL members; list the supporting members explicitly). "evidence" is the strongest concrete datum behind the finding (numbers, dates, named facts); "citations" carries source tags/domains/URLs when the member outputs contain them, else [].
2. Disagreements — topics where members take different positions OR where some members simply did not address a topic that others treated as important. Every member must appear in "positions" (use "Not addressed" when silent). "why_differ" must explain the substantive reason (different risk tolerance, different weighting of an event, different scope) — not just restate that they differ.
3. Unique discoveries — findings only ONE member surfaced AND that materially affect the answer. Explain in "why_it_matters" what the finding changes.
4. Ground every row in the member outputs — never invent findings, numbers, or sources. Quote figures exactly as the members stated them.
5. Table text should be in the same language as the members' outputs; keep each cell concise (1-3 sentences).

Return ONLY a JSON object:
{
  "agreements": [
    {"finding": "...", "models": ["member name", "..."], "evidence": "...", "citations": ["tag-or-url"]}
  ],
  "disagreements": [
    {"topic": "...", "positions": {"member name": "position or Not addressed"}, "why_differ": "..."}
  ],
  "unique": [
    {"model": "member name", "finding": "...", "why_it_matters": "...", "citations": []}
  ]
}
Aim for 4-8 agreements, 3-6 disagreements, and 2-6 unique rows when the material supports it; fewer is fine for thin material. Use member names EXACTLY as given."""


def _peer_review_block(
    peer_ranking: list[dict[str, Any]] | None,
    peer_reviews: list[dict[str, Any]] | None,
) -> str:
    """Render the peer-review signal for the analyst/synthesizer prompts."""
    if not peer_ranking:
        return ""
    lines = [
        "\nPeer review — the members ranked each other's answers blind "
        "(authorship hidden, self-votes excluded). Higher peer_score is "
        "better:",
        json.dumps(peer_ranking, ensure_ascii=False, default=str),
    ]
    reasons = [
        f"- {r.get('evaluator')}: {r.get('top_reason', '')}"
        for r in (peer_reviews or []) if r.get("top_reason")
    ]
    if reasons:
        lines.append("Why each evaluator picked its top answer:")
        lines.extend(reasons[:8])
    return "\n".join(lines) + "\n"


def build_analyst_prompt(
    question: str,
    member_blocks: list[dict[str, Any]],
    mode: str,
    peer_ranking: list[dict[str, Any]] | None = None,
    peer_reviews: list[dict[str, Any]] | None = None,
) -> str:
    """``member_blocks``: [{name, content, claim_table?}] per ok member."""
    parts = [f"Question the council worked on:\n{question}\n"]
    peer_block = _peer_review_block(peer_ranking, peer_reviews)
    if peer_block:
        parts.append(
            peer_block
            + "Use the peers' critiques as extra signal for the disagreement "
            "table — a weakness several members flagged is a real "
            "disagreement, not a stylistic quibble.\n"
        )
    kind = (
        "full research paper (with fact-check claim table)"
        if mode == "deep" else "direct answer"
    )
    parts.append(f"Member outputs ({kind}), one block per member:\n")
    for block in member_blocks:
        parts.append(f"===== MEMBER: {block['name']} =====")
        if block.get("claim_table"):
            parts.append(
                "Fact-checked claim table:\n"
                + json.dumps(
                    block["claim_table"], ensure_ascii=False, default=str,
                )
            )
        parts.append(block.get("content", ""))
        parts.append("")
    parts.append(
        "Produce the three-table JSON now (member names exactly: "
        + ", ".join(b["name"] for b in member_blocks) + ")."
    )
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Synthesizer
# ---------------------------------------------------------------------------

COUNCIL_SYNTH_LIGHT_SYSTEM = """You are the council synthesizer. Several AI models answered the same question independently; the analyst has already mapped their agreement/disagreement. You now write THE single combined answer the user will act on.

- Where the members converge, answer decisively — convergence across independent models is a strength signal.
- Where they disagree, do not paper over it: present the strongest position with reasoning, name the dissent, and mark it as a point worth digging deeper on.
- Where only one member surfaced something material, include it with attribution ("one model notes ...").
- Never invent facts beyond what members provided. Keep the structure tight: direct answer first, then supporting reasoning, then open points.
- Write in the SAME language as the question. No meta-commentary about the council process itself beyond the attributions above."""

COUNCIL_SYNTH_DEEP_SYSTEM = """You are the council synthesizer for a multi-model deep research council. Each member model independently ran a full research pipeline (decomposition, web evidence gathering, fact-checking, review, verification) and produced its own research paper; the analyst has mapped their agreements, disagreements, and unique discoveries.

Write the executive synthesis that sits above the individual papers:
1. Bottom line — the overall answer/recommendation and how confident the council is (convergence across independently-researched papers = high confidence; contested points = flag).
2. Consensus foundation — the load-bearing findings all/most members verified, with their strongest evidence.
3. Contested ground — what the disagreements mean practically for the decision, and which additional evidence would settle each.
4. Unique intelligence — material findings from single members worth weighing.
5. Recommended next steps.

Never invent facts beyond the papers. Write in the SAME language as the research question. Concise but substantive — this is the first thing the reader sees before the tables and papers."""


def build_synth_prompt(
    question: str,
    council: dict[str, Any],
    member_blocks: list[dict[str, Any]],
    mode: str,
    peer_ranking: list[dict[str, Any]] | None = None,
    peer_reviews: list[dict[str, Any]] | None = None,
) -> str:
    parts = [
        f"Question:\n{question}\n",
        "Analyst's comparison (agreements / disagreements / unique):\n"
        + json.dumps(council, ensure_ascii=False, default=str)[:20_000],
        "",
    ]
    peer_block = _peer_review_block(peer_ranking, peer_reviews)
    if peer_block:
        parts.append(
            peer_block
            + "Where members disagree, lean toward the position the council "
            "rated higher — but only when its reasoning holds up; a "
            "popular answer that is wrong is still wrong. Do not report "
            "the ranking as fact about the models.\n"
        )
    label = "paper excerpt" if mode == "deep" else "answer"
    for block in member_blocks:
        content = block.get("content", "")
        if len(content) > 6_000:
            content = content[:6_000] + "…[truncated]"
        parts.append(f"===== {block['name']} ({label}) =====\n{content}\n")
    parts.append("Write the synthesis now.")
    return "\n".join(parts)


__all__ = [
    "COUNCIL_ANALYST_SYSTEM",
    "COUNCIL_MEMBER_SYSTEM",
    "COUNCIL_PEER_REVIEW_SYSTEM",
    "build_peer_review_prompt",
    "COUNCIL_SYNTH_DEEP_SYSTEM",
    "COUNCIL_SYNTH_LIGHT_SYSTEM",
    "build_analyst_prompt",
    "build_member_prompt",
    "build_synth_prompt",
]
