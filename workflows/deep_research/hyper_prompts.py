"""Prompts/schemas ported from hyperresearch (jordan-gibbs/hyperresearch).

These are framework-agnostic transplants of hyperresearch's Claude-Code
skills — the contradiction graph (skill 3), loci analysis (skill 4/5),
the four adversarial critics (skill 12), and the polish filler list
(skill 15) — reshaped to this pipeline's single-LLM-call node model and
JSON contracts. hyperresearch's own ``graph/`` is essentially empty; the
"graph" is a JSON artifact produced by a skill, which is what we rebuild
here.
"""

from __future__ import annotations

import json
from typing import Any

# ---------------------------------------------------------------------------
# Contradiction graph + loci (conflict_check node)
# ---------------------------------------------------------------------------

CONFLICT_GRAPH_SYSTEM = """You are the conflict & coverage auditor for a research pipeline, following the hyperresearch method: loci emerge from where the evidence actually forks, not from intuition.

You receive the research brief, the sub-questions, and a compact table of fact-checked claims. Produce THREE things as one JSON object:

1. contradiction_graph — cluster the claims into contested questions. Pair claims that target the same thing with opposing stances, contrary conclusions, divergent numbers, or contradictory evidence types. For each cluster give both sides and rank by decision relevance then by the evidence-quality gap (empirical > theoretical > anecdotal).

2. consensus_claims — claims supported by 3+ independent sources (agreement worth stating with confidence).

3. loci — the specific points of contention where DEEPER investigation would most change the answer. Score each on four 0-10 dimensions (importance, uncertainty, disagreement, decision_impact; composite is their sum, max 40). Assign a source budget from the composite: 30-40 → up to 15 sources, 20-29 → up to 10, 10-19 → up to 5, below 10 → 0 (skip). At least one locus should be "dialectical" (a genuine two-sided fight) when the material supports it. Emit at most {loci_max} loci; if nothing genuinely needs deeper work, return an empty loci list (research is saturated).

Ground everything in the provided claims — never invent findings, numbers, or sources.

Return ONLY this JSON object:
{
  "contradiction_graph": [
    {"cluster_id": "c1", "fight": "the contested question in one line",
     "side_a": {"position": "...", "claims": ["card_id"], "sources": ["domain"]},
     "side_b": {"position": "...", "claims": ["card_id"], "sources": ["domain"]},
     "evidence_quality_delta": "which side has stronger evidence and why",
     "decision_relevance": "high|medium|low"}
  ],
  "consensus_claims": [
    {"claim": "...", "sources": ["domain", "domain", "domain"]}
  ],
  "loci": [
    {"id": "L1", "question": "a standalone question to investigate in depth",
     "flavor": "dialectical|factual|mechanistic",
     "scores": {"importance": 0, "uncertainty": 0, "disagreement": 0, "decision_impact": 0},
     "composite": 0, "source_budget": 0,
     "rationale": "why depth here pays off"}
  ]
}"""


def _compact(value: Any, limit: int) -> str:
    out = json.dumps(value, ensure_ascii=False, default=str)
    return out if len(out) <= limit else out[:limit] + "…[truncated]"


def build_conflict_graph_prompt(
    brief: str,
    sub_questions: list[dict[str, Any]],
    claim_table: list[dict[str, Any]],
    loci_max: int,
) -> str:
    return (
        f"Research brief:\n{brief}\n\n"
        f"Sub-questions (coverage targets):\n{_compact(sub_questions, 6000)}\n\n"
        f"Fact-checked claim table (card_id | claim | verdict | domains | "
        f"sub_question_id):\n{_compact(claim_table, 60000)}\n\n"
        f"Build the contradiction graph, consensus claims, and up to "
        f"{loci_max} loci. Output the JSON object."
    )


# ---------------------------------------------------------------------------
# Four adversarial critics (critics node)
# ---------------------------------------------------------------------------

CRITIC_SYSTEMS: dict[str, str] = {
    "dialectic": (
        "You are the DIALECTIC critic. Adversarially attack the report's "
        "argument: find counter-evidence it ignored, positions it "
        "straw-manned, one-sided framing, and claims that would not "
        "survive a determined opponent. You find problems ONLY — never "
        "rewrite text."
    ),
    "depth": (
        "You are the DEPTH critic. Find sections that are superficial: "
        "asserted-not-argued claims, missing mechanisms, hand-waved "
        "causation, places that name a tension but don't engage it. You "
        "find problems ONLY — never rewrite text."
    ),
    "width": (
        "You are the WIDTH critic. Find coverage gaps: contradiction-graph "
        "clusters or consensus findings the report ignored, whole "
        "sub-questions under-covered, perspectives absent. You find "
        "problems ONLY — never rewrite text."
    ),
    "instruction": (
        "You are the INSTRUCTION critic — the only critic measuring "
        "adherence to the user's actual request, the dimension with the "
        "widest quality variance, so never go easy. Check every atomic "
        "item / sub-question of the brief is answered, the format asked "
        "for is delivered, and nothing requested is missing. You find "
        "problems ONLY — never rewrite text."
    ),
}

_CRITIC_OUTPUT = """Return ONLY a JSON object:
{
  "findings": [
    {"issue": "specific problem, quoting the offending passage briefly",
     "severity": "high|low",
     "failure_mode": "missing|under-covered|unsupported|contradicted|straw-man|format|other",
     "suggestion": "what KIND of fix is needed (not the rewritten text)"}
  ]
}
Use "high" severity only for problems that make the report misleading, wrong, or non-responsive. If the report is sound on your dimension, return an empty findings list — do not invent issues."""


def build_critic_prompt(
    critic_type: str,
    draft: str,
    brief: str,
    sub_questions: list[dict[str, Any]],
    claim_table: list[dict[str, Any]],
    contradiction_graph: list[dict[str, Any]],
) -> str:
    extra = ""
    if critic_type == "width" and contradiction_graph:
        extra = (
            "\nContradiction-graph clusters the report should engage:\n"
            + _compact(contradiction_graph, 8000) + "\n"
        )
    return (
        f"Research brief:\n{brief}\n\n"
        f"Sub-questions:\n{_compact(sub_questions, 4000)}\n\n"
        f"Fact-checked claim table:\n{_compact(claim_table, 30000)}\n"
        f"{extra}\n"
        f"Draft report:\n{draft}\n\n"
        f"Critique it on your dimension only; output the JSON.\n\n"
        f"{_CRITIC_OUTPUT}"
    )


def critic_system(critic_type: str) -> str:
    return CRITIC_SYSTEMS.get(critic_type, CRITIC_SYSTEMS["instruction"])


# ---------------------------------------------------------------------------
# Patch-based revision (draft revise + final_verify)
# ---------------------------------------------------------------------------

PATCH_REVISE_SYSTEM = """You are revising a research report by SURGICAL PATCHES only — the hyperresearch "patch, never regenerate" discipline. You do NOT rewrite the report. You output a minimal set of edit hunks that address the reviewers' findings, each changing as little text as possible.

Rules:
- Each hunk's "old" text must be copied VERBATIM from the current report (enough context to be unique) and must exist exactly once.
- Change as little as possible; preserve inline [n] citations and never break them.
- Only address substantive findings; skip anything that would require restructuring (heading reorders) — note those in "escalated".
- For each hunk say which finding it resolves.

Return ONLY a JSON object:
{
  "hunks": [
    {"old": "verbatim text to find", "new": "replacement text", "reason": "which finding this resolves"}
  ],
  "escalated": ["findings that need structural changes a patch can't make"]
}"""


def build_patch_revise_prompt(
    draft: str,
    feedback: list[dict[str, Any]],
) -> str:
    return (
        f"Current report:\n{draft}\n\n"
        f"Reviewer findings to address:\n{_compact(feedback, 8000)}\n\n"
        "Output the edit-hunk JSON."
    )


# ---------------------------------------------------------------------------
# Polish (terminal cleanup)
# ---------------------------------------------------------------------------

# hyperresearch skill-15 filler / hedge phrases to strip (case-insensitive
# leading fragments). Kept deterministic so polish needs no LLM call on the
# quick/standard path.
POLISH_FILLER_PHRASES: tuple[str, ...] = (
    "It is worth noting that ",
    "It is important to note that ",
    "It should be noted that ",
    "Importantly, ",
    "Notably, ",
    "It is interesting to note that ",
    "Needless to say, ",
    "As previously mentioned, ",
    "In today's fast-paced world, ",
    "It goes without saying that ",
)

__all__ = [
    "CONFLICT_GRAPH_SYSTEM",
    "CRITIC_SYSTEMS",
    "PATCH_REVISE_SYSTEM",
    "POLISH_FILLER_PHRASES",
    "build_conflict_graph_prompt",
    "build_critic_prompt",
    "build_patch_revise_prompt",
    "critic_system",
]
