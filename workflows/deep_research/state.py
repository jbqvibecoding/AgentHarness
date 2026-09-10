"""Pipeline state for the ``deep_research`` workflow.

Reducer semantics (``MiniDAGRunner._merge_state`` + ``extract_reducers``):

* ``Annotated[list, operator.add]`` fields **accumulate** — nodes return
  only the *new* items and the runner appends them. The initial state must
  seed these fields as ``[]`` or the reducer never fires (reducers only
  apply when the key already exists in state).
* Everything else is last-write-wins — nodes return the full new value.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict


class DeepResearchState(TypedDict, total=False):
    """State shared across all deep_research pipeline nodes.

    Dict shapes (documented here; plain dicts on the wire):

    * ``sub_questions`` item: ``{id, question, rationale, status, iteration}``
      with ``status`` in ``pending | researched | failed``.
    * ``evidence_cards`` item: ``{card_id, sub_question_id, claim, quote,
      sources: [{url, title}], confidence: high|medium|low, flags: [str],
      iteration}``.
    * ``research_notes`` item: ``{sub_question_id, question, summary,
      status, turns_used}`` — self-contained so downstream readers don't
      need to join back to ``sub_questions``.
    * ``fact_check_results`` item: ``{card_id, verdict, corrected_claim,
      notes}`` with ``verdict`` in ``supported | partially_supported |
      unsupported | contradicted | unverified``.
    * ``conflicts`` item: ``{card_ids, description, severity,
      resolution_hint}``.
    * ``gap_questions`` item: same shape as ``sub_questions`` items
      (``id`` assigned when merged); ``[]`` means converged.
    * ``review_feedback`` item: ``{issue, severity, suggestion}``.
    * ``citation_mapping``: ``{str(citation_num): {url, title}}`` — global
      citation numbers shared by draft / final report (string keys so the
      dict survives JSON round-trips unchanged).
    """

    # Seeded by the runner / BenchmarkSession
    task_id: str
    original_question: str
    metadata: dict[str, Any]

    # plan
    research_brief: str
    sub_questions: list[dict[str, Any]]

    # research fan-out (accumulating)
    evidence_cards: Annotated[list[dict[str, Any]], operator.add]
    research_notes: Annotated[list[dict[str, Any]], operator.add]
    research_iteration: int
    citation_mapping: dict[str, dict[str, str]]

    # fact check (accumulating)
    fact_check_results: Annotated[list[dict[str, Any]], operator.add]

    # conflict / gap analysis (replaced each round)
    conflicts: list[dict[str, Any]]
    gap_questions: list[dict[str, Any]]

    # hyperresearch contradiction-graph + loci (replaced each round)
    contradiction_graph: list[dict[str, Any]]
    consensus_claims: list[dict[str, Any]]
    loci: list[dict[str, Any]]

    # evidence vault (hyperresearch): filesystem dir; None/absent = disabled
    vault_dir: str

    # patch / critic bookkeeping
    patch_log: Annotated[list[dict[str, Any]], operator.add]
    polish_log: list[dict[str, Any]]

    # writing loop
    draft_report: str
    review_verdict: str
    review_feedback: list[dict[str, Any]]
    revision_count: int

    # citation integrity (``citation_audit`` node). ``references`` is the
    # ``[{url, title}]`` whitelist in citation order — list position ``i``
    # is ``[i + 1]``. Both audit dicts are summaries, replaced wholesale.
    references: list[dict[str, str]]
    citation_audit: dict[str, Any]
    numeric_grounding: dict[str, Any]

    # terminal output — the Scheduler requires non-empty ``report`` or
    # ``final_content`` in the final state.
    verification_summary: str
    report: str
    final_content: str

    # bookkeeping
    errors: Annotated[list[str], operator.add]
    current_phase: str


#: List-typed state fields that must be seeded ``[]`` by the runner so the
#: ``operator.add`` reducers apply (reducers only fire on pre-existing keys).
REDUCED_LIST_FIELDS: tuple[str, ...] = (
    "evidence_cards",
    "research_notes",
    "fact_check_results",
    "errors",
    "patch_log",
)


class CouncilState(TypedDict, total=False):
    """State for the council pipelines (deep_council_research / model_council).

    Dict shapes:

    * ``members`` item: ``{name, slug, model}``.
    * ``member_results`` item: ``{name, slug, status: ok|failed,
      report (answer or full research paper), claim_table (deep only),
      verification_summary (deep only), error}``.
    * ``council``: the three-table dict — see ``council_tables.py``.
    """

    task_id: str
    original_question: str
    metadata: dict[str, Any]     # metadata["council_mode"] = "deep"|"light"

    members: list[dict[str, Any]]
    member_results: list[dict[str, Any]]

    # peer review (anonymized cross-ranking); empty when the stage is off
    peer_reviews: list[dict[str, Any]]   # {evaluator, ranking, critiques, top_reason}
    peer_ranking: list[dict[str, Any]]   # {model, peer_score, average_rank, ballots_counted}

    council: dict[str, Any]
    council_tables_md: str
    synthesis: str

    report: str
    final_content: str

    errors: Annotated[list[str], operator.add]
    current_phase: str


__all__ = ["CouncilState", "DeepResearchState", "REDUCED_LIST_FIELDS"]
