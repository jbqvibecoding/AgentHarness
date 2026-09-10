"""Pipeline spec: ``deep_research``.

Multi-agent deep research with internal checks and balances:

    plan → research_fanout → fact_check → conflict_check
        ├─(gaps & iterations left)→ research_fanout   (re-research loop)
        └─→ draft → review ─┬─(revise & budget left)→ draft   (revision loop)
                            └─→ final_verify → citation_audit → polish → END

Parallelism happens INSIDE ``research_fanout`` / ``fact_check`` via
``asyncio.gather`` over reentrant ``run_agent_loop`` sub-agents — MiniDAG
itself walks nodes sequentially. Condition functions return literal
``to_phase`` node ids (graph-builder contract).
"""

from __future__ import annotations

from agent_harness.models.pipeline_spec import (
    CompressionConfig,
    NodeDefinition,
    PipelineSpec,
    TransitionSpec,
)

from workflows.deep_research.agents import ALL_AGENT_DEFS, COUNCIL_AGENT_DEFS

_NODES_PKG = "workflows.deep_research.nodes"
_CONDITIONS = "workflows.deep_research.conditions"

DEEP_RESEARCH_SPEC = PipelineSpec(
    pipeline_id="deep_research",
    name="Deep Research (multi-agent)",
    description=(
        "Decomposes a research question, fans out parallel researcher "
        "sub-agents for cited evidence, fact-checks claims against their "
        "sources, audits conflicts/gaps (with bounded re-research), then "
        "drafts, reviews, and globally verifies a cited report."
    ),
    entry_point="plan",
    terminal_nodes=["polish"],
    state_type="workflows.deep_research.state.DeepResearchState",
    agent_definitions=ALL_AGENT_DEFS,
    nodes=[
        NodeDefinition(
            node_id="plan",
            role_id="dr_planner",
            node_function=f"{_NODES_PKG}.plan.plan_node",
            display_label="Planning research and decomposing the question",
            output_fields=["research_brief", "sub_questions"],
        ),
        NodeDefinition(
            node_id="research_fanout",
            role_id="dr_researcher",
            node_function=f"{_NODES_PKG}.research.research_fanout_node",
            display_label="Researching sub-questions in parallel",
            output_fields=["evidence_cards", "research_notes"],
        ),
        NodeDefinition(
            node_id="fact_check",
            role_id="dr_fact_checker",
            node_function=f"{_NODES_PKG}.fact_check.fact_check_node",
            display_label="Fact-checking claims against their sources",
            output_fields=["fact_check_results"],
        ),
        NodeDefinition(
            node_id="conflict_check",
            role_id="dr_conflict_checker",
            node_function=f"{_NODES_PKG}.conflict_check.conflict_check_node",
            display_label="Auditing conflicts and coverage gaps",
            compression=CompressionConfig(
                max_field_tokens={
                    "evidence_cards": 30_000,
                    "fact_check_results": 8_000,
                },
            ),
            output_fields=["conflicts", "gap_questions"],
        ),
        NodeDefinition(
            node_id="draft",
            role_id="dr_writer",
            node_function=f"{_NODES_PKG}.draft.draft_node",
            display_label="Writing the report draft",
            compression=CompressionConfig(
                max_field_tokens={"evidence_cards": 40_000},
            ),
            output_fields=["draft_report"],
        ),
        NodeDefinition(
            node_id="review",
            role_id="dr_reviewer",
            node_function=f"{_NODES_PKG}.review.review_node",
            display_label="Reviewing the draft",
            output_fields=["review_verdict", "review_feedback"],
        ),
        NodeDefinition(
            node_id="final_verify",
            role_id="dr_verifier",
            node_function=f"{_NODES_PKG}.final_verify.final_verify_node",
            display_label="Final global verification",
            output_fields=["report", "final_content", "verification_summary"],
        ),
        NodeDefinition(
            node_id="citation_audit",
            role_id="dr_verifier",
            node_function=f"{_NODES_PKG}.citation_audit.citation_audit_node",
            display_label="Auditing citations and figures",
            output_fields=[
                "report", "final_content", "references",
                "citation_audit", "numeric_grounding", "citation_mapping",
            ],
        ),
        NodeDefinition(
            node_id="polish",
            role_id="dr_reviewer",
            node_function=f"{_NODES_PKG}.polish.polish_node",
            display_label="Polishing the final report",
            output_fields=["report", "final_content"],
        ),
    ],
    transitions=[
        TransitionSpec(from_phase="plan", to_phase="research_fanout"),
        TransitionSpec(from_phase="research_fanout", to_phase="fact_check"),
        TransitionSpec(from_phase="fact_check", to_phase="conflict_check"),
        # Both edges out of conflict_check share one condition function
        # which returns the target node id.
        TransitionSpec(
            from_phase="conflict_check",
            to_phase="research_fanout",
            condition=f"{_CONDITIONS}.route_after_conflict_check",
        ),
        TransitionSpec(from_phase="conflict_check", to_phase="draft"),
        TransitionSpec(from_phase="draft", to_phase="review"),
        TransitionSpec(
            from_phase="review",
            to_phase="draft",
            condition=f"{_CONDITIONS}.route_after_review",
        ),
        TransitionSpec(from_phase="review", to_phase="final_verify"),
        TransitionSpec(from_phase="final_verify", to_phase="citation_audit"),
        TransitionSpec(from_phase="citation_audit", to_phase="polish"),
        TransitionSpec(from_phase="polish", to_phase="__END__"),
    ],
)

# ---------------------------------------------------------------------------
# Deep Council Research — one full deep_research run per council member,
# then cross-model analysis (agree / disagree / unique) + synthesis.
# ---------------------------------------------------------------------------

DEEP_COUNCIL_RESEARCH_SPEC = PipelineSpec(
    pipeline_id="deep_council_research",
    name="Deep Council Research (multi-model)",
    description=(
        "Runs the full deep_research pipeline once per council member "
        "model (COUNCIL_MODEL_* slots), producing one research paper per "
        "model, then a council analyst maps where the models agree, "
        "disagree, and what each uniquely discovered, and a synthesizer "
        "writes the combined executive report."
    ),
    entry_point="council_research_fanout",
    terminal_nodes=["council_synthesis"],
    state_type="workflows.deep_research.state.CouncilState",
    agent_definitions=COUNCIL_AGENT_DEFS,
    nodes=[
        NodeDefinition(
            node_id="council_research_fanout",
            role_id="dr_council_analyst",
            node_function=(
                f"{_NODES_PKG}.council_fanout.council_research_fanout_node"
            ),
            display_label="Running full research once per council model",
            output_fields=["members", "member_results"],
        ),
        NodeDefinition(
            node_id="peer_review",
            role_id="dr_council_analyst",
            node_function=f"{_NODES_PKG}.peer_review.peer_review_node",
            display_label="Members reviewing each other's reports blind",
            output_fields=["peer_reviews", "peer_ranking"],
        ),
        NodeDefinition(
            node_id="council_synthesis",
            role_id="dr_council_synthesizer",
            node_function=(
                f"{_NODES_PKG}.council_synthesis.council_synthesis_node"
            ),
            display_label="Comparing models and synthesizing the council report",
            output_fields=["council", "council_tables_md", "synthesis",
                           "report", "final_content"],
        ),
    ],
    transitions=[
        TransitionSpec(
            from_phase="council_research_fanout",
            to_phase="peer_review",
        ),
        TransitionSpec(from_phase="peer_review", to_phase="council_synthesis"),
        TransitionSpec(from_phase="council_synthesis", to_phase="__END__"),
    ],
)

__all__ = ["DEEP_COUNCIL_RESEARCH_SPEC", "DEEP_RESEARCH_SPEC"]
