"""Pipeline spec: ``model_council`` — lightweight multi-model council.

``member_answers`` fans the question out to every configured member model
(one direct completion each, in parallel); ``council_synthesis`` (shared
with deep_council_research) produces the three comparison tables and the
combined answer.
"""

from __future__ import annotations

from agent_harness.models.pipeline_spec import (
    NodeDefinition,
    PipelineSpec,
    TransitionSpec,
)

from workflows.deep_research.agents import COUNCIL_AGENT_DEFS

MODEL_COUNCIL_SPEC = PipelineSpec(
    pipeline_id="model_council",
    name="Model Council (multi-model answers)",
    description=(
        "Runs the same question across all configured council member "
        "models in parallel (direct answers, no web research), then a "
        "council analyst maps agreements/disagreements/unique findings "
        "and a synthesizer merges them into one combined answer."
    ),
    entry_point="member_answers",
    terminal_nodes=["council_synthesis"],
    state_type="workflows.deep_research.state.CouncilState",
    agent_definitions=COUNCIL_AGENT_DEFS,
    nodes=[
        NodeDefinition(
            node_id="member_answers",
            role_id="dr_council_analyst",
            node_function=(
                "workflows.model_council.nodes.answers.member_answers_node"
            ),
            display_label="Asking each council model in parallel",
            output_fields=["members", "member_results"],
        ),
        NodeDefinition(
            node_id="peer_review",
            role_id="dr_council_analyst",
            node_function=(
                "workflows.deep_research.nodes.peer_review.peer_review_node"
            ),
            display_label="Members reviewing each other's answers blind",
            output_fields=["peer_reviews", "peer_ranking"],
        ),
        NodeDefinition(
            node_id="council_synthesis",
            role_id="dr_council_synthesizer",
            node_function=(
                "workflows.deep_research.nodes.council_synthesis"
                ".council_synthesis_node"
            ),
            display_label="Comparing answers and synthesizing one response",
            output_fields=["council", "council_tables_md", "synthesis",
                           "report", "final_content"],
        ),
    ],
    transitions=[
        TransitionSpec(from_phase="member_answers", to_phase="peer_review"),
        TransitionSpec(from_phase="peer_review", to_phase="council_synthesis"),
        TransitionSpec(from_phase="council_synthesis", to_phase="__END__"),
    ],
)

__all__ = ["MODEL_COUNCIL_SPEC"]
