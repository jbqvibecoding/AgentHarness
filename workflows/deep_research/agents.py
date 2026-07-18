"""AgentDefinitions for the ``deep_research`` pipeline roles.

Role ids carry a ``dr_`` prefix so they can't collide with roles from
other workflows in the shared AgentRegistry. System prompts live in
``prompts.py`` and are date-templated at call time — the registry copy
here is the static persona (nodes format ``{date}`` themselves).
"""

from __future__ import annotations

from agent_harness.models.agent_definition import AgentDefinition

from workflows.deep_research import prompts as P

_WEB_TOOLS = ["web_search", "web_fetch"]

PLANNER_DEF = AgentDefinition(
    role_id="dr_planner",
    display_name="Research Planner",
    system_prompt=P.PLANNER_SYSTEM,
    allowed_tools=[],
    color="#6366f1",
    icon="map",
    description="Decomposes the question into a brief plus independent sub-questions.",
)

RESEARCHER_DEF = AgentDefinition(
    role_id="dr_researcher",
    display_name="Evidence Researcher",
    system_prompt=P.RESEARCHER_SYSTEM,
    allowed_tools=list(_WEB_TOOLS),
    color="#10b981",
    icon="search",
    description="ReAct sub-agent gathering cited evidence cards for one sub-question.",
)

FACT_CHECKER_DEF = AgentDefinition(
    role_id="dr_fact_checker",
    display_name="Fact Checker",
    system_prompt=P.FACT_CHECKER_SYSTEM,
    allowed_tools=[*_WEB_TOOLS, "vault_get", "vault_search"],
    color="#f59e0b",
    icon="shield-check",
    description="Re-fetches cited sources and cross-references claims; issues verdicts.",
)

CONFLICT_CHECKER_DEF = AgentDefinition(
    role_id="dr_conflict_checker",
    display_name="Conflict & Gap Auditor",
    system_prompt=P.CONFLICT_SYSTEM,
    allowed_tools=[],
    color="#ef4444",
    icon="git-compare",
    description="Detects cross-claim contradictions and coverage gaps; emits gap questions.",
)

WRITER_DEF = AgentDefinition(
    role_id="dr_writer",
    display_name="Report Writer",
    system_prompt=P.WRITER_SYSTEM,
    allowed_tools=[],
    color="#3b82f6",
    icon="pen",
    description="Writes/revises the cited markdown report from verified evidence.",
)

REVIEWER_DEF = AgentDefinition(
    role_id="dr_reviewer",
    display_name="Draft Reviewer",
    system_prompt=P.REVIEWER_SYSTEM,
    allowed_tools=[],
    color="#a855f7",
    icon="eye",
    description="Critic: finds problems only (coverage, citations, discipline); approve|revise.",
)

VERIFIER_DEF = AgentDefinition(
    role_id="dr_verifier",
    display_name="Global Verifier",
    system_prompt=P.VERIFIER_SYSTEM,
    allowed_tools=["web_search", "vault_get", "vault_search"],
    color="#14b8a6",
    icon="badge-check",
    description="Final gate: per-claim audit vs verdicts, surgical corrections, honest summary.",
)

# Council roles (deep_council_research / model_council). Not part of
# ALL_AGENT_DEFS — they ride the council specs' agent_definitions instead.
COUNCIL_ANALYST_DEF = AgentDefinition(
    role_id="dr_council_analyst",
    display_name="Council Analyst",
    system_prompt="",  # council_prompts.COUNCIL_ANALYST_SYSTEM at call time
    allowed_tools=[],
    color="#8b5cf6",
    icon="table",
    description="Compares member-model outputs into agree/disagree/unique tables.",
)

COUNCIL_SYNTH_DEF = AgentDefinition(
    role_id="dr_council_synthesizer",
    display_name="Council Synthesizer",
    system_prompt="",
    allowed_tools=[],
    color="#0ea5e9",
    icon="merge",
    description="Merges member-model outputs into one combined answer/synthesis.",
)

COUNCIL_AGENT_DEFS: list[AgentDefinition] = [
    COUNCIL_ANALYST_DEF,
    COUNCIL_SYNTH_DEF,
]

ALL_AGENT_DEFS: list[AgentDefinition] = [
    PLANNER_DEF,
    RESEARCHER_DEF,
    FACT_CHECKER_DEF,
    CONFLICT_CHECKER_DEF,
    WRITER_DEF,
    REVIEWER_DEF,
    VERIFIER_DEF,
]

__all__ = [
    "ALL_AGENT_DEFS",
    "COUNCIL_AGENT_DEFS",
    "COUNCIL_ANALYST_DEF",
    "COUNCIL_SYNTH_DEF",
    "PLANNER_DEF",
    "RESEARCHER_DEF",
    "FACT_CHECKER_DEF",
    "CONFLICT_CHECKER_DEF",
    "WRITER_DEF",
    "REVIEWER_DEF",
    "VERIFIER_DEF",
]
