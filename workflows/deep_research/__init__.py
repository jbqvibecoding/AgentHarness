"""deep_research — multi-agent deep research workflow.

Seven roles with internal checks and balances: a planner decomposes the
question; parallel researchers gather cited evidence; fact-checkers
re-fetch sources and issue per-claim verdicts; a conflict auditor hunts
contradictions and coverage gaps (looping research while gaps remain); a
writer drafts, a critic reviews (bounded revision loop); and a global
verifier performs the terminal audit that gates the final report.

Run standalone:  ``uv run python -m workflows.deep_research.run --question "..."``
"""

from __future__ import annotations

from agent_harness.core.runtime.registries.workflows import WorkflowContext

from workflows.deep_research.agents import ALL_AGENT_DEFS
from workflows.deep_research.spec import DEEP_RESEARCH_SPEC


def register(ctx: WorkflowContext) -> None:
    for agent_def in ALL_AGENT_DEFS:
        ctx.register_agent(agent_def)
    ctx.register_pipeline(DEEP_RESEARCH_SPEC)
