"""model_council — lightweight multi-model council (no deep research).

The same question is answered independently by every configured council
member model (single direct completion, no tools), then the shared
council analyst maps agreement/disagreement/unique findings and the
synthesizer produces one combined answer. Use for quick questions,
brainstorming, and cross-validation; for verified web research use the
``deep_research`` / ``deep_council_research`` pipelines instead.

Run standalone:
``uv run python -m workflows.deep_research.run --pipeline model_council --question "..."``
"""

from __future__ import annotations

from agent_harness.core.runtime.registries.workflows import WorkflowContext

from workflows.deep_research.agents import COUNCIL_AGENT_DEFS
from workflows.model_council.spec import MODEL_COUNCIL_SPEC


def register(ctx: WorkflowContext) -> None:
    for agent_def in COUNCIL_AGENT_DEFS:
        ctx.register_agent(agent_def)
    ctx.register_pipeline(MODEL_COUNCIL_SPEC)
