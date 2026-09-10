"""``swarm`` research mode — a coordinator that decides its own fan-out.

The default ``fanout`` mode spawns exactly one researcher per pending
sub-question. That is predictable and cheap, and it is also rigid: the plan
node must decide how much work each sub-question deserves *before any
searching has happened*. A sub-question that turns out to need five angles
gets one researcher; one that resolves in a single search gets the same.

Swarm mode moves that decision to where the information is. A coordinator
ReAct loop (role ``dr_coordinator``) reads the brief and the sub-questions,
then uses ``create_subagent`` / ``assign_task`` / ``collect_reports`` to
open and feed researchers as it learns what the question actually needs.
Researchers run as durable :class:`AgentBus` sessions, so they get spawn
guards, cooperative stop, and fan-in classification rather than a bare
``asyncio.gather``.

**The seven-role DAG does not change.** The checks and balances this
pipeline exists for — fact-check, conflict audit, adversarial review,
global verification, citation audit — are what make its output
trustworthy, and all of them sit downstream of this node. Swarm mode
changes only *how* the research node gathers evidence. Its output contract
is identical: ``evidence_cards`` + ``research_notes`` + a merged
``citation_mapping``, audited and looped over by ``conflict_check`` exactly
as before.

**Attribution.** The coordinator is told to name each session after the
sub-question it serves, which is how evidence gets attributed back. A
session whose name matches no known sub-question is not dropped: the
coordinator found a line of enquiry the planner missed, so it is appended
to ``sub_questions`` and recorded. Losing that would be worse than
recording it under a generated id.

**Falling back.** If the coordinator errors, or produces no evidence at
all, this node falls back to the fixed fan-out for the same round. An empty
research round is the one outcome the downstream stages cannot work with.

Not ported from FrontierAgent's ``agent_team``: its coordinator node and
sub-agent runtime (3000+ lines) are bound to that repo's bwrap sandbox,
worktrees and profile system. This is a thin coordinator over the same
generic ``AgentBus``.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from agent_harness.models.node_context import NodeContext

from workflows.deep_research.citations import merge_sources
from workflows.deep_research.config import get_cfg
from workflows.deep_research.nodes.common import profile_name, today
from workflows.deep_research.prompts import with_date

logger = logging.getLogger(__name__)

COORDINATOR_SYSTEM = """You coordinate a team of research sub-agents. Today's date is {date}.

You do NOT research yourself. You decide who researches what, then report which researcher covered what.

Your tools:
- create_subagent — open researchers. Each needs a `name` and a `system_prompt` describing its focus.
- assign_task — give a named researcher one specific, self-contained question. Non-blocking: assign several, then keep working.
- collect_reports — wait for outstanding researchers when you have nothing else to do.
- add_task / update_task — keep a visible board of what is outstanding.

Rules that matter:
- NAME EVERY SESSION AFTER THE SUB-QUESTION IT SERVES, using the exact id in brackets below — e.g. `sq2_market-size`. This is how the pipeline attributes evidence. A session covering a line of enquiry that is NOT in the list gets a descriptive name instead.
- Open a researcher per genuinely distinct angle, not mechanically one per sub-question. Two sub-questions answered from the same sources can share one researcher; one sub-question spanning three jurisdictions deserves three.
- Never run more than {max_parallel} researchers at once. Assign, collect, then open more.
- A researcher's task must be self-contained: it sees only the text you send it, never this conversation. Restate the question in full.
- Every sub-question listed below must be covered by at least one researcher before you finish.

When you have all the reports, reply in plain text with which researcher covered which sub-question. Do NOT write the research report yourself — a later stage does that."""

# Session names are model-authored, so the id is extracted rather than
# assumed: `sq2_market-size`, `sq2-market-size` and `SQ2` must all resolve.
# Underscore is an explicit separator here — ``\b`` would not treat it as
# one, so ``\bsq\d+\b`` fails on exactly the ``sq2_topic`` shape the
# coordinator is told to use.
_SQ_ID_RE = re.compile(r"(?<![a-z0-9])sq\d+(?![a-z0-9])", re.IGNORECASE)


def _coordinator_prompt(
    state: dict[str, Any], targets: list[dict[str, Any]],
) -> str:
    lines = [
        f"Research question:\n{state.get('original_question', '')}\n",
        f"Research brief:\n{state.get('research_brief', '')}\n",
        "Sub-questions that still need evidence:",
    ]
    for sq in targets:
        rationale = sq.get("rationale") or ""
        lines.append(
            f"- [{sq['id']}] {sq['question']}"
            + (f"\n      (why it matters: {rationale})" if rationale else ""),
        )
    lines.append(
        "\nOpen researchers, assign each a self-contained question, and "
        "collect their reports. Every sub-question above must be covered.",
    )
    return "\n".join(lines)


def _sq_id_from_session(name: str, known: set[str]) -> str:
    """Map a model-authored session name back to a sub-question id."""
    for match in _SQ_ID_RE.findall(name or ""):
        candidate = match.lower()
        if candidate in known:
            return candidate
    return ""


async def research_swarm_node(
    state: dict[str, Any], ctx: NodeContext,
) -> dict[str, Any]:
    """Coordinator-driven research round. Same output shape as fan-out."""
    from workflows.deep_research.nodes.research import _merge_gap_questions

    sub_questions = _merge_gap_questions(state)
    targets = [s for s in sub_questions if s.get("status") == "pending"]
    iteration = int(state.get("research_iteration", 0)) + 1
    if not targets:
        logger.info("deep_research swarm: no pending sub-questions")
        return {
            "research_iteration": iteration,
            "sub_questions": sub_questions,
            "gap_questions": [],
            "current_phase": "research",
        }

    errors: list[str] = []
    try:
        sessions = await _run_coordinator(state, ctx, targets)
    except Exception as exc:  # noqa: BLE001 — fall back rather than fail
        logger.warning(
            "deep_research swarm (task=%s) failed (%s) — falling back to the "
            "fixed fan-out for this round", ctx.task_id, exc,
        )
        return await _fallback(state, ctx, [f"swarm: {exc}"[:500]])

    delta = _harvest(state, ctx, sessions, sub_questions, iteration, errors)
    if not delta.get("evidence_cards"):
        logger.warning(
            "deep_research swarm (task=%s): no evidence — falling back",
            ctx.task_id,
        )
        return await _fallback(
            state, ctx, [*errors, "swarm: produced no evidence"],
        )
    return delta


async def _fallback(
    state: dict[str, Any], ctx: NodeContext, errors: list[str],
) -> dict[str, Any]:
    """Run the fixed fan-out for this round instead.

    The state still carries the same pending sub-questions, so the fan-out
    picks up exactly the work the swarm did not complete.
    """
    from workflows.deep_research.nodes.research import research_fanout_node

    delta = await research_fanout_node(state, ctx)
    merged = list(delta.get("errors") or [])
    merged.extend(errors)
    delta["errors"] = merged
    return delta


async def _run_coordinator(
    state: dict[str, Any], ctx: NodeContext, targets: list[dict[str, Any]],
) -> list[Any]:
    """Run the coordinator loop; return the sessions it left on the bus."""
    from agent_harness.components.agent_bus import AgentBus
    from agent_harness.core.runtime.registries import services as registry
    from agent_harness.core.runtime.resources.manager import ResourceManager

    from workflows.deep_research.observers.auto_fan_in import AutoFanInObserver
    from workflows.deep_research.observers.no_progress_guard import NoProgressGuard
    from workflows.deep_research.observers.unassigned_nudge import (
        UnassignedAgentNudge,
    )
    from workflows.deep_research.subagent import run_subagent

    max_parallel = int(get_cfg(state, "max_parallel_subagents"))
    bus = AgentBus(resource_manager=registry.get_optional(ResourceManager))
    # The orchestration tools resolve the bus from the service registry,
    # so it has to be registered before the coordinator's first tool call.
    registry.register(AgentBus, bus)

    try:
        await run_subagent(
            role_id="dr_coordinator",
            system_prompt=with_date(
                COORDINATOR_SYSTEM.replace("{max_parallel}", str(max_parallel)),
                today(),
            ),
            user_message=_coordinator_prompt(state, targets),
            max_turns=int(get_cfg(state, "coordinator_max_turns")),
            task_id=ctx.task_id,
            profile_name=profile_name(state),
            timeout_s=float(get_cfg(state, "subagent_timeout_s")) * 2,
            extra_observers=[
                AutoFanInObserver(),
                UnassignedAgentNudge(),
                NoProgressGuard(),
            ],
            state=state,
        )
        return list(bus.list_sessions_for_task(ctx.task_id))
    finally:
        try:
            await bus.cleanup_task(ctx.task_id)
        except Exception as exc:  # noqa: BLE001 — cleanup is best-effort
            logger.debug("swarm bus cleanup: %s", exc)


def _harvest(
    state: dict[str, Any],
    ctx: NodeContext,
    sessions: list[Any],
    sub_questions: list[dict[str, Any]],
    iteration: int,
    errors: list[str],
) -> dict[str, Any]:
    """Turn the coordinator's sessions into the node's standard output."""
    from workflows.deep_research.nodes.research import _parse_evidence

    by_id = {s["id"]: s for s in sub_questions}
    known = {str(i).lower() for i in by_id}
    citation_mapping = dict(state.get("citation_mapping") or {})
    new_cards: list[dict[str, Any]] = []
    new_notes: list[dict[str, Any]] = []
    extra = 0

    for session in sessions:
        report = _session_report(session)
        if not report:
            continue
        name = str(getattr(session, "name", "") or "")
        sq_id = _sq_id_from_session(name, known)
        if not sq_id:
            # A line of enquiry the planner did not list. Recording it
            # under a generated id keeps its evidence; dropping it would
            # silently discard research already paid for.
            extra += 1
            sq_id = f"swarm{extra}"
            sub_questions.append({
                "id": sq_id,
                "question": name or f"additional line of enquiry {extra}",
                "rationale": "opened by the research coordinator",
                "status": "pending",
                "iteration": iteration,
            })
            by_id[sq_id] = sub_questions[-1]

        cards = _parse_evidence(report, sq_id, iteration)
        for card in cards:
            citation_mapping, _ = merge_sources(citation_mapping, card["sources"])
        target = by_id[sq_id]
        target["status"] = "researched" if cards else "failed"
        new_cards.extend(cards)
        new_notes.append({
            "sub_question_id": sq_id,
            "question": target.get("question", ""),
            "summary": report[:2000],
            "status": target["status"],
            "turns_used": 0,
            "session": name,
        })

    logger.info(
        "deep_research swarm (task=%s): %d cards from %d session(s)",
        ctx.task_id, len(new_cards), len(new_notes),
    )
    return {
        "evidence_cards": new_cards,
        "research_notes": new_notes,
        "sub_questions": sub_questions,
        "gap_questions": [],
        "research_iteration": iteration,
        "citation_mapping": citation_mapping,
        "errors": errors,
        "current_phase": "research",
    }


def _session_report(session: Any) -> str:
    """The last report text a session produced, or ``""``.

    Sessions are durable and may have served several tasks; the most
    recent completed one carries the findings the coordinator acted on.
    """
    for attr in ("last_result", "result"):
        result = getattr(session, attr, None)
        content = getattr(result, "final_content", None)
        if isinstance(content, str) and content.strip():
            return content
    messages = getattr(session, "messages", None) or []
    for message in reversed(messages):
        if not isinstance(message, dict):
            continue
        if message.get("role") != "assistant":
            continue
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return content
    return ""


__all__ = ["research_swarm_node"]
