"""Phase E tests: agent_bus wiring, swarm attribution, and mode dispatch.

No LLM calls: the coordinator loop is stubbed and the bus is driven with
fake sessions. What is pinned here is the contract that lets swarm mode
exist at all — same input/output shape as the fixed fan-out, so nothing
downstream can tell which mode ran.
"""

from __future__ import annotations

import types

import pytest

from agent_harness.components.agent_bus import AgentBus
from agent_harness.components.agent_bus.fan_in import (
    COMPLETE_STOP_REASONS,
    INCOMPLETE_STOP_REASONS,
    classify_completion,
    format_report_block,
)
from agent_harness.components.agent_bus.models import SubAgentResult
from agent_harness.components.agent_bus.spawn_guard import SpawnGuard
from agent_harness.components.task_board_types import (
    BOARD_TOOLS,
    BoardCounts,
    count_resolutions,
)
from agent_harness.models.pipeline_spec import NodeDefinition, SubAgentProfile
from agent_harness.models.task_budget import TaskBudget
from workflows.deep_research.nodes.research_swarm import (
    _harvest,
    _sq_id_from_session,
)


class _Ctx:
    node_id = "research_fanout"
    role_id = "dr_researcher"
    task_id = "t-swarm"


# --------------------------------------------------------------------------
# Spec extension
# --------------------------------------------------------------------------

def test_sub_agent_profiles_are_optional_and_additive():
    """Existing nodes must be unaffected by the new field."""
    plain = NodeDefinition(node_id="n", role_id="r")
    assert plain.sub_agent_profiles == {}

    with_profiles = NodeDefinition(
        node_id="n", role_id="r",
        sub_agent_profiles={
            "researcher": SubAgentProfile(role_id="dr_researcher", max_turns=12),
        },
    )
    assert with_profiles.sub_agent_profiles["researcher"].max_turns == 12
    assert with_profiles.sub_agent_profiles["researcher"].budget_fraction == 0.1


# --------------------------------------------------------------------------
# Fan-in classification
# --------------------------------------------------------------------------

def test_a_clean_finish_is_complete_and_a_capped_one_is_not():
    done = SubAgentResult(
        question="q", role_id="dr_researcher", final_content="findings",
        success=True, metadata={"stopped_by": "submit_report"},
    )
    capped = SubAgentResult(
        question="q", role_id="dr_researcher", final_content="partial",
        success=True, metadata={"stopped_by": "max_turns"},
    )
    assert classify_completion(done).status == "complete"
    assert classify_completion(capped).status != "complete"


def test_the_two_stop_reason_sets_do_not_overlap():
    """An overlap would make a branch's status depend on lookup order."""
    assert not (COMPLETE_STOP_REASONS & INCOMPLETE_STOP_REASONS)


def test_a_failed_branch_is_reported_not_silently_dropped():
    failed = SubAgentResult(
        question="q", role_id="dr_researcher", final_content="",
        success=False, error="network unreachable",
    )
    info = classify_completion(failed)
    assert info.status == "failed"
    block = format_report_block("researcher-sq1", failed, info)
    assert "researcher-sq1" in block
    assert "failed" in block


def test_report_blocks_are_parseable_envelopes():
    result = SubAgentResult(
        question="q", role_id="dr_researcher", final_content="the body",
        success=True, metadata={"stopped_by": "submit_report"},
    )
    block = format_report_block("researcher-sq2", result)
    assert block.startswith("<report")
    assert "the body" in block
    assert block.rstrip().endswith("</report>")


# --------------------------------------------------------------------------
# Spawn guard
# --------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_spawn_guard_tracks_and_releases_reservations():
    guard = SpawnGuard(TaskBudget(max_parallel=2, max_tokens=100_000))
    await guard.acquire("job-1", depth=0, estimated_tokens=100)
    await guard.acquire("job-2", depth=0, estimated_tokens=100)
    assert guard.stats()["active"] == 2
    guard.release("job-1")
    assert guard.stats()["active"] == 1


@pytest.mark.asyncio
async def test_spawn_guard_releases_even_when_the_spawn_raises():
    """RAII: a leaked slot would deadlock the next round of researchers."""
    guard = SpawnGuard(TaskBudget(max_parallel=1, max_tokens=100_000))
    with pytest.raises(RuntimeError):
        async with guard.reservation("job-x", depth=0):
            raise RuntimeError("spawn blew up")
    assert guard.stats()["active"] == 0


def test_spawn_guard_reports_its_budget():
    guard = SpawnGuard(TaskBudget(max_parallel=3, max_tokens=500))
    stats = guard.stats()
    assert stats["max_parallel"] == 3


# --------------------------------------------------------------------------
# Task board vocabulary
# --------------------------------------------------------------------------

def test_board_counts_exclude_cancelled_from_the_denominator():
    """A cancelled task is not outstanding work, so it must not read as such."""
    counts = count_resolutions(["resolved", "open", "cancelled"])
    assert isinstance(counts, BoardCounts)
    assert counts.active == 2


def test_board_tools_are_named_consistently():
    assert BOARD_TOOLS == {"add_task", "update_task"}


# --------------------------------------------------------------------------
# Swarm attribution
# --------------------------------------------------------------------------

def test_session_names_resolve_to_sub_question_ids():
    known = {"sq1", "sq2", "sq3"}
    assert _sq_id_from_session("sq2_market-size", known) == "sq2"
    assert _sq_id_from_session("SQ3-jurisdictions", known) == "sq3"
    assert _sq_id_from_session("sq1", known) == "sq1"


def test_underscore_separated_names_resolve():
    r"""Regression: ``\b`` does not treat ``_`` as a boundary.

    ``sq2_topic`` is exactly the naming convention the coordinator is
    told to use, and ``\bsq\d+\b`` silently failed on all of it.
    """
    assert _sq_id_from_session("sq2_market-size", {"sq2"}) == "sq2"


def test_an_unknown_or_partial_id_does_not_resolve():
    known = {"sq1", "sq2"}
    # sq10 must not match sq1.
    assert _sq_id_from_session("sq10_extra", known) == ""
    assert _sq_id_from_session("sq9_unknown", known) == ""
    assert _sq_id_from_session("competitor-landscape", known) == ""
    assert _sq_id_from_session("", known) == ""


def _session(name: str, content: str):
    return types.SimpleNamespace(
        name=name,
        last_result=types.SimpleNamespace(final_content=content),
    )


# Fenced, as the researcher prompt requires. A bare array is read as a
# single object by ``extract_json_block``'s brace-scan fallback, which
# would silently degrade to one source-less card.
_EVIDENCE = """Findings below.

```json
[
  {"claim": "The market grew 12% in 2024.",
   "quote": "Revenue grew 12 percent.",
   "sources": [{"url": "https://a.example/report", "title": "A"}],
   "confidence": "high"}
]
```"""


def _state():
    return {
        "task_id": "t-swarm",
        "original_question": "How big is the market?",
        "research_brief": "brief",
        "metadata": {},
        "citation_mapping": {},
        "sub_questions": [
            {"id": "sq1", "question": "size?", "status": "pending"},
            {"id": "sq2", "question": "growth?", "status": "pending"},
        ],
    }


def test_harvest_attributes_evidence_by_session_name():
    state = _state()
    sub_questions = [dict(s) for s in state["sub_questions"]]
    delta = _harvest(
        state, _Ctx(),
        [_session("sq1_size", _EVIDENCE), _session("sq2_growth", _EVIDENCE)],
        sub_questions, iteration=1, errors=[],
    )
    ids = {c["sub_question_id"] for c in delta["evidence_cards"]}
    assert ids == {"sq1", "sq2"}
    assert all(s["status"] == "researched" for s in delta["sub_questions"])
    # The citation mapping must be merged, not left empty.
    assert delta["citation_mapping"]


def test_harvest_keeps_evidence_from_an_unplanned_line_of_enquiry():
    """The coordinator may open an angle the planner never listed.

    Dropping its evidence would discard research already paid for, so it
    is recorded under a generated sub-question instead.
    """
    state = _state()
    sub_questions = [dict(s) for s in state["sub_questions"]]
    delta = _harvest(
        state, _Ctx(), [_session("regulatory-exposure", _EVIDENCE)],
        sub_questions, iteration=1, errors=[],
    )
    assert delta["evidence_cards"], "the evidence must survive"
    new_ids = {s["id"] for s in delta["sub_questions"]} - {"sq1", "sq2"}
    assert len(new_ids) == 1
    added = next(s for s in delta["sub_questions"] if s["id"] in new_ids)
    assert added["question"] == "regulatory-exposure"
    assert "coordinator" in added["rationale"]


def test_harvest_ignores_sessions_that_produced_nothing():
    state = _state()
    sub_questions = [dict(s) for s in state["sub_questions"]]
    delta = _harvest(
        state, _Ctx(), [_session("sq1_size", ""), _session("sq2_growth", "")],
        sub_questions, iteration=1, errors=[],
    )
    assert delta["evidence_cards"] == []
    assert delta["research_notes"] == []


def test_harvest_output_matches_the_fanout_contract():
    """Nothing downstream may be able to tell which mode ran."""
    state = _state()
    sub_questions = [dict(s) for s in state["sub_questions"]]
    delta = _harvest(
        state, _Ctx(), [_session("sq1_size", _EVIDENCE)],
        sub_questions, iteration=2, errors=[],
    )
    assert set(delta) >= {
        "evidence_cards", "research_notes", "sub_questions",
        "gap_questions", "research_iteration", "citation_mapping",
        "errors", "current_phase",
    }
    assert delta["research_iteration"] == 2
    assert delta["gap_questions"] == []
    assert delta["current_phase"] == "research"
    card = delta["evidence_cards"][0]
    assert set(card) >= {
        "card_id", "sub_question_id", "claim", "quote", "sources",
        "confidence", "flags", "iteration",
    }


# --------------------------------------------------------------------------
# Mode dispatch
# --------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_default_mode_is_the_fixed_fanout(monkeypatch):
    from workflows.deep_research.nodes import research as mod

    called = {}

    async def fake_fanout(state, ctx):
        called["fanout"] = True
        return {}

    monkeypatch.setattr(mod, "research_fanout_node", fake_fanout)
    await mod.research_node({"metadata": {}}, _Ctx())
    assert called.get("fanout")


@pytest.mark.asyncio
async def test_swarm_mode_dispatches_to_the_coordinator(monkeypatch):
    from workflows.deep_research.nodes import research as mod
    from workflows.deep_research.nodes import research_swarm as swarm

    called = {}

    async def fake_swarm(state, ctx):
        called["swarm"] = True
        return {}

    monkeypatch.setattr(swarm, "research_swarm_node", fake_swarm)
    await mod.research_node(
        {"metadata": {"deep_research": {"research_mode": "swarm"}}}, _Ctx(),
    )
    assert called.get("swarm")


def test_deep_depth_selects_swarm_and_default_stays_fanout():
    from workflows.deep_research.config import get_cfg

    assert get_cfg({"metadata": {}}, "research_mode") == "fanout"
    assert get_cfg(
        {"metadata": {"deep_research": {"depth": "deep"}}}, "research_mode",
    ) == "swarm"
    assert get_cfg(
        {"metadata": {"deep_research": {"depth": "quick"}}}, "research_mode",
    ) == "fanout"


# --------------------------------------------------------------------------
# Coordinator role
# --------------------------------------------------------------------------

def test_the_coordinator_cannot_research_itself():
    """Given web tools it would search instead of shaping the team."""
    from workflows.deep_research.agents import COORDINATOR_DEF

    assert "web_search" not in COORDINATOR_DEF.allowed_tools
    assert "web_fetch" not in COORDINATOR_DEF.allowed_tools
    assert "create_subagent" in COORDINATOR_DEF.allowed_tools
    assert "assign_task" in COORDINATOR_DEF.allowed_tools


def test_the_coordinator_role_is_registered():
    from workflows.deep_research.agents import ALL_AGENT_DEFS

    assert "dr_coordinator" in {d.role_id for d in ALL_AGENT_DEFS}


def test_the_coordinator_prompt_demands_id_bearing_session_names():
    """Attribution depends on it, so the instruction must be explicit."""
    from workflows.deep_research.nodes.research_swarm import COORDINATOR_SYSTEM

    assert "NAME EVERY SESSION" in COORDINATOR_SYSTEM
    assert "{max_parallel}" in COORDINATOR_SYSTEM


def test_the_bus_constructs_without_any_registered_services():
    """A workflow must be able to build one without global setup."""
    assert AgentBus() is not None
