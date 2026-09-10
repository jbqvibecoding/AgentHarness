"""Phase B tests: token budget, additive stop reasons, provider failover.

Pure-function and fake-client tests — no API key, no network.
"""

from __future__ import annotations

import asyncio

import pytest

from agent_harness.core.llm import LLMResponse
from agent_harness.core.loop_types import LoopConfig, TurnContext
from agent_harness.infra.llm import FallbackEntry, LLMFallbackChain
from agent_harness.infra.retriable import (
    classify_error,
    is_retriable_with_fallback,
    is_transient_network,
)
from workflows.deep_research.budget import (
    RunBudget,
    RunBudgetObserver,
    _usage_tokens,
    budget_from_state,
    get_run_budget,
    release_run_budget,
)
from workflows.deep_research.stop_reason import (
    LOOP_CAPPED,
    TOKEN_CAPPED,
    TURN_CAPPED,
    WALL_CAPPED,
    StopReasonRegistry,
    collect_stop_reason,
    describe,
    reason_from_stopped_by,
)


class APIError(Exception):
    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


# --------------------------------------------------------------------------
# Error classification — what a different provider can and cannot fix
# --------------------------------------------------------------------------

def test_plain_5xx_is_transient_not_a_provider_switch():
    """A bare 503 means back off on the same key.

    Switching providers on it burns the fallback tier for a blip the next
    request would likely survive.
    """
    err = APIError("Service Unavailable", 503)
    assert is_transient_network(err)
    assert not is_retriable_with_fallback(err)
    assert classify_error(err) == "transient_network"


def test_overload_and_model_not_found_do_advance_the_chain():
    for message, expected in [
        ("overloaded_error: upstream at capacity", "overloaded"),
        ("model_not_found: no channel available", "model_unavailable"),
        ("insufficient credit balance", "credit_exhausted"),
        ("invalid_api_key", "auth_failure"),
    ]:
        err = APIError(message, 503)
        assert is_retriable_with_fallback(err), message
        assert classify_error(err) == expected


def test_context_length_wins_over_everything():
    err = APIError("maximum context length is 8192 tokens", 400)
    assert classify_error(err) == "context_length"


# --------------------------------------------------------------------------
# Fallback chain
# --------------------------------------------------------------------------

class _Failing:
    model = "primary-model"

    def __init__(self, exc: BaseException) -> None:
        self.exc = exc
        self.calls = 0

    async def chat(self, messages, **kwargs):  # noqa: ANN001, ARG002
        self.calls += 1
        raise self.exc


class _Working:
    model = "backup-model"

    def __init__(self) -> None:
        self.calls = 0

    async def chat(self, messages, **kwargs):  # noqa: ANN001, ARG002
        self.calls += 1
        return LLMResponse(content="answer from backup", model=self.model)


@pytest.mark.asyncio
async def test_chain_advances_on_a_failure_another_provider_can_fix():
    primary = _Failing(APIError("model_not_found: no channel", 503))
    backup = _Working()
    chain = LLMFallbackChain([
        FallbackEntry(primary, ("retriable",), "openai"),
        FallbackEntry(backup, (), "anthropic"),
    ])
    resp = await chain.chat([])
    assert resp.content == "answer from backup"
    assert primary.calls == 1 and backup.calls == 1
    # Billing must be able to tell which vendor actually served the call.
    assert resp.response_metadata["provider_actually_used"] == "anthropic"
    assert resp.response_metadata["model_actually_used"] == "backup-model"
    assert resp.response_metadata["fallback_used"] == 1


@pytest.mark.asyncio
async def test_chain_holds_on_a_transient_so_backoff_can_do_its_job():
    primary = _Failing(APIError("Service Unavailable", 503))
    backup = _Working()
    chain = LLMFallbackChain([
        FallbackEntry(primary, ("retriable",)),
        FallbackEntry(backup, ()),
    ])
    with pytest.raises(APIError):
        await chain.chat([])
    assert backup.calls == 0


@pytest.mark.asyncio
async def test_chain_exposes_the_primary_model_name():
    chain = LLMFallbackChain([FallbackEntry(_Working(), ())])
    assert chain.model == "backup-model"


# --------------------------------------------------------------------------
# Token accounting
# --------------------------------------------------------------------------

def test_usage_tokens_reads_both_provider_spellings():
    """OpenAI says prompt/completion; Anthropic says input/output.

    Reading only one pair counts zero on half the providers, which is how a
    budget guard silently never fires.
    """
    assert _usage_tokens({"input_tokens": 10, "output_tokens": 5}) == 15
    assert _usage_tokens({"prompt_tokens": 10, "completion_tokens": 5}) == 15
    assert _usage_tokens({}) == 0
    assert _usage_tokens(None) == 0
    assert _usage_tokens({"prompt_tokens": "bad"}) == 0


def test_branch_slice_divides_the_run_total_by_parallelism():
    budget = RunBudget(budget_from_state({
        "metadata": {"deep_research": {
            "max_run_tokens": 900, "max_parallel_subagents": 3,
        }},
    }))
    assert budget.max_tokens == 900
    assert budget.branch_slice == 300


def test_spend_accumulates_per_role_and_reports_exhaustion():
    budget = RunBudget(budget_from_state({
        "metadata": {"deep_research": {"max_run_tokens": 100}},
    }))
    budget.spend(40, role_id="dr_researcher")
    budget.spend(30, role_id="dr_researcher")
    budget.spend(20, role_id="dr_writer")
    assert budget.spent == 90
    assert budget.remaining == 10
    assert not budget.exhausted
    budget.spend(10, role_id="dr_writer")
    assert budget.exhausted
    summary = budget.summary()
    assert summary["by_role"] == {"dr_researcher": 70, "dr_writer": 30}


def test_run_budget_registry_is_per_task_and_releasable():
    state = {"metadata": {"deep_research": {"max_run_tokens": 500}}}
    assert get_run_budget("task-a") is None, "no implicit creation"
    created = get_run_budget("task-a", state)
    assert created is not None
    assert get_run_budget("task-a") is created, "same budget for the same task"
    assert get_run_budget("task-b", state) is not created
    summary = release_run_budget("task-a")
    assert summary is not None and summary["max_tokens"] == 500
    assert get_run_budget("task-a") is None
    release_run_budget("task-b")


# --------------------------------------------------------------------------
# Budget observer
# --------------------------------------------------------------------------

def _turn_ctx(usage: dict[str, int] | None) -> TurnContext:
    return TurnContext(
        turn=1, max_turns=10, task_id="t", role_id="dr_researcher",
        ai_text="", thinking="", tool_calls=[], messages=[],
        usage=usage, metadata={},
    )


@pytest.mark.asyncio
async def test_observer_stops_the_branch_without_raising():
    """Exhaustion must return an Intervention, never raise.

    Raising would discard the evidence the branch already gathered, which
    is the whole reason it was allowed to run.
    """
    budget = RunBudget(budget_from_state({
        "metadata": {"deep_research": {
            "max_run_tokens": 100, "max_parallel_subagents": 1,
        }},
    }))
    obs = RunBudgetObserver(budget, run_id="r1", role_id="dr_researcher")
    await obs.on_loop_start(LoopConfig())
    await obs.on_llm_response(_turn_ctx({"prompt_tokens": 60, "completion_tokens": 50}))
    intervention = await obs.on_turn_end(_turn_ctx(None))
    assert intervention is not None
    assert intervention.stop_reason == "budget_exhausted"
    assert "report what you have found" in intervention.inject_messages[0]
    # And the additive channel now carries the reason.
    assert obs.consume_stop_reason("r1") == TOKEN_CAPPED
    assert obs.consume_stop_reason("r1") is None, "pop-once"


@pytest.mark.asyncio
async def test_observer_warns_once_before_the_cap():
    budget = RunBudget(budget_from_state({
        "metadata": {"deep_research": {"max_run_tokens": 100}},
    }))
    obs = RunBudgetObserver(
        budget, run_id="r2", branch_limit=10_000, warn_ratio=0.5,
    )
    await obs.on_loop_start(LoopConfig())
    await obs.on_llm_response(_turn_ctx({"input_tokens": 60, "output_tokens": 0}))
    first = await obs.on_turn_end(_turn_ctx(None))
    assert first is not None and first.stop_reason is None
    second = await obs.on_turn_end(_turn_ctx(None))
    assert second is None, "the warning must not repeat every turn"


@pytest.mark.asyncio
async def test_branch_slice_caps_a_runaway_before_the_run_total():
    """One greedy branch must not drain the pool its siblings share."""
    budget = RunBudget(budget_from_state({
        "metadata": {"deep_research": {
            "max_run_tokens": 1000, "max_parallel_subagents": 4,
        }},
    }))
    obs = RunBudgetObserver(budget, run_id="r3")
    await obs.on_loop_start(LoopConfig())
    await obs.on_llm_response(_turn_ctx({"input_tokens": 250, "output_tokens": 0}))
    intervention = await obs.on_turn_end(_turn_ctx(None))
    assert intervention is not None and intervention.stop_reason == "budget_exhausted"
    assert not budget.exhausted, "the run as a whole still has budget left"


# --------------------------------------------------------------------------
# Additive stop reasons
# --------------------------------------------------------------------------

def test_registry_keeps_the_first_reason_and_pops_once():
    registry = StopReasonRegistry()
    registry.record("run-1", TOKEN_CAPPED)
    registry.record("run-1", LOOP_CAPPED)
    assert registry.consume("run-1") == TOKEN_CAPPED
    assert registry.consume("run-1") is None


def test_registry_is_bounded():
    registry = StopReasonRegistry(max_runs=3)
    for i in range(5):
        registry.record(f"run-{i}", TOKEN_CAPPED)
    assert registry.consume("run-0") is None, "oldest evicted"
    assert registry.consume("run-4") == TOKEN_CAPPED


class _Guard:
    def __init__(self, reason: str | None) -> None:
        self._reason = reason
        self.consumed = 0

    def consume_stop_reason(self, run_id: str) -> str | None:  # noqa: ARG002
        self.consumed += 1
        reason, self._reason = self._reason, None
        return reason


def test_collect_takes_the_first_guard_and_drains_the_rest():
    first, second = _Guard(TOKEN_CAPPED), _Guard(LOOP_CAPPED)
    plain = object()
    assert collect_stop_reason([plain, first, second], "r") == TOKEN_CAPPED
    # Both guards were asked, so nothing is left to leak into a later run.
    assert first.consumed == 1 and second.consumed == 1
    assert second.consume_stop_reason("r") is None


def test_collect_survives_a_guard_that_throws():
    class _Broken:
        def consume_stop_reason(self, run_id: str) -> str | None:
            raise RuntimeError("guard exploded")

    assert collect_stop_reason([_Broken(), _Guard(WALL_CAPPED)], "r") == WALL_CAPPED


def test_stopped_by_translates_to_the_additive_vocabulary():
    assert reason_from_stopped_by("max_turns") == TURN_CAPPED
    assert reason_from_stopped_by("budget_exhausted") == TOKEN_CAPPED
    assert reason_from_stopped_by("wall_deadline") == WALL_CAPPED
    assert reason_from_stopped_by("repeated_tool_calls") == LOOP_CAPPED
    # A normal finish carries no reason, so its absence means "not capped".
    assert reason_from_stopped_by("final_answer") is None
    assert reason_from_stopped_by("") is None
    assert reason_from_stopped_by(None) is None


def test_every_reason_has_reader_facing_phrasing():
    from workflows.deep_research.stop_reason import STOP_REASONS

    for reason in STOP_REASONS:
        assert describe(reason), reason
    assert describe(None) == ""


# --------------------------------------------------------------------------
# Routing
# --------------------------------------------------------------------------

def test_research_loop_ends_when_the_budget_is_spent():
    from workflows.deep_research.conditions import route_after_conflict_check

    state = {
        "task_id": "budget-route",
        "gap_questions": [{"id": "g1"}],
        "research_iteration": 1,
        "metadata": {"deep_research": {
            "max_research_iterations": 3, "max_run_tokens": 100,
        }},
    }
    budget = get_run_budget("budget-route", state)
    assert budget is not None
    try:
        assert route_after_conflict_check(state) == "research_fanout"
        budget.spend(100)
        assert route_after_conflict_check(state) == "draft"
    finally:
        release_run_budget("budget-route")


def test_routing_is_unaffected_when_budgeting_is_off():
    from workflows.deep_research.conditions import route_after_conflict_check

    state = {
        "gap_questions": [{"id": "g1"}],
        "research_iteration": 0,
        "metadata": {"deep_research": {"max_research_iterations": 2}},
    }
    assert route_after_conflict_check(state) == "research_fanout"


# --------------------------------------------------------------------------
# Wall-time split
# --------------------------------------------------------------------------

def test_deep_research_owns_its_wall_and_reserves_finalization_time():
    from agent_harness.components.finalization.budget import soft_wall_deadline_s
    from agent_harness.scheduling.scheduler import _resolve_wall_time_s
    from workflows.deep_research import defaults

    assert defaults.TASK_WALL_TIME_MODE == "soft_research"
    assert _resolve_wall_time_s(None, "deep_research") == defaults.TASK_WALL_TIME_S
    # An explicit caller value still wins.
    assert _resolve_wall_time_s(60, "deep_research") == 60
    # Research gets the wall minus the finalize reserve...
    research = soft_wall_deadline_s(
        defaults.TASK_WALL_TIME_S, defaults.RESEARCH_FINALIZE_RESERVE_S,
    )
    assert research < defaults.TASK_WALL_TIME_S
    # ...but never less than half, so a short wall cannot starve research.
    assert soft_wall_deadline_s(600, 900) == 300


def test_unknown_pipeline_keeps_the_previous_no_deadline_behaviour():
    from agent_harness.scheduling.scheduler import _resolve_wall_time_s

    assert _resolve_wall_time_s(None, "some_other_pipeline") is None
    assert _resolve_wall_time_s(None, None) is None


# --------------------------------------------------------------------------
# Damaged-history recovery
# --------------------------------------------------------------------------

def test_finalize_replays_a_healthy_transcript_as_is():
    from workflows.deep_research.subagent import _finalize_messages

    history = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "research X"},
        {"role": "assistant", "content": "found it"},
    ]
    out = _finalize_messages("sys", history)
    assert len(out) == len(history) + 1
    assert out[:3] == history


def test_finalize_flattens_a_transcript_that_would_break_tool_pairing():
    """An orphaned tool message makes the provider reject the replay.

    Salvaging the branch matters more than preserving role structure, so a
    malformed history is flattened into one plain-text block instead.
    """
    from workflows.deep_research.subagent import _finalize_messages

    history = [
        {"role": "user", "content": "research X"},
        {"role": "tool", "content": "orphaned result", "tool_call_id": "nope"},
    ]
    out = _finalize_messages("sys prompt", history)
    assert len(out) == 2
    assert out[0]["role"] == "system"
    assert out[1]["role"] == "user"
    assert "research X" in out[1]["content"]


def test_leaked_tool_call_markup_is_scrubbed():
    from workflows.deep_research.subagent import _strip_leaked_tool_calls

    text = 'Findings.<tool_call>{"name":"web_search"}</tool_call> More.'
    cleaned = _strip_leaked_tool_calls(text)
    assert "tool_call" not in cleaned
    assert "Findings." in cleaned and "More." in cleaned


def test_scrubbing_leaves_ordinary_text_alone():
    from workflows.deep_research.subagent import _strip_leaked_tool_calls

    assert _strip_leaked_tool_calls("plain findings") == "plain findings"
    assert _strip_leaked_tool_calls("") == ""


if __name__ == "__main__":  # pragma: no cover
    asyncio.run(test_chain_advances_on_a_failure_another_provider_can_fix())
