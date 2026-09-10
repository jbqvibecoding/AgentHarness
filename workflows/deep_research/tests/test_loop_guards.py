"""Phase D tests: the guard stack a research branch runs under.

The guards divide the work three ways and this suite pins that division:
call-pattern repetition, result stagnation, and prose repetition are
different failures, and a run can hit any of them without hitting the
others.
"""

from __future__ import annotations

import pytest

from agent_harness.components.observers.context_size_guard import ContextSizeGuard
from agent_harness.components.observers.duplicate_query_rollback import (
    DuplicateQueryRollbackObserver,
)
from agent_harness.components.observers.react_step_tracker import (
    ReactStepTracker,
    _caps,
    _clip,
)
from agent_harness.core.loop_types import (
    WALL_DEADLINE_MONOTONIC_KEY,
    LoopConfig,
    ToolResult,
    TurnContext,
    wall_deadline_remaining_s,
)
from workflows.deep_research.observers.tool_progress import (
    ToolProgressGuard,
    _jaccard,
    _shingles,
)
from workflows.deep_research.stop_reason import STAGNATION_CAPPED


def _ctx(turn: int = 1, usage: dict | None = None, **over) -> TurnContext:
    fields = {
        "turn": turn, "max_turns": 20, "task_id": "t", "role_id": "dr_researcher",
        "ai_text": "", "thinking": "", "tool_calls": [], "messages": [],
        "usage": usage, "metadata": {},
    }
    fields.update(over)
    return TurnContext(**fields)


def _result(name: str, body: str, *, is_error: bool = False) -> ToolResult:
    return ToolResult(
        name=name, args={}, result=body, duration_ms=10,
        tool_call_id="c1", is_error=is_error,
    )


# --------------------------------------------------------------------------
# ContextSizeGuard — the usage-key bug
# --------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_context_guard_trips_on_openai_style_usage():
    """Regression: reading only ``input_tokens`` saw 0 on OpenAI-compatible
    endpoints, so this guard never fired where it was most needed."""
    guard = ContextSizeGuard(max_input_tokens=1000)
    await guard.on_loop_start(LoopConfig())
    intervention = await guard.on_llm_response(_ctx(usage={"prompt_tokens": 5000}))
    assert intervention is not None
    assert intervention.stop_reason == "budget_exhausted"


@pytest.mark.asyncio
async def test_context_guard_still_trips_on_anthropic_style_usage():
    guard = ContextSizeGuard(max_input_tokens=1000)
    await guard.on_loop_start(LoopConfig())
    assert await guard.on_llm_response(_ctx(usage={"input_tokens": 5000})) is not None


@pytest.mark.asyncio
async def test_context_guard_stays_quiet_under_the_limit_and_fires_once():
    guard = ContextSizeGuard(max_input_tokens=10_000)
    await guard.on_loop_start(LoopConfig())
    assert await guard.on_llm_response(_ctx(usage={"prompt_tokens": 500})) is None
    assert await guard.on_llm_response(_ctx(usage={"prompt_tokens": 50_000})) is not None
    # A second trip would inject the same stop twice.
    assert await guard.on_llm_response(_ctx(usage={"prompt_tokens": 50_000})) is None


# --------------------------------------------------------------------------
# ReactStepTracker — bounded
# --------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_react_steps_are_capped_so_long_runs_do_not_grow_unbounded():
    tracker = ReactStepTracker()
    ctx = _ctx()
    await tracker.on_tool_result(ctx, _result("web_fetch", "x" * 200_000))
    step = ctx.metadata["react_steps"][0]
    result_cap, _thinking_cap, _args_cap = _caps()
    assert len(step["tool_result"]) < result_cap + 200
    assert "truncated" in step["tool_result"]
    # The pre-caps field name still resolves, so an existing reader keeps
    # working instead of silently reading None.
    assert step["tool_result_preview"] == step["tool_result"]


def test_clip_marks_truncation_and_leaves_short_text_alone():
    assert _clip("abc", 100) == "abc"
    clipped = _clip("y" * 500, 50)
    assert clipped.startswith("y") and "truncated" in clipped


@pytest.mark.asyncio
async def test_react_step_keeps_the_fields_callers_read():
    tracker = ReactStepTracker()
    ctx = _ctx(turn=3, thinking="deliberating")
    await tracker.on_tool_result(ctx, _result("web_search", "results here"))
    step = ctx.metadata["react_steps"][0]
    assert step["turn"] == 3
    assert step["tool_name"] == "web_search"
    assert "results here" in step["tool_result"]


# --------------------------------------------------------------------------
# DuplicateQueryRollback — the upgraded version
# --------------------------------------------------------------------------

def test_rollback_budget_has_a_floor_that_keeps_it_enabled():
    """max_consecutive_rollbacks=1 would mean "never roll back"."""
    assert DuplicateQueryRollbackObserver(
        max_consecutive_rollbacks=1,
    )._max_consecutive_rollbacks == 2


def test_rollback_watches_search_but_not_fetch_by_default():
    """Re-fetching a URL is legitimate; re-running a search is not."""
    observer = DuplicateQueryRollbackObserver()
    assert "web_search" in observer._tool_names
    assert "web_fetch" not in observer._tool_names


def test_rollback_tool_set_is_configurable():
    observer = DuplicateQueryRollbackObserver(tool_names={"scholar_search"})
    assert observer._tool_names == frozenset({"scholar_search"})


# --------------------------------------------------------------------------
# Similarity primitives
# --------------------------------------------------------------------------

# A search-result page's worth of varied prose. Repetitive filler would
# collapse to a handful of distinct bigrams, where a few words of noise
# swing the score wildly — real results do not look like that.
_PAGE = " ".join(
    f"result {i} the company reported segment {i} revenue growth of "
    f"{i} percent citing demand in region {i}"
    for i in range(40)
)


def test_similarity_sees_through_trailing_noise():
    """Exact-match comparison would never fire on real search results."""
    noisy = _PAGE + " Updated 14:32 GMT. Advertisement. Cookie notice."
    assert _jaccard(_shingles(_PAGE), _shingles(noisy)) > 0.9


def test_similarity_sees_through_reordering():
    lines = _PAGE.split("result ")
    reordered = "result ".join([lines[0]] + list(reversed(lines[1:])))
    assert _jaccard(_shingles(_PAGE), _shingles(reordered)) > 0.85


def test_similarity_separates_genuinely_different_material():
    other = " ".join(
        f"the treaty article {i} was ratified after negotiation round {i}"
        for i in range(40)
    )
    assert _jaccard(_shingles(_PAGE), _shingles(other)) < 0.1


def test_shingles_handle_degenerate_input():
    assert _shingles("") == frozenset()
    assert _jaccard(frozenset(), _shingles("anything at all")) == 0.0


# --------------------------------------------------------------------------
# ToolProgressGuard — result stagnation
# --------------------------------------------------------------------------

_SAME = "the same market analysis repeated verbatim across every result " * 40
_NEW = "an entirely different filing covering supply chain exposure " * 40


@pytest.mark.asyncio
async def test_guard_warns_before_it_stops():
    """The model gets one chance to change approach first."""
    guard = ToolProgressGuard(run_id="r", warn_after=2, stop_after=4)
    await guard.on_loop_start(LoopConfig())
    ctx = _ctx()

    interventions = []
    for turn in range(6):
        await guard.on_tool_result(_ctx(turn=turn), _result("web_search", _SAME))
        interventions.append(await guard.on_turn_end(ctx))

    warnings = [i for i in interventions if i and i.stop_reason is None]
    stops = [i for i in interventions if i and i.stop_reason]
    assert warnings, "a warning must precede the stop"
    assert stops, "persistent stagnation must eventually stop the branch"
    assert interventions.index(warnings[0]) < interventions.index(stops[0])


@pytest.mark.asyncio
async def test_stopping_reports_through_the_additive_channel():
    guard = ToolProgressGuard(run_id="run-9", warn_after=1, stop_after=2)
    await guard.on_loop_start(LoopConfig())
    for _ in range(4):
        await guard.on_tool_result(_ctx(), _result("web_search", _SAME))
        await guard.on_turn_end(_ctx())
    assert guard.consume_stop_reason("run-9") == STAGNATION_CAPPED
    assert guard.consume_stop_reason("run-9") is None


@pytest.mark.asyncio
async def test_new_information_resets_the_streak():
    """A researcher that recovers must not carry a penalty forward."""
    guard = ToolProgressGuard(run_id="r", warn_after=2, stop_after=3)
    await guard.on_loop_start(LoopConfig())
    for _ in range(2):
        await guard.on_tool_result(_ctx(), _result("web_search", _SAME))
        await guard.on_turn_end(_ctx())
    await guard.on_tool_result(_ctx(), _result("web_search", _NEW))
    await guard.on_turn_end(_ctx())
    assert guard._repeats.get("web_search", 0) == 0
    # And one more repeat is not immediately fatal.
    await guard.on_tool_result(_ctx(), _result("web_search", _NEW))
    assert await guard.on_turn_end(_ctx()) is None


@pytest.mark.asyncio
async def test_guard_ignores_errors_short_bodies_and_untracked_tools():
    guard = ToolProgressGuard(run_id="r", warn_after=1, stop_after=2)
    await guard.on_loop_start(LoopConfig())
    for _ in range(6):
        # An error string is not evidence of stagnation.
        await guard.on_tool_result(_ctx(), _result("web_search", _SAME, is_error=True))
        # Too short to score meaningfully.
        await guard.on_tool_result(_ctx(), _result("web_search", "tiny"))
        # Not a content-gathering tool: returning the same thing is its job.
        await guard.on_tool_result(_ctx(), _result("vault_get", _SAME))
        assert await guard.on_turn_end(_ctx()) is None


@pytest.mark.asyncio
async def test_guard_never_rewrites_the_tool_result():
    """on_tool_result's return value replaces the result.

    Rewriting a researcher's evidence would be a far worse intervention
    than stopping the loop, so the decision is carried to on_turn_end.
    """
    guard = ToolProgressGuard(run_id="r", warn_after=1, stop_after=2)
    await guard.on_loop_start(LoopConfig())
    for _ in range(5):
        assert await guard.on_tool_result(
            _ctx(), _result("web_search", _SAME),
        ) is None


def test_stop_after_cannot_precede_the_warning():
    guard = ToolProgressGuard(warn_after=5, stop_after=2)
    assert guard._stop_after > guard._warn_after


# --------------------------------------------------------------------------
# Wall-deadline channel
# --------------------------------------------------------------------------

def test_wall_deadline_is_absent_without_a_scope():
    assert wall_deadline_remaining_s() is None


def test_wall_deadline_reads_a_monotonic_value_off_the_scope():
    import time

    from agent_harness.core.execution_context import (
        ExecutionScope,
        reset_current_execution_scope,
        set_current_execution_scope,
    )

    scope = ExecutionScope(task_id="t")
    scope.metadata[WALL_DEADLINE_MONOTONIC_KEY] = time.monotonic() + 120
    token = set_current_execution_scope(scope)
    try:
        remaining = wall_deadline_remaining_s()
        assert remaining is not None and 100 < remaining <= 120
    finally:
        reset_current_execution_scope(token)


def test_a_non_numeric_deadline_reads_as_no_deadline():
    from agent_harness.core.execution_context import (
        ExecutionScope,
        reset_current_execution_scope,
        set_current_execution_scope,
    )

    scope = ExecutionScope(task_id="t")
    scope.metadata[WALL_DEADLINE_MONOTONIC_KEY] = "not a number"
    token = set_current_execution_scope(scope)
    try:
        assert wall_deadline_remaining_s() is None
    finally:
        reset_current_execution_scope(token)


# --------------------------------------------------------------------------
# The assembled stack
# --------------------------------------------------------------------------

def test_the_branch_stack_covers_all_three_failure_shapes():
    from workflows.deep_research.subagent import _build_observers

    names = {type(o).__name__ for o in _build_observers(["web_search"], "r")}
    # call pattern, result quality, prose
    assert "DuplicateQueryRollbackObserver" in names
    assert "RepetitionGuard" in names
    assert "ToolProgressGuard" in names
    assert "TextRepetitionGuard" in names
    # a host that keeps failing gets quarantined rather than retried forever
    assert "StuckTargetGuard" in names


def test_every_guard_in_the_stack_is_awaited_by_the_loop():
    """A non-critical observer's Intervention is discarded.

    A guard that cannot stop anything is decoration, so any guard whose
    job is to intervene must declare critical.
    """
    from workflows.deep_research.subagent import _build_observers

    intervening = {
        "DuplicateQueryRollbackObserver", "RepetitionGuard",
        "TextRepetitionGuard", "ToolProgressGuard", "StuckTargetGuard",
    }
    for observer in _build_observers(["web_search"], "r"):
        if type(observer).__name__ in intervening:
            assert observer.critical, type(observer).__name__


def test_prompt_cache_wraps_only_claude_upstreams():
    from agent_harness.infra.prompt_cache import (
        AnthropicPromptCacheAdapter,
        maybe_wrap_for_prompt_cache,
    )

    class _Client:
        def __init__(self, model: str) -> None:
            self.model = model

    claude = maybe_wrap_for_prompt_cache(
        _Client("claude-opus-4"), provider="anthropic", model="claude-opus-4",
    )
    other = maybe_wrap_for_prompt_cache(
        _Client("gpt-4o"), provider="openai", model="gpt-4o",
    )
    assert isinstance(claude, AnthropicPromptCacheAdapter)
    assert not isinstance(other, AnthropicPromptCacheAdapter)
