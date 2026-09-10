"""Shared sub-agent runner for the ``deep_research`` fan-out nodes.

``run_subagent`` wraps ``run_agent_loop`` the way
``workflows/react_base/nodes/main_agent.py`` does, minus the benchmark
plumbing: per-role LLM from the deep_research profile, role-gated tools
from the ResourceManager, the rollback observer stack, a per-branch
wall-clock timeout, and a single tool-free "summarize now" recovery call
when the loop runs out of turns.

The react_base observers imported here are generic LoopObservers with no
react_base coupling; they should eventually be promoted to
``agent_harness.components.observers``.
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any, Awaitable, Sequence

from agent_harness.components.observers.duplicate_query_rollback import (
    DuplicateQueryRollbackObserver,
)
from agent_harness.components.observers.leaked_tool_call_retry import (
    LeakedToolCallRetryObserver,
)
from agent_harness.components.observers.repetition_guard import RepetitionGuard
from agent_harness.components.observers.stuck_target_guard import StuckTargetGuard
from agent_harness.components.observers.text_repetition_guard import (
    TextRepetitionGuard,
)
from agent_harness.core.loop_types import (
    AgentLoopResult,
    LoopConfig,
    LoopPolicy,
)
from agent_harness.core.messages import system_msg, text_of, user_msg
from agent_harness.core.runtime import registry
from agent_harness.core.runtime.loop.agent_loop import run_agent_loop
from agent_harness.core.runtime.resources.manager import ResourceManager

from workflows.deep_research.observers.tool_progress import ToolProgressGuard
from workflows.react_base.observers.empty_search_rollback import (
    EmptySearchRollbackObserver,
)
from workflows.react_base.observers.refusal_rollback import (
    RefusalRollbackObserver,
)
from workflows.react_base.observers.tool_call_normalizer import (
    ToolCallArgsNormalizer,
)

logger = logging.getLogger(__name__)


def _strip_thinking(text: str) -> str:
    text = text or ""
    if "</think>" in text:
        return text.rsplit("</think>", 1)[-1].strip()
    return text.strip()


# Text-mode tool-call markup some models emit into visible content instead
# of into structured tool calls. Ported from FrontierAgent
# (workflows/stateful_react_agent/_runtime.py, Apache-2.0). Replaying this
# markup back to the model in a recovery prompt invites it to "answer" the
# leaked call rather than finalize.
_LEAKED_TOOL_CALL_BLOCK_RE = re.compile(
    r"<\s*tool_call[^>]*>[\s\S]*?<\s*/\s*tool_call\s*>", re.IGNORECASE,
)
_LEAKED_TOOL_RESPONSE_BLOCK_RE = re.compile(
    r"<\s*tool_response[^>]*>[\s\S]*?<\s*/\s*tool_response\s*>", re.IGNORECASE,
)
_LEAKED_FUNCTION_BLOCK_RE = re.compile(
    r"<\s*function\s*=[\s\S]*?<\s*/\s*function\s*>", re.IGNORECASE,
)
_LEAKED_TAG_FRAGMENT_RE = re.compile(
    r"<\s*(tool_call|tool_response|function)\b[^>]*>?", re.IGNORECASE,
)


def _strip_leaked_tool_calls(text: str) -> str:
    """Remove text-mode tool-call markup that leaked into visible content."""
    if not text:
        return text
    out = _LEAKED_TOOL_CALL_BLOCK_RE.sub("", text)
    out = _LEAKED_TOOL_RESPONSE_BLOCK_RE.sub("", out)
    out = _LEAKED_FUNCTION_BLOCK_RE.sub("", out)
    return _LEAKED_TAG_FRAGMENT_RE.sub("", out).strip()


def _build_observers(tool_names: list[str], run_id: str) -> list[Any]:
    """The guard stack every research branch runs under.

    The four repetition/refusal guards watch the *call pattern*;
    ``ToolProgressGuard`` watches the *results*, which is a different
    failure — a researcher issuing genuinely varied searches that all come
    back with the same material is stuck too, and no call-pattern guard
    sees it. ``TextRepetitionGuard`` covers the third shape: prose that
    starts looping regardless of what the tools return.
    """
    return [
        ToolCallArgsNormalizer(),
        LeakedToolCallRetryObserver(tool_names=tool_names),
        DuplicateQueryRollbackObserver(),
        RefusalRollbackObserver(),
        EmptySearchRollbackObserver(),
        RepetitionGuard(),
        TextRepetitionGuard(),
        StuckTargetGuard(),
        ToolProgressGuard(run_id=run_id),
    ]


def _trajectory_observer(
    *, task_id: str, role_id: str, run_id: str, scope_metadata: dict[str, Any] | None,
) -> Any | None:
    """Record this branch's transcript, so ``recover_result`` has a source.

    A tool result is capped before it reaches the model and the remainder is
    discarded with nothing but a truncation marker — but the full body was
    already handed to the observers, so writing it here is what makes it
    recoverable at all. Without a trajectory file the tool is inert.

    Writes under the run's vault directory when there is one (it is already
    the run's scratch space) and is skipped entirely otherwise, so a run
    with nowhere to write does not silently scatter files.
    """
    vault_dir = (scope_metadata or {}).get("vault_dir")
    if not vault_dir:
        return None
    try:
        from pathlib import Path

        from agent_harness.components.observers.trajectory import (
            TrajectoryFileObserver,
        )

        return TrajectoryFileObserver(
            Path(vault_dir) / "trajectories",
            filename=run_id.replace(":", "_"),
            formats=("jsonl",),
        )
    except Exception as exc:  # noqa: BLE001 — recording is not the job
        logger.debug(
            "trajectory recording unavailable (task=%s role=%s): %s",
            task_id, role_id, exc,
        )
        return None


def _stamp_stop_reason(
    result: AgentLoopResult, observers: list[Any], run_id: str,
) -> None:
    """Record *why* a branch stopped early, as an optional extra field.

    A guard's own reason wins over the one inferred from ``stopped_by``:
    the guard knows which limit it enforced, whereas ``stopped_by`` is the
    loop engine's coarser view of the same event.

    Nothing is written when the branch finished normally, so consumers can
    treat the key's presence as "this branch was cut short".
    """
    from workflows.deep_research.stop_reason import (
        collect_stop_reason,
        reason_from_stopped_by,
    )

    reason = (
        collect_stop_reason(observers, run_id)
        or reason_from_stopped_by(getattr(result, "stopped_by", None))
    )
    if not reason:
        return
    metadata = getattr(result, "metadata", None)
    if isinstance(metadata, dict):
        metadata["stop_reason"] = reason
    logger.info("subagent capped early (run=%s): %s", run_id, reason)


def _budget_observer(
    *, task_id: str, role_id: str, run_id: str, state: dict[str, Any] | None,
) -> Any | None:
    """The run-wide token budget guard, when this run has a budget.

    Returns ``None`` when budgeting is off (``max_run_tokens`` of 0) or when
    the caller passed no state to create the budget from, so an unbudgeted
    call site behaves exactly as before.
    """
    from workflows.deep_research.budget import RunBudgetObserver, get_run_budget

    budget = get_run_budget(task_id, state)
    if budget is None or budget.max_tokens <= 0:
        return None
    warn_ratio = 0.8
    if state is not None:
        from workflows.deep_research.config import get_cfg

        warn_ratio = float(get_cfg(state, "budget_warn_ratio"))
    return RunBudgetObserver(
        budget, run_id=run_id, role_id=role_id, warn_ratio=warn_ratio,
    )


async def run_subagent(
    *,
    role_id: str,
    system_prompt: str,
    user_message: str,
    max_turns: int,
    task_id: str,
    profile_name: str = "default",
    timeout_s: float = 900.0,
    tool_result_max_chars: int = 30_000,
    extra_observers: list[Any] | None = None,
    scope_metadata: dict[str, Any] | None = None,
    state: dict[str, Any] | None = None,
) -> AgentLoopResult:
    """Run one isolated ReAct sub-agent and return its loop result.

    Raises ``asyncio.TimeoutError`` on wall-clock overrun and propagates
    loop-level infra errors — callers decide whether a branch failure is
    fatal (fan-out nodes treat it as a failed sub-question).

    ``extra_observers`` are appended to the default rollback stack (e.g. a
    ``VaultWriterObserver``); ``scope_metadata`` is merged into the loop's
    ExecutionScope (e.g. ``{"vault_dir": ...}`` for the vault tools).

    Passing ``state`` enrols the branch in the run's shared token budget and
    stamps ``result.metadata["stop_reason"]`` when a limit cut it short. The
    branch still returns its findings in that case — a capped branch reports
    what it has rather than raising and losing it.
    """
    from workflows.deep_research.profile import get_llm_for_role

    llm = get_llm_for_role(role_id, profile_name)
    tools = registry.get(ResourceManager).get_tools_for_role(role_id)
    tool_names = [t.name for t in tools]

    config = LoopConfig(
        max_turns=max_turns,
        task_id=task_id,
        role_id=role_id,
        no_tool_max_retries=0,
        tool_timeout=180,
        llm_timeout=300,
        tool_result_max_chars=tool_result_max_chars,
        max_llm_retries=8,
        retry_wait_fixed=30,
        loop_policy=LoopPolicy(no_tool_behavior="stop"),
        context_overflow_guard=True,
        max_context_length=120_000,
        max_completion_tokens=16_384,
    )

    # One run id per branch so guards keyed on it never collide between
    # sibling branches sharing a task_id.
    run_id = f"{task_id}:{role_id}:{id(config):x}"
    observers = _build_observers(tool_names, run_id)
    budget_obs = _budget_observer(
        task_id=task_id, role_id=role_id, run_id=run_id, state=state,
    )
    if budget_obs is not None:
        observers.append(budget_obs)
    trajectory_obs = _trajectory_observer(
        task_id=task_id, role_id=role_id, run_id=run_id,
        scope_metadata=scope_metadata,
    )
    if trajectory_obs is not None:
        observers.append(trajectory_obs)
    if extra_observers:
        observers = observers + list(extra_observers)

    result = await asyncio.wait_for(
        run_agent_loop(
            system_prompt=system_prompt,
            user_message=user_message,
            llm=llm,
            tools=tools,
            config=config,
            observers=observers,
            scope_metadata=scope_metadata,
        ),
        timeout=timeout_s,
    )
    _stamp_stop_reason(result, observers, run_id)

    result.final_content = _strip_thinking(result.final_content)
    if result.final_content:
        return result

    # Out of turns (or empty visible answer): one tool-free recovery call
    # asking for the required JSON now — simplified react_base
    # ``_force_final_answer``; failure is non-fatal, caller sees "".
    if result.stopped_by in {
        "max_turns", "context_limit_reached", "no_tool", "budget_exhausted",
    }:
        try:
            messages = _finalize_messages(system_prompt, result.messages)
            resp = await asyncio.wait_for(llm.chat(messages), timeout=300)
            result.final_content = _strip_thinking(text_of(resp.content))
        except Exception as exc:  # noqa: BLE001 — branch stays non-fatal
            logger.warning(
                "subagent forced-summary failed (role=%s): %s", role_id, exc,
            )
    return result


_FINALIZE_INSTRUCTION = (
    "Stop researching now. Output your findings immediately in the exact "
    "final-answer format required by your instructions (the fenced JSON "
    "block)."
)


def _finalize_messages(system_prompt: str, messages: list[Any]) -> list[Any]:
    """Build the forced-summary request, repairing a damaged history first.

    A branch that ran out of turns often ends mid-tool-call — an assistant
    turn whose tool calls never got their results. Replaying that verbatim
    asks the provider to accept a history that violates its own tool-call
    pairing rules, so the salvage attempt fails with a 400 and the branch's
    research is lost for a formatting reason.

    When the transcript is malformed the history is flattened into one
    labelled plain-text block instead (framework nudges and leaked tool-call
    text stripped), which no longer has a protocol to violate. Healthy
    transcripts are replayed as-is, since the real conversation is better
    context than a flattened summary of it.
    """
    healthy = list(messages) + [user_msg(_FINALIZE_INSTRUCTION)]
    try:
        from agent_harness.components.finalization.recovery import (
            COMMON_RECOVERY_NUDGE_PREFIXES,
            build_recovery_context,
            has_malformed_tool_protocol,
        )

        if not has_malformed_tool_protocol(messages):
            return healthy
        context = build_recovery_context(
            messages,
            strip_thinking=_strip_thinking,
            strip_leaked_tool_calls=_strip_leaked_tool_calls,
            nudge_prefixes=COMMON_RECOVERY_NUDGE_PREFIXES,
            empty_fallback="(no usable research transcript)",
        )
        logger.info("subagent finalize: replaying a repaired transcript")
        return [
            system_msg(system_prompt),
            user_msg(f"{context}\n\n{_FINALIZE_INSTRUCTION}"),
        ]
    except Exception as exc:  # noqa: BLE001 — recovery is best-effort
        logger.debug("recovery-context build skipped: %s", exc)
        return healthy


async def gather_with_limit(
    coros: Sequence[Awaitable[Any]],
    limit: int,
) -> list[Any]:
    """Run coroutines with a concurrency cap; exceptions become results.

    Mirrors open_deep_research's supervisor fan-out
    (``asyncio.gather(..., return_exceptions=True)``) with a semaphore —
    partial failure never cancels sibling branches.
    """
    sem = asyncio.Semaphore(max(1, int(limit)))

    async def _bounded(coro: Awaitable[Any]) -> Any:
        async with sem:
            return await coro

    return list(await asyncio.gather(
        *[_bounded(c) for c in coros], return_exceptions=True,
    ))


__all__ = ["gather_with_limit", "run_subagent"]
