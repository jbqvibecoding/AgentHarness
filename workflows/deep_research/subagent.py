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
from typing import Any, Awaitable, Sequence

from agent_harness.components.observers.leaked_tool_call_retry import (
    LeakedToolCallRetryObserver,
)
from agent_harness.core.loop_types import (
    AgentLoopResult,
    LoopConfig,
    LoopPolicy,
)
from agent_harness.core.messages import text_of, user_msg
from agent_harness.core.runtime import registry
from agent_harness.core.runtime.loop.agent_loop import run_agent_loop
from agent_harness.core.runtime.resources.manager import ResourceManager

from workflows.react_base.observers.duplicate_query_rollback import (
    DuplicateQueryRollbackObserver,
)
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


def _build_observers(tool_names: list[str]) -> list[Any]:
    return [
        ToolCallArgsNormalizer(),
        LeakedToolCallRetryObserver(tool_names=tool_names),
        DuplicateQueryRollbackObserver(),
        RefusalRollbackObserver(),
        EmptySearchRollbackObserver(),
    ]


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
) -> AgentLoopResult:
    """Run one isolated ReAct sub-agent and return its loop result.

    Raises ``asyncio.TimeoutError`` on wall-clock overrun and propagates
    loop-level infra errors — callers decide whether a branch failure is
    fatal (fan-out nodes treat it as a failed sub-question).
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

    result = await asyncio.wait_for(
        run_agent_loop(
            system_prompt=system_prompt,
            user_message=user_message,
            llm=llm,
            tools=tools,
            config=config,
            observers=_build_observers(tool_names),
        ),
        timeout=timeout_s,
    )

    result.final_content = _strip_thinking(result.final_content)
    if result.final_content:
        return result

    # Out of turns (or empty visible answer): one tool-free recovery call
    # asking for the required JSON now — simplified react_base
    # ``_force_final_answer``; failure is non-fatal, caller sees "".
    if result.stopped_by in {"max_turns", "context_limit_reached", "no_tool"}:
        try:
            messages = list(result.messages)
            messages.append(user_msg(
                "Stop researching now. Output your findings immediately in "
                "the exact final-answer format required by your "
                "instructions (the fenced JSON block).",
            ))
            resp = await asyncio.wait_for(llm.chat(messages), timeout=300)
            result.final_content = _strip_thinking(text_of(resp.content))
        except Exception as exc:  # noqa: BLE001 — branch stays non-fatal
            logger.warning(
                "subagent forced-summary failed (role=%s): %s", role_id, exc,
            )
    return result


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
