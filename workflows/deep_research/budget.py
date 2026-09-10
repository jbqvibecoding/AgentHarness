"""Cumulative token budget for one ``deep_research`` run.

The framework's :class:`~agent_harness.components.observers.budget_observer.BudgetObserver`
caps a *single* agent loop. A deep-research run is dozens of loops — one per
researcher branch, per fact-check batch, per council member — plus a
double-digit number of single-shot node calls, and nothing was adding them
up. A run could therefore cost an unbounded amount as long as no individual
loop misbehaved, which is precisely the shape a fan-out pipeline produces.

This module adds the missing layer: one :class:`RunBudget` per task, shared
by every branch, holding the run's :class:`~agent_harness.models.task_budget.TaskBudget`.

Two limits apply to a branch, and the tighter one wins:

* the **run total**, so parallel branches cannot collectively overshoot;
* a **per-branch slice**, so one runaway branch cannot eat the whole run
  before its siblings have spent anything.

Exhaustion never raises. :class:`RunBudgetObserver` stops its loop the way
the framework observer does — a stop reason and an injected message — so
the branch finishes its turn and returns the evidence it already gathered.
It reports that cap through the additive ``stop_reason`` channel in
:mod:`workflows.deep_research.stop_reason`.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

from agent_harness.core.loop_types import Intervention, LoopConfig, TurnContext
from agent_harness.models.task_budget import TaskBudget

from workflows.deep_research.config import get_cfg
from workflows.deep_research.stop_reason import TOKEN_CAPPED, StopReasonRegistry

logger = logging.getLogger(__name__)

# task_id -> RunBudget. Branches run as coroutines in this process, so a
# plain dict under a lock is enough; a contextvar would not survive the
# hop into ``asyncio.gather``-spawned branches without extra plumbing.
_RUN_BUDGETS: dict[str, RunBudget] = {}
_REGISTRY_LOCK = threading.Lock()


def _usage_tokens(usage: dict[str, Any] | None) -> int:
    """Sum one response's tokens, tolerating both key spellings.

    OpenAI-compatible endpoints report ``prompt_tokens`` /
    ``completion_tokens``; the Anthropic-shaped path reports
    ``input_tokens`` / ``output_tokens``. Reading only one pair silently
    counts zero on half the providers — which is how a budget guard ends up
    never firing at all.
    """
    if not usage:
        return 0
    prompt = usage.get("input_tokens")
    if prompt is None:
        prompt = usage.get("prompt_tokens", 0)
    completion = usage.get("output_tokens")
    if completion is None:
        completion = usage.get("completion_tokens", 0)
    try:
        return int(prompt or 0) + int(completion or 0)
    except (TypeError, ValueError):
        return 0


class RunBudget:
    """Token accounting shared by every loop in one run."""

    def __init__(self, budget: TaskBudget) -> None:
        self.budget = budget
        self._lock = threading.Lock()
        self._spent = 0
        self._by_role: dict[str, int] = {}
        self.stop_reasons = StopReasonRegistry()

    @property
    def max_tokens(self) -> int:
        return int(self.budget.max_tokens)

    @property
    def spent(self) -> int:
        with self._lock:
            return self._spent

    @property
    def remaining(self) -> int:
        return max(0, self.max_tokens - self.spent)

    @property
    def exhausted(self) -> bool:
        return self.max_tokens > 0 and self.spent >= self.max_tokens

    @property
    def branch_slice(self) -> int:
        """Per-branch allowance: the run total split across parallel slots.

        Sized off ``max_parallel`` rather than the number of sub-questions:
        it is the number of branches actually in flight that determines how
        fast the shared pool drains.
        """
        slots = max(1, int(self.budget.max_parallel))
        return max(1, self.max_tokens // slots)

    def spend(self, tokens: int, *, role_id: str = "") -> None:
        if tokens <= 0:
            return
        with self._lock:
            self._spent += tokens
            if role_id:
                self._by_role[role_id] = self._by_role.get(role_id, 0) + tokens

    def summary(self) -> dict[str, Any]:
        with self._lock:
            return {
                "max_tokens": self.max_tokens,
                "spent_tokens": self._spent,
                "remaining_tokens": max(0, self.max_tokens - self._spent),
                "exhausted": self.max_tokens > 0 and self._spent >= self.max_tokens,
                "by_role": dict(sorted(self._by_role.items())),
            }


def budget_from_state(state: dict[str, Any]) -> TaskBudget:
    """Build the run's ``TaskBudget`` from the depth-resolved config."""
    return TaskBudget(
        max_tokens=int(get_cfg(state, "max_run_tokens")),
        max_parallel=int(get_cfg(state, "max_parallel_subagents")),
        max_wall_time_s=int(get_cfg(state, "subagent_timeout_s")),
    )


def get_run_budget(task_id: str, state: dict[str, Any] | None = None) -> RunBudget | None:
    """Fetch (or, given ``state``, create) the budget for ``task_id``.

    Returns ``None`` when no budget exists and none can be created, so
    callers can treat budgeting as optional rather than branching on config
    themselves.
    """
    if not task_id:
        return None
    with _REGISTRY_LOCK:
        existing = _RUN_BUDGETS.get(task_id)
        if existing is not None or state is None:
            return existing
        budget = RunBudget(budget_from_state(state))
        _RUN_BUDGETS[task_id] = budget
        logger.info(
            "deep_research budget (task=%s): %d tokens, %d per branch",
            task_id, budget.max_tokens, budget.branch_slice,
        )
        return budget


def release_run_budget(task_id: str) -> dict[str, Any] | None:
    """Drop the run's budget and return its final summary."""
    with _REGISTRY_LOCK:
        budget = _RUN_BUDGETS.pop(task_id, None)
    return budget.summary() if budget is not None else None


class RunBudgetObserver:
    """Charge one loop's usage to the run budget and stop it when spent.

    ``critical = True`` so the loop awaits it and honours the returned
    ``Intervention`` — a fire-and-forget observer could not stop anything.
    """

    critical = True

    def __init__(
        self,
        run_budget: RunBudget,
        *,
        run_id: str,
        role_id: str = "",
        branch_limit: int | None = None,
        warn_ratio: float = 0.8,
    ) -> None:
        self._budget = run_budget
        self._run_id = run_id
        self._role_id = role_id
        self._branch_limit = (
            int(branch_limit) if branch_limit else run_budget.branch_slice
        )
        self._warn_ratio = warn_ratio
        self._branch_spent = 0
        self._warned = False

    # -- LoopObserver hooks ------------------------------------------------

    async def on_loop_start(self, config: LoopConfig) -> None:  # noqa: ARG002
        self._branch_spent = 0
        self._warned = False

    async def on_llm_response(self, ctx: TurnContext) -> None:
        tokens = _usage_tokens(ctx.usage)
        if not tokens:
            return
        self._branch_spent += tokens
        self._budget.spend(tokens, role_id=self._role_id)
        ctx.metadata["run_budget_spent"] = self._budget.spent
        ctx.metadata["run_budget_limit"] = self._budget.max_tokens
        ctx.metadata["branch_budget_spent"] = self._branch_spent

    async def on_turn_end(self, ctx: TurnContext) -> Intervention | None:  # noqa: ARG002
        run_over = self._budget.exhausted
        branch_over = (
            self._branch_limit > 0 and self._branch_spent >= self._branch_limit
        )
        if run_over or branch_over:
            self._budget.stop_reasons.record(self._run_id, TOKEN_CAPPED)
            scope = "run" if run_over else "branch"
            # Stop, but let the turn finish: the branch keeps whatever
            # evidence it already gathered instead of losing it to a raise.
            return Intervention(
                stop_reason="budget_exhausted",
                inject_messages=[
                    f"Token budget exhausted ({scope}: "
                    f"{self._budget.spent:,}/{self._budget.max_tokens:,} run, "
                    f"{self._branch_spent:,}/{self._branch_limit:,} branch). "
                    "Stop researching and report what you have found so far "
                    "in the required final-answer format now.",
                ],
            )

        if not self._warned and self._budget.max_tokens > 0:
            ratio = self._budget.spent / self._budget.max_tokens
            if ratio >= self._warn_ratio:
                self._warned = True
                return Intervention(inject_messages=[
                    f"{int(ratio * 100)}% of the run's token budget is spent. "
                    "Start converging — gather only what you still need.",
                ])
        return None

    # -- additive stop-reason channel --------------------------------------

    def consume_stop_reason(self, run_id: str) -> str | None:
        return self._budget.stop_reasons.consume(run_id)


__all__ = [
    "RunBudget",
    "RunBudgetObserver",
    "budget_from_state",
    "get_run_budget",
    "release_run_budget",
]
