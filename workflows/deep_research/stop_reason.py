"""Additive stop-reason reporting for capped sub-agent branches.

Design adapted from DeerFlow 2.0 (MIT), whose guardrail middlewares each
expose ``consume_stop_reason(run_id)`` and whose executor collects whichever
fired. No DeerFlow code is used — its middlewares are LangGraph-bound; this
is the same idea rebuilt on our ``LoopObserver`` protocol.

Two ideas are worth restating because they are what make capped branches
useful instead of merely survivable.

**Additive, not a new status.** A branch can be cut short by any of several
independent limits — tokens, turns, repeated tool calls, wall clock, result
stagnation. Modelling each as a new ``status`` value multiplies the states
every consumer must handle and breaks the ones written before it existed.
Instead ``status`` keeps its existing vocabulary and an *optional*
``stop_reason`` rides alongside: readers that know about it get the detail,
readers that don't ignore an unknown key.

**A capped branch still reports.** Guards here never raise. The loop is
allowed to finish its turn and hand back whatever evidence it gathered —
half a sub-question's worth of cited evidence is worth a great deal more
than an exception that discards it.

Guards are collected structurally, by looking for the method, so adding a
sixth guard later needs no change here or in the runner.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)

# Vocabulary. Keep these stable — they reach result.json and the report's
# Verification appendix, and downstream readers match on them.
TOKEN_CAPPED = "token_capped"
TURN_CAPPED = "turn_capped"
LOOP_CAPPED = "loop_capped"
WALL_CAPPED = "wall_capped"
STAGNATION_CAPPED = "stagnation_capped"

STOP_REASONS: tuple[str, ...] = (
    TOKEN_CAPPED, TURN_CAPPED, LOOP_CAPPED, WALL_CAPPED, STAGNATION_CAPPED,
)

#: ``AgentLoopResult.stopped_by`` values that mean "a limit cut this short",
#: mapped to the additive reason. ``stopped_by`` is the loop engine's own
#: field and is not extensible by observers, so this translates it.
_STOPPED_BY_TO_REASON: dict[str, str] = {
    "budget_exhausted": TOKEN_CAPPED,
    "context_limit_reached": TOKEN_CAPPED,
    "max_turns": TURN_CAPPED,
    "max_attempts": TURN_CAPPED,
    "repeated_tool_calls": LOOP_CAPPED,
    "cross_turn_repetition": LOOP_CAPPED,
    "thrash_no_progress": LOOP_CAPPED,
    "wall_deadline": WALL_CAPPED,
}

_MAX_TRACKED_RUNS = 512


@runtime_checkable
class StopReasonGuard(Protocol):
    """Structural contract for a guard that can cap a run.

    Deliberately not a base class: guards live in different packages and
    must not import each other or this module to participate.
    """

    def consume_stop_reason(self, run_id: str) -> str | None:
        """Return and clear this guard's reason for capping ``run_id``."""
        ...


class StopReasonRegistry:
    """Bounded, pop-once store one guard can use to record why it capped.

    Pop-once matters: the runner reads the reason *after* the loop returns,
    so it has to outlive the run's own cleanup, and it must not leak into
    the next run that happens to reuse the id.
    """

    def __init__(self, max_runs: int = _MAX_TRACKED_RUNS) -> None:
        self._reasons: OrderedDict[str, str] = OrderedDict()
        self._max_runs = max_runs

    def record(self, run_id: str, reason: str) -> None:
        """Record ``reason`` for ``run_id``. First writer wins.

        First-writer-wins because the first limit to fire is the one that
        actually shaped the run; a later guard noticing the same truncated
        state would otherwise overwrite the real cause.
        """
        if not run_id or run_id in self._reasons:
            return
        self._reasons[run_id] = reason
        while len(self._reasons) > self._max_runs:
            self._reasons.popitem(last=False)

    def consume(self, run_id: str) -> str | None:
        return self._reasons.pop(run_id, None)

    def clear(self, run_id: str) -> None:
        self._reasons.pop(run_id, None)


def collect_stop_reason(observers: list[Any], run_id: str) -> str | None:
    """Ask every guard in ``observers`` why it capped ``run_id``.

    Returns the first reason found, in observer order, and drains the rest
    so nothing is left behind for a later run. Guards are detected by the
    presence of the method rather than by type, so a guard added later
    participates without touching this function.
    """
    found: str | None = None
    for obs in observers or ():
        consume = getattr(obs, "consume_stop_reason", None)
        if not callable(consume):
            continue
        try:
            reason = consume(run_id)
        except Exception as exc:  # noqa: BLE001 — telemetry must never break a run
            logger.debug("stop-reason guard %r failed: %s", type(obs).__name__, exc)
            continue
        if reason and found is None:
            found = reason
    return found


def reason_from_stopped_by(stopped_by: str | None) -> str | None:
    """Translate the loop engine's ``stopped_by`` into an additive reason.

    Covers the limits enforced by the loop itself rather than by an
    observer, so a branch that ran out of turns reports the same way as one
    a guard cut short.
    """
    return _STOPPED_BY_TO_REASON.get((stopped_by or "").strip())


def describe(reason: str | None) -> str:
    """One-line human phrasing, for the report's Verification appendix."""
    return {
        TOKEN_CAPPED: "stopped early: token budget exhausted",
        TURN_CAPPED: "stopped early: ran out of research turns",
        LOOP_CAPPED: "stopped early: repeating itself",
        WALL_CAPPED: "stopped early: out of time",
        STAGNATION_CAPPED: "stopped early: searches stopped returning new information",
    }.get(reason or "", "")


__all__ = [
    "LOOP_CAPPED",
    "STAGNATION_CAPPED",
    "STOP_REASONS",
    "StopReasonGuard",
    "StopReasonRegistry",
    "TOKEN_CAPPED",
    "TURN_CAPPED",
    "WALL_CAPPED",
    "collect_stop_reason",
    "describe",
    "reason_from_stopped_by",
]
