"""Stagnation guard — stop a branch whose tools stop returning new information.

Design adapted from DeerFlow 2.0's ``ToolProgressMiddleware`` (MIT). No
DeerFlow code is used: theirs subclasses LangChain's ``AgentMiddleware``,
which does not exist here. This is the same idea on our ``LoopObserver``.

It is deliberately **complementary** to the repetition guards, not a
duplicate of them. Those watch the *call pattern* — the same tool with the
same arguments, again. This watches the *results*: a researcher that keeps
issuing plausibly-different searches and keeps getting back the same
material is making no progress either, and no call-pattern guard will
notice, because every call really is different.

That is the common shape of a saturated sub-question. The evidence is
already gathered; further turns spend budget re-reading it. Stopping lets
the branch report what it has while its siblings still have budget to
spend — and the additive ``stop_reason`` channel says why, so the report
can disclose that the sub-question was cut short rather than exhausted.

Three states per (tool, run):

* ACTIVE  — results are still bringing something new.
* WARNED  — repeats crossed ``warn_after``; the model is told once,
            because sometimes it simply needs to try a different angle.
* BLOCKED — repeats crossed ``stop_after``; the loop stops.

Similarity is Jaccard over word shingles rather than equality: two search
results that differ only in ordering, ads or a timestamp are the same
information from the researcher's point of view, and exact-match
comparison would never fire on them.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from agent_harness.core.loop_types import (
    BaseObserver,
    Intervention,
    LoopConfig,
    ToolResult,
    TurnContext,
)

from workflows.deep_research.stop_reason import (
    STAGNATION_CAPPED,
    StopReasonRegistry,
)

logger = logging.getLogger(__name__)

_WORD_RE = re.compile(r"[a-z0-9]+")

#: Results are compared on their leading text only. The head carries the
#: identity of a search result page or an article; comparing whole bodies
#: makes long documents look similar merely because they are long.
_COMPARE_CHARS = 4_000

#: Below this, a "result" is an error string or an empty page — too short
#: for a similarity score to mean anything.
_MIN_COMPARE_CHARS = 200

DEFAULT_SIMILARITY = 0.85
DEFAULT_WARN_AFTER = 3
DEFAULT_STOP_AFTER = 5

_WARN_MESSAGE = (
    "Your last few {tool} calls returned essentially the same information. "
    "Either search a genuinely different angle — a different source type, "
    "a different time period, a different phrasing of the question — or, "
    "if you have what you need, stop and report your findings now in the "
    "required final-answer format."
)

_STOP_MESSAGE = (
    "Repeated {tool} calls are returning no new information, so further "
    "searching will not improve this answer. Report your findings now in "
    "the required final-answer format, and say plainly which parts of the "
    "question you could not resolve."
)


def _shingles(text: str) -> frozenset[str]:
    """Word bigrams of ``text``'s head, for order-tolerant comparison."""
    words = _WORD_RE.findall(text[:_COMPARE_CHARS].lower())
    if len(words) < 2:
        return frozenset(words)
    return frozenset(
        f"{words[i]} {words[i + 1]}" for i in range(len(words) - 1)
    )


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a or not b:
        return 0.0
    union = len(a | b)
    return len(a & b) / union if union else 0.0


class ToolProgressGuard(BaseObserver):
    """Stop a loop whose tool results stop carrying new information.

    ``critical`` so the loop awaits the hook and honours the returned
    ``Intervention``; a passive observer's return value is discarded.
    """

    critical: bool = True

    #: Only content-gathering tools are watched. A tool whose job is to
    #: return the same thing every time (a vault lookup of one URL, a
    #: recovery read) is not stagnating when it does so.
    DEFAULT_TOOL_NAMES: frozenset[str] = frozenset({"web_search", "web_fetch"})

    def __init__(
        self,
        *,
        run_id: str = "",
        tool_names: frozenset[str] | set[str] | None = None,
        similarity: float = DEFAULT_SIMILARITY,
        warn_after: int = DEFAULT_WARN_AFTER,
        stop_after: int = DEFAULT_STOP_AFTER,
        registry: StopReasonRegistry | None = None,
    ) -> None:
        self._run_id = run_id
        self._tool_names = frozenset(tool_names or self.DEFAULT_TOOL_NAMES)
        self._similarity = float(similarity)
        self._warn_after = max(1, int(warn_after))
        # stop_after must leave room for the warning to land, otherwise the
        # model is stopped before it ever gets a chance to change approach.
        self._stop_after = max(self._warn_after + 1, int(stop_after))
        self._registry = registry if registry is not None else StopReasonRegistry()
        self._recent: dict[str, frozenset[str]] = {}
        self._repeats: dict[str, int] = {}
        self._warned: set[str] = set()
        self._pending: list[str] = []
        self._blocked_tool = ""

    async def on_loop_start(self, config: LoopConfig) -> None:  # noqa: ARG002
        self._recent.clear()
        self._repeats.clear()
        self._warned.clear()
        self._pending.clear()
        self._blocked_tool = ""

    async def on_tool_result(
        self, ctx: TurnContext, result: ToolResult,
    ) -> Any | None:
        """Score this result against the last one from the same tool.

        Returns ``None`` always — this hook's return value replaces the tool
        result, and rewriting a researcher's evidence would be a far worse
        intervention than stopping the loop. The decision is carried to
        ``on_turn_end`` instead.
        """
        name = result.name
        if name not in self._tool_names or result.is_error:
            return None
        text = result.result if isinstance(result.result, str) else str(result.result)
        if len(text) < _MIN_COMPARE_CHARS:
            return None

        current = _shingles(text)
        previous = self._recent.get(name)
        self._recent[name] = current
        if previous is None:
            return None

        if _jaccard(previous, current) < self._similarity:
            # Genuine new material resets the streak: a researcher that
            # recovers its footing should not carry a penalty forward.
            self._repeats[name] = 0
            self._warned.discard(name)
            return None

        repeats = self._repeats.get(name, 0) + 1
        self._repeats[name] = repeats
        if repeats >= self._stop_after:
            self._blocked_tool = name
            logger.info(
                "ToolProgressGuard: %s returned no new information %d times "
                "(turn %d) — stopping the branch",
                name, repeats, ctx.turn,
            )
        elif repeats >= self._warn_after and name not in self._warned:
            self._warned.add(name)
            self._pending.append(_WARN_MESSAGE.format(tool=name))
        return None

    async def on_turn_end(self, ctx: TurnContext) -> Intervention | None:  # noqa: ARG002
        """Deliver a queued warning, or stop the loop.

        Messages are queued here rather than injected from
        ``on_tool_result`` so they land between turns. Injecting mid-turn
        would split an assistant tool-call message from its results, which
        providers reject outright.
        """
        if self._blocked_tool:
            tool, self._blocked_tool = self._blocked_tool, ""
            self._pending.clear()
            self._registry.record(self._run_id, STAGNATION_CAPPED)
            # Stop, but do not raise: the branch keeps the evidence it
            # already gathered and reports it.
            return Intervention(
                stop_reason="stagnation",
                inject_messages=[_STOP_MESSAGE.format(tool=tool)],
            )
        if self._pending:
            messages, self._pending = self._pending, []
            return Intervention(inject_messages=messages)
        return None

    def consume_stop_reason(self, run_id: str) -> str | None:
        """Additive stop-reason channel; see :mod:`workflows.deep_research.stop_reason`."""
        return self._registry.consume(run_id)


__all__ = ["ToolProgressGuard"]
