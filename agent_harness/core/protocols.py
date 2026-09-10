"""Protocols + dataclasses shared by ``core/runtime`` and ``components/``.

Two structural items live here:

- ``PhaseContext`` / ``ToolCallContext`` dataclasses + the
  ``ExecutionMiddleware`` ABC, consumed by the middleware base in
  ``components.middleware``.
- ``PhaseMiddlewareChain`` Protocol, looked up via the service registry
  by ``core.runtime.dag.graph_builder``.
"""

from __future__ import annotations

import time
from abc import ABC
from dataclasses import dataclass, field
from collections.abc import AsyncIterator
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class EventSink(Protocol):
    """Append-only event writer. Structural so any store can satisfy it."""

    async def append(
        self,
        task_id: Any = "",
        event_type: Any = None,
        payload: dict[str, Any] | None = None,
        agent_role: str = "system",
    ) -> Any: ...

    def replay(self, task_id: str) -> AsyncIterator[Any]: ...


@runtime_checkable
class EventReader(Protocol):
    """Query side of the event store, including per-agent inboxes."""

    async def get_events(
        self,
        task_id: Any,
        event_type: Any = None,
        after_id: int = 0,
        limit: int | None = None,
    ) -> list[Any]: ...

    async def get_events_for_agent(
        self,
        to_agent: str,
        after_id: int = 0,
        limit: int = 50,
        *,
        task_id: str | Any | None = None,
    ) -> list[Any]: ...


@dataclass
class PhaseContext:
    """Context passed through phase-level middleware."""

    task_id: str
    phase_id: str
    role_id: str = ""
    display_label: str = ""
    state: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    start_time: float = 0.0

    def __post_init__(self) -> None:
        if self.start_time == 0.0:
            self.start_time = time.time()


@dataclass
class ToolCallContext:
    """Context passed through tool-call middleware."""

    task_id: str
    phase_id: str
    role_id: str = ""
    tool_name: str = ""
    tool_args: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


class ExecutionMiddleware(ABC):
    """Base class for execution middleware hooks.

    Default implementations are pass-through; subclasses override only
    the hooks they care about.
    """

    async def before_phase(self, ctx: PhaseContext) -> PhaseContext:
        return ctx

    async def after_phase(
        self,
        ctx: PhaseContext,
        result: dict[str, Any],
    ) -> dict[str, Any]:
        return result

    async def before_tool_call(self, ctx: ToolCallContext) -> ToolCallContext:
        return ctx

    async def after_tool_call(self, ctx: ToolCallContext, result: str) -> str:
        return result

    async def on_error(
        self,
        ctx: PhaseContext,
        error: Exception,
    ) -> Exception | None:
        return error


@runtime_checkable
class PhaseMiddlewareChain(Protocol):
    """Phase/tool middleware chain consumed by ``core/runtime``."""

    @property
    def middlewares(self) -> list[ExecutionMiddleware]: ...

    async def run_before_phase(self, ctx: PhaseContext) -> PhaseContext: ...

    async def run_after_phase(
        self,
        ctx: PhaseContext,
        result: dict[str, Any],
    ) -> dict[str, Any]: ...

    async def run_on_error(
        self,
        ctx: PhaseContext,
        error: Exception,
    ) -> Exception | None: ...


__all__ = [
    "ExecutionMiddleware",
    "PhaseContext",
    "PhaseMiddlewareChain",
    "ToolCallContext",
]
