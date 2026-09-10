"""Per-tool metadata: concurrency, timeouts, and result-size handling.

Adapted from FrontierAgent (``plugins/tools/meta.py``, Apache-2.0), trimmed
to the tools this repo actually ships. Read by ``_overflow`` to decide how
to truncate an oversized tool result, and available to schedulers that care
whether a tool is safe to run in parallel.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

# Inline cap for exec/compute results before overflow-to-disk. Compute
# output is the one category where the tail carries the verdict and the
# middle is usually noise, so it gets a per-tool cap while content tools
# keep their full bodies.
_EXEC_RESULT_MAX_CHARS = int(
    os.environ.get("TOOL_EXEC_RESULT_MAX_CHARS", "8000") or 8000,
)


@dataclass(frozen=True)
class ToolMeta:
    """Metadata for one tool.

    Attributes:
        is_read_only: Only reads; never modifies state.
        is_destructive: Can cause irreversible changes.
        concurrency_safe: Safe to run in parallel with other tools.
        timeout: Maximum execution time in seconds.
        category: Logical grouping (web, file, compute, search).
        max_result_chars: Characters kept inline; 0 means no per-tool cap.
        result_is_ranked: The result is ordered by relevance, best first, so
            a contiguous head beats a head-plus-tail split — the tail of a
            ranked list is its worst entries. False (the default) means the
            result is sequential (a document, a transcript, an exec log),
            where the end carries the conclusion and is worth keeping.
    """

    is_read_only: bool = True
    is_destructive: bool = False
    concurrency_safe: bool = True
    timeout: int = 45
    category: str = ""
    max_result_chars: int = 0
    result_is_ranked: bool = False


# ``max_result_chars=0`` on the content tools is deliberate: research
# rewards full bodies (a paper buries its result 30K characters in), and
# the loop's own ``tool_result_max_chars`` already bounds total context.
TOOL_META: dict[str, ToolMeta] = {
    "web_search": ToolMeta(
        timeout=30, category="web", result_is_ranked=True,
    ),
    "web_fetch": ToolMeta(timeout=60, category="web"),
    "vault_get": ToolMeta(timeout=15, category="file"),
    "vault_search": ToolMeta(
        timeout=15, category="search", result_is_ranked=True,
    ),
    # Reads this run's own transcript in-process — no sandbox, no network.
    # ``max_result_chars=0`` because the tool paginates its own slice; a
    # per-tool overflow on top would truncate the very content the call
    # exists to recover.
    "recover_result": ToolMeta(timeout=30, category="file"),
    "run_python_code": ToolMeta(
        is_read_only=False,
        concurrency_safe=True,
        timeout=120,
        category="compute",
        max_result_chars=_EXEC_RESULT_MAX_CHARS,
    ),
}


def get_tool_meta(tool_name: str) -> ToolMeta:
    """Metadata for ``tool_name``; safe defaults when it is not registered."""
    return TOOL_META.get(tool_name, ToolMeta())


__all__ = ["TOOL_META", "ToolMeta", "get_tool_meta"]
