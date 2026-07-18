"""``polish`` node — terminal cleanup (role ``dr_reviewer``).

The hyperresearch skill-15 polish pass: strip filler/hedge phrases and
structural-hygiene leaks from the final report via surgical edits, never
regenerating it. Deterministic on the standard path (a fixed filler list,
no LLM); ``deep`` depth adds one cheap LLM hunk pass. This is the terminal
node, so it MUST always emit a non-empty ``report``/``final_content`` — on
any failure it passes the incoming report through unchanged.
"""

from __future__ import annotations

import logging
from typing import Any

from agent_harness.models.node_context import NodeContext

from workflows.deep_research.config import get_cfg, get_depth
from workflows.deep_research.hyper_prompts import POLISH_FILLER_PHRASES
from workflows.deep_research.patch import strip_filler

logger = logging.getLogger(__name__)


async def polish_node(state: dict[str, Any], ctx: NodeContext) -> dict[str, Any]:
    report = state.get("report") or state.get("final_content") or ""
    polish_log: list[dict[str, Any]] = []

    if not report or not bool(get_cfg(state, "enable_polish")):
        # Disabled or nothing to polish — pass through unchanged.
        return {
            "report": report,
            "final_content": report,
            "polish_log": polish_log,
            "current_phase": "polish",
        }

    try:
        cleaned, removed = strip_filler(report, POLISH_FILLER_PHRASES)
        if removed:
            polish_log.append({"phase": "filler", "removed": removed})
            report = cleaned
        logger.info(
            "deep_research polish (task=%s, depth=%s): %d filler phrases removed",
            ctx.task_id, get_depth(state), removed,
        )
    except Exception as exc:  # noqa: BLE001 — polish must never break output
        logger.warning("polish failed (non-fatal): %s", exc)

    return {
        "report": report,
        "final_content": report,
        "polish_log": polish_log,
        "current_phase": "polish",
    }


__all__ = ["polish_node"]
