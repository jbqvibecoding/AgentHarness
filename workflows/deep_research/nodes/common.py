"""Helpers shared by the ``deep_research`` node functions."""

from __future__ import annotations

import asyncio
import logging
import re
from datetime import date
from typing import Any

from agent_harness.core.messages import system_msg, text_of, user_msg

logger = logging.getLogger(__name__)


def profile_name(state: dict[str, Any]) -> str:
    return (state.get("metadata") or {}).get("profile") or "default"


def today() -> str:
    return date.today().isoformat()


def _charge_budget(
    state: dict[str, Any], role_id: str, usage: dict[str, int] | None,
) -> None:
    """Charge a single-shot node call to the run's token budget.

    The planner, conflict-checker, writer, reviewer and verifier never run
    an agent loop, so the loop-level budget observer never sees them — yet
    the writer's evidence block is one of the largest prompts in the run.
    Leaving them uncounted would understate spend badly enough to make the
    budget meaningless.
    """
    task_id = str(state.get("task_id") or "")
    if not task_id:
        return
    try:
        from workflows.deep_research.budget import _usage_tokens, get_run_budget

        budget = get_run_budget(task_id, state)
        if budget is not None:
            budget.spend(_usage_tokens(usage), role_id=role_id)
    except Exception as exc:  # noqa: BLE001 — accounting must never fail a node
        logger.debug("budget accounting skipped for role=%s: %s", role_id, exc)


async def call_role_llm(
    *,
    role_id: str,
    state: dict[str, Any],
    system_prompt: str,
    user_prompt: str,
    retries: int = 3,
    timeout_s: float = 600.0,
) -> str:
    """One system+user chat completion for a single-shot role node.

    Retries transient failures with a short fixed wait; raises after the
    last attempt so callers can apply their node-specific fallback.
    """
    from workflows.deep_research.profile import get_llm_for_role

    llm = get_llm_for_role(role_id, profile_name(state))
    messages = [system_msg(system_prompt), user_msg(user_prompt)]
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            resp = await asyncio.wait_for(
                llm.chat(messages, timeout=timeout_s), timeout=timeout_s,
            )
            _charge_budget(state, role_id, getattr(resp, "usage", None))
            text = text_of(resp.content) or ""
            if "</think>" in text:
                text = text.rsplit("</think>", 1)[-1]
            text = text.strip()
            if text:
                return text
            last_exc = RuntimeError("LLM returned empty content")
        except Exception as exc:  # noqa: BLE001 — retried below
            last_exc = exc
            logger.warning(
                "deep_research %s LLM attempt %d/%d failed: %s",
                role_id, attempt + 1, retries, exc,
            )
        if attempt < retries - 1:
            await asyncio.sleep(15)
    raise RuntimeError(
        f"deep_research role '{role_id}' LLM failed after {retries} "
        f"attempts: {last_exc}",
    ) from last_exc


def _domain(url: str) -> str:
    m = re.match(r"^(?:https?://)?([^/]+)", url or "")
    return m.group(1) if m else ""


def claim_table(state: dict[str, Any], max_rows: int = 200) -> list[dict[str, Any]]:
    """Compact per-claim view joining cards with their latest verdicts.

    Deliberately small (claim capped at 200 chars, domains not full URLs)
    so conflict_check / review / verify prompts stay within budget.
    """
    verdicts: dict[str, dict[str, Any]] = {}
    for r in state.get("fact_check_results") or []:
        verdicts[str(r.get("card_id"))] = r

    rows: list[dict[str, Any]] = []
    for card in state.get("evidence_cards") or []:
        card_id = str(card.get("card_id"))
        v = verdicts.get(card_id) or {}
        rows.append({
            "card_id": card_id,
            "claim": (v.get("corrected_claim") or card.get("claim") or "")[:200],
            "verdict": v.get("verdict", "unverified"),
            "domains": sorted({
                _domain(s.get("url", "")) for s in card.get("sources") or []
            } - {""}),
            "sub_question_id": card.get("sub_question_id", ""),
            "iteration": card.get("iteration", 0),
        })
        if len(rows) >= max_rows:
            break
    return rows


def usable_cards(state: dict[str, Any]) -> list[dict[str, Any]]:
    """Cards fit for writing: everything except contradicted/unsupported,
    with corrected claims substituted in."""
    verdicts = {
        str(r.get("card_id")): r for r in state.get("fact_check_results") or []
    }
    out: list[dict[str, Any]] = []
    for card in state.get("evidence_cards") or []:
        v = verdicts.get(str(card.get("card_id"))) or {}
        verdict = v.get("verdict", "unverified")
        merged = dict(card)
        merged["verdict"] = verdict
        if v.get("corrected_claim"):
            merged["claim"] = v["corrected_claim"]
        out.append(merged)
    return out


__all__ = [
    "call_role_llm",
    "claim_table",
    "profile_name",
    "today",
    "usable_cards",
]
