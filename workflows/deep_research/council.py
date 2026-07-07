"""Council member management for the multi-model council pipelines.

A "council" is a set of member models (all OpenAI-compatible endpoints)
that independently work the same question — either as full deep_research
sub-runs (``deep_council_research``) or as single direct answers
(``model_council``) — plus a synthesizer model that analyzes agreement /
disagreement / unique findings and writes the combined result.

Members are configured as slots in the profile YAML (env-interpolated):

.. code-block:: yaml

    council:
      members:
        - {name: "${COUNCIL_NAME_1}", model: "${COUNCIL_MODEL_1}",
           base_url: "${COUNCIL_BASE_URL_1}", api_key: "${COUNCIL_API_KEY_1}"}
        # slots 2..N alike — slots with an empty model are skipped
      synthesizer:
        model: "${COUNCIL_SYNTH_MODEL}"   # empty -> default model

Empty ``base_url``/``api_key`` fall back to the profile's ``default``
block, so pointing all members at one OpenRouter-style endpoint only
needs the three ``COUNCIL_MODEL_n`` vars.
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any

from agent_harness.core.llm import LLMClient
from agent_harness.core.messages import system_msg, text_of, user_msg

logger = logging.getLogger(__name__)

MIN_MEMBERS = 2


def _slugify(name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", name).strip("-").lower()
    return slug or "member"


def load_council_members(profile: dict[str, Any]) -> list[dict[str, Any]]:
    """Resolve configured member slots; raise with guidance if < 2 usable."""
    default = profile.get("default") or {}
    raw_members = ((profile.get("council") or {}).get("members")) or []

    members: list[dict[str, Any]] = []
    seen_slugs: set[str] = set()
    for slot in raw_members:
        if not isinstance(slot, dict):
            continue
        model = str(slot.get("model") or "").strip()
        if not model:
            continue
        name = str(slot.get("name") or "").strip() or model
        slug = _slugify(name)
        while slug in seen_slugs:
            slug += "-2"
        seen_slugs.add(slug)
        members.append({
            "name": name,
            "slug": slug,
            "model": model,
            "base_url": str(slot.get("base_url") or "").strip()
            or default.get("base_url"),
            "api_key": str(slot.get("api_key") or "").strip()
            or default.get("api_key"),
            "temperature": slot.get("temperature", default.get("temperature", 0.7)),
            "max_tokens": slot.get("max_tokens", default.get("max_tokens", 8192)),
        })

    if len(members) < MIN_MEMBERS:
        raise ValueError(
            f"Council needs at least {MIN_MEMBERS} member models but "
            f"{len(members)} are configured. Set COUNCIL_MODEL_1 and "
            f"COUNCIL_MODEL_2 (plus COUNCIL_BASE_URL_n / COUNCIL_API_KEY_n "
            f"when they differ from the default OPENAI_* endpoint) in the "
            f"AgentHarness .env.",
        )
    return members


def _build_client(cfg: dict[str, Any]) -> LLMClient:
    from agent_harness.infra.openai_client import OpenAIClient

    return OpenAIClient(
        model=cfg["model"],
        api_key=cfg.get("api_key") or "dummy",
        base_url=cfg.get("base_url"),
        temperature=float(cfg.get("temperature", 0.7)),
        max_completion_tokens=int(cfg.get("max_tokens", 8192)),
        default_headers={
            "HTTP-Referer": "agent_harness",
            "X-Title": "AgentHarness-Council",
        },
    )


def get_llm_for_member(member: dict[str, Any]) -> LLMClient:
    return _build_client(member)


def get_synthesizer_llm(profile: dict[str, Any]) -> LLMClient:
    """Synthesizer/analyst model: council.synthesizer.model or default."""
    default = dict(profile.get("default") or {})
    synth = dict(((profile.get("council") or {}).get("synthesizer")) or {})
    cfg = dict(default)
    cfg.update({k: v for k, v in synth.items() if v not in (None, "")})
    if not cfg.get("model"):
        raise ValueError(
            "Council synthesizer resolves no model — set OPENAI_MODEL "
            "(or COUNCIL_SYNTH_MODEL) in the AgentHarness .env.",
        )
    # Synthesis benefits from low temperature and room for long tables.
    cfg.setdefault("temperature", 0.2)
    cfg["max_tokens"] = max(int(cfg.get("max_tokens", 8192)), 16_384)
    return _build_client(cfg)


def build_member_profile(
    base_profile: dict[str, Any], member: dict[str, Any],
) -> dict[str, Any]:
    """Derive a deep_research profile that forces ALL roles to this member.

    Role blocks keep their behavioral knobs (temperature/max_tokens) but
    lose any per-role ``model``/endpoint override — in a council sub-run
    every role must speak with the member's voice.
    """
    default = dict(base_profile.get("default") or {})
    default.update({
        "model": member["model"],
        "base_url": member.get("base_url"),
        "api_key": member.get("api_key"),
    })
    roles: dict[str, Any] = {}
    for role_id, block in (base_profile.get("roles") or {}).items():
        if not isinstance(block, dict):
            continue
        kept = {
            k: v for k, v in block.items()
            if k not in {"model", "base_url", "api_key"}
        }
        if kept:
            roles[role_id] = kept
    return {"default": default, "roles": roles}


async def chat_with_retries(
    llm: LLMClient,
    *,
    system_prompt: str,
    user_prompt: str,
    retries: int = 3,
    timeout_s: float = 600.0,
    retry_wait_s: float = 15.0,
) -> str:
    """One system+user completion with fixed-wait retries.

    Same skeleton as ``nodes.common.call_role_llm`` but takes an explicit
    client, so council members and the synthesizer (which are not profile
    roles) can share it.
    """
    messages = [system_msg(system_prompt), user_msg(user_prompt)]
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            resp = await asyncio.wait_for(
                llm.chat(messages, timeout=timeout_s), timeout=timeout_s,
            )
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
                "council chat attempt %d/%d failed (model=%s): %s",
                attempt + 1, retries, getattr(llm, "model", "?"), exc,
            )
        if attempt < retries - 1:
            await asyncio.sleep(retry_wait_s)
    raise RuntimeError(
        f"council chat failed after {retries} attempts: {last_exc}",
    ) from last_exc


__all__ = [
    "MIN_MEMBERS",
    "build_member_profile",
    "chat_with_retries",
    "get_llm_for_member",
    "get_synthesizer_llm",
    "load_council_members",
]
