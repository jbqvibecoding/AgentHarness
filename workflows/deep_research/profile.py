"""Per-role LLM configuration for ``deep_research``.

Profiles live in ``workflows/deep_research/profiles/*.yaml``:

.. code-block:: yaml

    default:
      model: ${OPENAI_MODEL}
      api_key: ${OPENAI_API_KEY}
      base_url: ${OPENAI_BASE_URL}
      temperature: 0.7
      max_tokens: 8192
    roles:                    # optional per-role overrides, merged over default
      dr_fact_checker:
        model: ${DR_FACT_CHECK_MODEL:-${OPENAI_MODEL}}

``get_llm_for_role`` merges ``default`` with the role's override block and
builds an ``OpenAIClient``. Roles without an override share the default
client instance (cached per profile), so a single-model setup costs one
client.

A ``fallback`` list adds provider failover. Each entry is a partial config
merged over the same resolved role config, so a second endpoint usually
needs only the keys that differ:

.. code-block:: yaml

    default:
      model: ${OPENAI_MODEL}
      fallback:
        - model: ${DR_FALLBACK_MODEL}
          base_url: ${DR_FALLBACK_BASE_URL}
          api_key: ${DR_FALLBACK_API_KEY}
          provider: anthropic
          triggers: [retriable]      # default; see infra.llm.fallback

Entries whose model resolves empty are dropped, so a profile can name a
fallback that only exists when its env vars are set. Without a usable
entry the role gets the plain client and nothing changes.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from agent_harness.core.llm import LLMClient

logger = logging.getLogger(__name__)

_PROFILES_DIR = Path(__file__).resolve().parent / "profiles"
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# {profile_name: (profile_dict, {cache_key: LLMClient})}
_cache: dict[str, tuple[dict[str, Any], dict[str, LLMClient]]] = {}


def load_profile(name: str) -> dict[str, Any]:
    """Load a profile YAML with ``${VAR:-default}`` env resolution."""
    import yaml
    from dotenv import load_dotenv

    from agent_harness.infra.config import _resolve_env_vars

    load_dotenv(_PROJECT_ROOT / ".env", override=False)

    path = _PROFILES_DIR / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"deep_research profile not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return _resolve_env_vars(raw)


def _build_client(cfg: dict[str, Any]) -> LLMClient:
    from agent_harness.infra.openai_client import OpenAIClient
    from agent_harness.infra.prompt_cache import maybe_wrap_for_prompt_cache

    client = OpenAIClient(
        model=cfg["model"],
        api_key=cfg.get("api_key") or "dummy",
        base_url=cfg.get("base_url"),
        temperature=float(cfg.get("temperature", 0.7)),
        max_completion_tokens=int(cfg.get("max_tokens", 8192)),
        default_headers={
            "HTTP-Referer": "agent_harness",
            "X-Title": "AgentHarness-DeepResearch",
        },
    )
    # Claude-family upstreams bill a re-sent system prompt every turn. A
    # research sub-agent replays the same long role prompt across a dozen
    # turns, so marking it cacheable is a pure saving. No-op on every other
    # provider, so this needs no gating at the call site.
    return maybe_wrap_for_prompt_cache(
        client, provider=str(cfg.get("provider") or ""), model=str(cfg["model"]),
    )


def _fallback_entries(cfg: dict[str, Any]) -> list[Any]:
    """Build ``FallbackEntry`` tiers from a role config's ``fallback`` list.

    Each entry inherits the role's resolved config, so a tier only has to
    name what differs. Tiers with no model — the usual shape when the
    profile references env vars that are not set — are dropped rather than
    built into a client that would fail on first use.
    """
    from agent_harness.infra.llm import FallbackEntry

    raw = cfg.get("fallback")
    if not isinstance(raw, list):
        return []
    entries: list[Any] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        merged = {k: v for k, v in cfg.items() if k != "fallback"}
        merged.update({
            k: v for k, v in item.items()
            if v is not None and v != "" and k not in ("provider", "triggers")
        })
        if not merged.get("model"):
            continue
        triggers = item.get("triggers") or ["retriable"]
        entries.append(FallbackEntry(
            model=_build_client(merged),
            triggers=tuple(str(t) for t in triggers),
            provider=str(item.get("provider") or ""),
        ))
    return entries


def _role_cfg(profile: dict[str, Any], role_id: str) -> dict[str, Any]:
    """Merge the role's override block over ``default``.

    Empty-string values are skipped so YAML like ``model: ${DR_X_MODEL}``
    degrades to the default when the env var is unset (the env resolver
    substitutes "" — it does not support nested ``${A:-${B}}`` defaults).
    """
    base = dict(profile.get("default") or {})
    override = (profile.get("roles") or {}).get(role_id) or {}
    base.update({
        k: v for k, v in override.items() if v is not None and v != ""
    })
    return base


def register_runtime_profile(name: str, profile: dict[str, Any]) -> None:
    """Register an in-memory profile (no YAML file), overwriting any prior.

    Used by the council pipelines to force every deep_research role onto a
    member model: the fan-out registers ``council/<slug>`` profiles and
    sub-runs select them via ``metadata["profile"]``.
    """
    _cache[name] = (profile, {})


def get_llm_for_role(role_id: str, profile_name: str = "default") -> LLMClient:
    """Return the (cached) LLM client for a role under a profile."""
    if profile_name not in _cache:
        _cache[profile_name] = (load_profile(profile_name), {})
    profile, clients = _cache[profile_name]

    cfg = _role_cfg(profile, role_id)
    if not cfg.get("model"):
        raise ValueError(
            f"deep_research profile '{profile_name}' resolves no model for "
            f"role '{role_id}' — set OPENAI_MODEL in .env or a 'model' key "
            f"in the profile."
        )
    # Share one client across roles with identical resolved config. The
    # fallback list is part of the identity: two roles on the same primary
    # but different failover tiers are different clients.
    cache_key = "|".join(
        str(cfg.get(k, "")) for k in (
            "model", "base_url", "api_key", "temperature", "max_tokens",
            "provider", "fallback",
        )
    )
    if cache_key not in clients:
        primary = _build_client(cfg)
        tiers = _fallback_entries(cfg)
        if tiers:
            from agent_harness.infra.llm import FallbackEntry, LLMFallbackChain

            client: LLMClient = LLMFallbackChain(
                entries=[
                    FallbackEntry(
                        model=primary,
                        triggers=("retriable",),
                        provider=str(cfg.get("provider") or ""),
                    ),
                    *tiers,
                ],
            )
            logger.info(
                "deep_research LLM for role=%s: model=%s base_url=%s "
                "(+%d fallback tier(s))",
                role_id, cfg.get("model"), cfg.get("base_url"), len(tiers),
            )
        else:
            client = primary
            logger.info(
                "deep_research LLM for role=%s: model=%s base_url=%s",
                role_id, cfg.get("model"), cfg.get("base_url"),
            )
        clients[cache_key] = client
    return clients[cache_key]


__all__ = ["get_llm_for_role", "load_profile"]
