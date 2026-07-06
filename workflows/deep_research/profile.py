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

    return OpenAIClient(
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
    # Share one client across roles with identical resolved config.
    cache_key = "|".join(
        str(cfg.get(k, "")) for k in ("model", "base_url", "api_key", "temperature", "max_tokens")
    )
    if cache_key not in clients:
        clients[cache_key] = _build_client(cfg)
        logger.info(
            "deep_research LLM for role=%s: model=%s base_url=%s",
            role_id, cfg.get("model"), cfg.get("base_url"),
        )
    return clients[cache_key]


__all__ = ["get_llm_for_role", "load_profile"]
