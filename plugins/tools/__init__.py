"""AgentHarness built-in tools.

Exposes the three tools the ``react_base`` workflow binds to its
``react_solver`` agent role: ``web_search`` / ``web_fetch`` /
``run_python_code``.
"""

from __future__ import annotations

from agent_harness.core.tool import Tool
from plugins.tools.run_python_code import run_python_code
from plugins.tools.vault_tools import vault_get, vault_search
from plugins.tools.web_fetch import web_fetch
from plugins.tools.web_search import web_search


_BUILTIN_TOOLS: list[Tool] = [
    web_search,
    web_fetch,
    run_python_code,
    # Local evidence-vault lookups for the deep_research workflow.
    # No-op safely when no vault is active (see vault_tools).
    vault_get,
    vault_search,
]


def get_builtin_tools() -> dict[str, Tool]:
    """Return all built-in tools as a name → tool dict."""
    return {t.name: t for t in _BUILTIN_TOOLS}


__all__ = [
    "get_builtin_tools",
    "run_python_code",
    "vault_get",
    "vault_search",
    "web_fetch",
    "web_search",
]
