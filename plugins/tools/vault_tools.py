"""Local vault lookup tools for the deep_research workflow.

``vault_get`` / ``vault_search`` read the per-run evidence vault populated
by researchers (see ``workflows/deep_research/vault.py``). They are purely
local (no network): the vault directory for the current sub-agent run is
resolved from the agent loop's ExecutionScope metadata
(``vault_dir``). When no vault is active they return a clear message so
the caller falls back to ``web_fetch``.

Registered as builtin tools; only roles that list them in
``allowed_tools`` (fact-checker, verifier) can call them.
"""

from __future__ import annotations

import logging

from agent_harness.core.tool import tool

logger = logging.getLogger(__name__)

_NO_VAULT = (
    "[vault unavailable] No evidence vault is active for this run. "
    "Use web_fetch to read the source directly."
)


def _handle() -> object | None:
    from workflows.deep_research.vault import VaultHandle, open_vault, vault_dir_from_scope

    vault_dir = vault_dir_from_scope()
    if not vault_dir:
        return None
    # The vault already exists on disk (researchers created it); open_vault
    # here just attaches a read handle to the same directory.
    handle = open_vault(vault_dir, enabled=True)
    if isinstance(handle, VaultHandle):
        return handle
    return None


@tool(name="vault_get")
async def vault_get(url: str) -> str:
    """Return the stored full text of a source already fetched in this research run.

    Prefer this over web_fetch when verifying a claim whose source URL was
    gathered earlier — it re-reads the exact text without another network
    fetch. Returns a not-found message if the URL is not in the vault.

    Args:
        url: The source URL to look up.
    """
    handle = _handle()
    if handle is None:
        return _NO_VAULT
    body = handle.get_source(url)
    handle.close()
    if not body:
        return (
            f"[not in vault] {url} was not stored in this run's evidence "
            f"vault. Use web_fetch to read it."
        )
    return body


@tool(name="vault_search")
async def vault_search(query: str) -> str:
    """Full-text search the sources already gathered in this research run.

    Returns matching stored sources (title + snippet + id) from the run's
    evidence vault. Useful to check whether a fact is already covered
    before fetching new pages.

    Args:
        query: Free-text search query (supports AND/OR/NOT).
    """
    handle = _handle()
    if handle is None:
        return _NO_VAULT
    rows = handle.search(query, k=5)
    handle.close()
    if not rows:
        return "[vault] No stored sources match that query."
    lines = [
        f"- {r['title']} (id={r['id']}): {r['snippet']}" for r in rows
    ]
    return "Stored sources matching your query:\n" + "\n".join(lines)


__all__ = ["vault_get", "vault_search"]
