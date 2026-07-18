"""Persistent evidence vault for the deep_research workflow.

A thin adapter over the ``hyperresearch`` engine
(https://github.com/jordan-gibbs/hyperresearch): every page a researcher
fetches is written as a markdown note with YAML frontmatter under
``<out>/vault/research/notes/``, indexed into SQLite FTS5, and
deduplicated by URL (plus a MinHash near-duplicate guard). Fact-checkers
and the verifier then re-read the stored source body from the vault
instead of re-fetching it — cheaper, faster, and shared across research
iterations and across council member sub-runs.

Everything here is lazy and optional: ``hyperresearch`` is an optional
dependency (``pip install agent-harness[vault]``). When it is not
installed, or the vault is disabled, every function degrades to a no-op
and the pipeline behaves exactly as it did before the vault existed.
"""

from __future__ import annotations

import hashlib
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_NEAR_DUP_JACCARD = 0.85
_MAX_BODY_CHARS = 200_000


def _hyperresearch_available() -> bool:
    try:
        import hyperresearch  # noqa: F401
    except Exception:
        return False
    return True


def _domain(url: str) -> str:
    m = re.match(r"^(?:https?://)?([^/]+)", url or "")
    return (m.group(1) if m else "").lower()


def _url_slug(url: str) -> str:
    """Deterministic filename stem for a URL — gives exact-URL dedup."""
    dom = re.sub(r"[^a-z0-9]+", "-", _domain(url)).strip("-") or "src"
    h = hashlib.sha1((url or "").encode("utf-8")).hexdigest()[:10]
    return f"{dom}-{h}"


class VaultHandle:
    """Wraps a hyperresearch ``Vault`` plus its notes directory."""

    def __init__(self, vault: Any, notes_dir: Path, root: Path) -> None:
        self._vault = vault
        self._notes_dir = notes_dir
        self.root = root

    # -- writing --------------------------------------------------------

    def save_source(
        self,
        *,
        url: str,
        title: str,
        body: str,
        suggested_by: str | None = None,
        tags: list[str] | None = None,
    ) -> str | None:
        """Persist a fetched source. Returns its note id, or None on failure.

        Exact-URL duplicates are skipped (the URL-keyed filename already
        exists); near-duplicate bodies (Jaccard > 0.85 vs an existing
        note) are skipped too, reusing hyperresearch's MinHash shingling.
        """
        if not url or not body:
            return None
        stem = _url_slug(url)
        path = self._notes_dir / f"{stem}.md"
        if path.exists():
            return stem  # exact-URL dedup

        body = body[:_MAX_BODY_CHARS]
        if self._is_near_duplicate(body):
            logger.debug("vault: near-duplicate body for %s — skipping", url)
            return None

        try:
            from hyperresearch.core.frontmatter import render_note
            from hyperresearch.models.note import NoteMeta, NoteType

            now = datetime.now(timezone.utc)
            meta = NoteMeta(
                title=(title or url)[:200],
                id=stem,
                tags=list(tags or ["deep_research"]),
                created=now,
                updated=now,
                source=url,
                source_domain=_domain(url),
                fetched_at=now,
                fetch_provider="deep_research",
                type=NoteType.NOTE,
                parent=suggested_by or None,
            )
            path.write_text(render_note(meta, body), encoding="utf-8")
            self._vault.auto_sync()
            return stem
        except Exception as exc:  # noqa: BLE001 — vault writes never fatal
            logger.warning("vault: save_source failed for %s: %s", url, exc)
            return None

    def _is_near_duplicate(self, body: str) -> bool:
        try:
            from hyperresearch.core.frontmatter import parse_frontmatter
            from hyperresearch.core.similarity import jaccard, shingle

            new_sh = shingle(body[:20_000])
            if not new_sh:
                return False
            for note_path in self._notes_dir.glob("*.md"):
                raw = note_path.read_text(encoding="utf-8")
                try:
                    _meta, existing_body = parse_frontmatter(raw)
                except Exception:  # noqa: BLE001
                    existing_body = raw
                if jaccard(new_sh, shingle(existing_body[:20_000])) > _NEAR_DUP_JACCARD:
                    return True
        except Exception:  # noqa: BLE001 — dedup is best-effort
            return False
        return False

    # -- reading --------------------------------------------------------

    def get_source(self, url_or_id: str) -> str | None:
        """Return the stored body for a URL (or note id), sans frontmatter."""
        stem = url_or_id
        if "://" in url_or_id or "." in url_or_id.split("/")[0]:
            stem = _url_slug(url_or_id)
        path = self._notes_dir / f"{stem}.md"
        if not path.exists():
            # fall back: maybe caller passed a raw note id
            path = self._notes_dir / f"{url_or_id}.md"
            if not path.exists():
                return None
        try:
            from hyperresearch.core.frontmatter import parse_frontmatter

            _meta, body = parse_frontmatter(
                path.read_text(encoding="utf-8"),
            )
            return body
        except Exception:  # noqa: BLE001
            return path.read_text(encoding="utf-8")

    def search(self, query: str, k: int = 5) -> list[dict[str, Any]]:
        """FTS5 search over stored sources → [{id, title, snippet, source}]."""
        try:
            from hyperresearch.search.fts import search_fts

            # Markdown is truth: re-index this connection's view of the notes
            # dir so a read handle sees sources written by another handle.
            try:
                self._vault.auto_sync()
            except Exception:  # noqa: BLE001
                pass
            rows = search_fts(self._vault._conn, query, limit=k)
            out: list[dict[str, Any]] = []
            for r in rows:
                out.append({
                    "id": r.get("id", ""),
                    "title": r.get("title", ""),
                    "snippet": r.get("snippet", ""),
                })
            return out
        except Exception as exc:  # noqa: BLE001
            logger.debug("vault: search failed: %s", exc)
            return []

    def source_count(self) -> int:
        return len(list(self._notes_dir.glob("*.md")))

    def close(self) -> None:
        try:
            self._vault.close()
        except Exception:  # noqa: BLE001
            pass


def open_vault(out_dir: str | Path, *, enabled: bool = True) -> VaultHandle | None:
    """Open (or initialize) a vault rooted at ``<out_dir>/vault/``.

    Returns None when disabled or when hyperresearch is unavailable — the
    caller then simply skips all vault interactions.
    """
    if not enabled or not _hyperresearch_available():
        if enabled and not _hyperresearch_available():
            logger.info(
                "vault requested but hyperresearch not installed — "
                "install with `pip install agent-harness[vault]`; "
                "continuing without a vault.",
            )
        return None
    try:
        from hyperresearch.core.vault import Vault

        root = Path(out_dir) / "vault"
        root.mkdir(parents=True, exist_ok=True)
        if (root / ".hyperresearch").exists():
            vault = Vault.discover(root)
        else:
            vault = Vault.init(root)
        notes_dir = root / "research" / "notes"
        notes_dir.mkdir(parents=True, exist_ok=True)
        return VaultHandle(vault, notes_dir, root)
    except Exception as exc:  # noqa: BLE001 — never break the pipeline
        logger.warning("vault: open_vault failed (%s) — continuing without", exc)
        return None


# ---------------------------------------------------------------------------
# Vault directory resolution for the local vault_search / vault_get tools.
#
# The vault dir is threaded to sub-agent tool calls through the agent
# loop's ExecutionScope metadata (``scope_metadata={"vault_dir": ...}``).
# ---------------------------------------------------------------------------

class VaultWriterObserver:
    """Loop observer that mirrors ``web_fetch`` results into the vault.

    Reuses the agent loop's ``on_tool_result`` hook: each successful
    single-URL ``web_fetch`` becomes a stored source note (so
    fact-checkers can re-read it via ``vault_get`` instead of
    re-fetching). Multi-URL fetches are skipped — their numbered blob
    can't be cleanly attributed to one source.
    """

    critical: bool = False

    def __init__(self, vault: VaultHandle, suggested_by: str | None = None) -> None:
        self._vault = vault
        self._suggested_by = suggested_by

    async def on_loop_start(self, config: Any) -> None:  # noqa: ANN401
        pass

    async def on_llm_delta(self, ctx: Any) -> None:  # noqa: ANN401
        return None

    async def on_llm_response(self, ctx: Any) -> None:  # noqa: ANN401
        return None

    async def on_tool_call(self, ctx: Any, tool_call: dict) -> None:  # noqa: ANN401
        return None

    async def on_tool_result(self, ctx: Any, result: Any) -> None:  # noqa: ANN401
        try:
            if getattr(result, "name", "") != "web_fetch":
                return None
            if getattr(result, "is_error", False):
                return None
            args = getattr(result, "args", {}) or {}
            url = args.get("url")
            if isinstance(url, list):
                url = url[0] if len(url) == 1 else None
            body = getattr(result, "result", "") or ""
            if url and body and not body.startswith("[ERROR]"):
                self._vault.save_source(
                    url=str(url), title=str(url), body=body,
                    suggested_by=self._suggested_by,
                )
        except Exception:  # noqa: BLE001 — never disturb the loop
            pass
        return None

    async def on_turn_end(self, ctx: Any) -> None:  # noqa: ANN401
        return None

    async def on_loop_end(self, result: Any) -> None:  # noqa: ANN401
        pass


def vault_dir_from_scope() -> str | None:
    try:
        from agent_harness.core.execution_context import (
            get_current_execution_scope,
        )

        scope = get_current_execution_scope()
        if scope is not None:
            return (scope.metadata or {}).get("vault_dir")
    except Exception:  # noqa: BLE001
        return None
    return None


__all__ = [
    "VaultHandle",
    "VaultWriterObserver",
    "open_vault",
    "vault_dir_from_scope",
]
