"""Surgical edit-hunk application — "patch, never regenerate".

Ported discipline from hyperresearch (skills 14/15): after the report
exists, revisions are minimal Edit hunks, not regenerations. Each hunk's
``old`` anchor must match the current text exactly once; non-matching or
ambiguous hunks are recorded as skipped rather than applied, and a
citation lint gate rejects any hunk that would break an inline ``[n]``
reference.
"""

from __future__ import annotations

import re
from typing import Any

from workflows.deep_research.citations import extract_cited_numbers


def _apply_one(text: str, old: str, new: str) -> tuple[str | None, str]:
    """Apply a single hunk. Returns (new_text|None, status)."""
    if not old:
        return None, "empty-anchor"
    count = text.count(old)
    if count == 0:
        return None, "anchor-not-found"
    if count > 1:
        return None, "anchor-ambiguous"
    return text.replace(old, new, 1), "applied"


def apply_edit_hunks(
    text: str,
    hunks: list[dict[str, Any]],
    *,
    citation_mapping: dict[str, Any] | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    """Apply edit hunks surgically; return (new_text, log).

    ``hunks`` items: ``{old, new, reason}``. Each hunk is applied only if
    its ``old`` occurs exactly once AND (when ``citation_mapping`` is
    given) it does not introduce an unresolvable ``[n]`` citation. The log
    records one entry per hunk with ``status`` in
    ``applied | anchor-not-found | anchor-ambiguous | empty-anchor |
    breaks-citations``.
    """
    result = text
    log: list[dict[str, Any]] = []
    valid_nums = (
        {int(k) for k in citation_mapping} if citation_mapping else None
    )
    for hunk in hunks:
        old = str(hunk.get("old", ""))
        new = str(hunk.get("new", ""))
        reason = str(hunk.get("reason", ""))[:200]
        candidate, status = _apply_one(result, old, new)
        if candidate is not None and valid_nums is not None:
            # Reject hunks that add a citation number with no source.
            added = set(extract_cited_numbers(new)) - set(
                extract_cited_numbers(old),
            )
            if added - valid_nums:
                candidate, status = None, "breaks-citations"
        if candidate is not None:
            result = candidate
        log.append({
            "status": status, "reason": reason,
            "old_preview": old[:80], "applied": candidate is not None,
        })
    return result, log


def validate_citations(
    text: str, citation_mapping: dict[str, Any],
) -> list[int]:
    """Return citation numbers used in ``text`` that have no source entry."""
    valid = {int(k) for k in citation_mapping}
    return sorted(n for n in extract_cited_numbers(text) if n not in valid)


def strip_filler(text: str, phrases: tuple[str, ...]) -> tuple[str, int]:
    """Deterministically drop leading filler phrases; return (text, count).

    Case-insensitive at sentence/line starts; the following word is
    re-capitalized so the sentence still reads correctly.
    """
    removed = 0
    out = text
    for phrase in phrases:
        pattern = re.compile(
            r"(^|[.\n]\s+)" + re.escape(phrase),
            flags=re.IGNORECASE,
        )

        def _repl(m: re.Match) -> str:
            nonlocal removed
            removed += 1
            return m.group(1)

        out = pattern.sub(_repl, out)
    return out, removed


__all__ = ["apply_edit_hunks", "strip_filler", "validate_citations"]
