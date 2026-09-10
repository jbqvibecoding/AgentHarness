"""Adapters between this pipeline's evidence shapes and ``workflows/_shared``.

The ported citation machinery (``workflows/_shared/citation_contract.py``,
``cited_report_finalizer.py``, ``citation_url_repair.py``) speaks two shapes
that ``deep_research`` does not use natively:

* a *references whitelist* — ``[{"url", "title"}]`` where list position
  ``i`` is citation ``[i + 1]``;
* an *evidence card* with a single ``{"source": {"url"}}`` mapping.

``deep_research`` instead carries ``citation_mapping``
(``{num_str: {"url", "title"}}``, built by :mod:`workflows.deep_research.citations`
during research fan-out and already deduplicated by normalized URL) and
evidence cards with a ``sources`` *list*. These adapters translate, so the
shared modules are reused verbatim rather than reimplemented.

Reusing ``citation_mapping`` as the whitelist — rather than rebuilding one
from the cards — is deliberate: researchers already cite against those
numbers, so rebuilding would silently retarget every ``[N]`` the writer was
given.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from workflows._shared.citation import citation_numbers
from workflows.deep_research.citations import CitationMapping, normalize_url

__all__ = [
    "cards_for_url_repair",
    "mapping_from_references",
    "references_from_mapping",
    "renumbered_references",
    "snippet_lookup_for",
]


def references_from_mapping(mapping: CitationMapping) -> list[dict[str, str]]:
    """Order ``citation_mapping`` into the shared whitelist shape.

    ``merge_sources`` numbers sources densely from 1, so ordering by the
    integer key makes list position ``i`` equal citation ``[i + 1]`` — the
    binding ``compose_citation_contract`` and
    ``finalize_report_with_canonical_references`` both assume. Non-numeric
    or URL-less entries are skipped; a gap in the numbering would shift
    every later citation, so it is logged by the caller rather than
    silently absorbed here.
    """
    refs: list[dict[str, str]] = []
    for num in sorted((mapping or {}), key=lambda k: int(k) if str(k).isdigit() else 0):
        doc = mapping.get(num) or {}
        url = (doc.get("url") or "").strip()
        if not url:
            continue
        refs.append({"url": url, "title": (doc.get("title") or "").strip()})
    return refs


def mapping_from_references(
    references: list[dict[str, str]],
) -> CitationMapping:
    """Inverse of :func:`references_from_mapping`.

    Used after the finalizer renumbers by first appearance (which also
    drops uncited sources) so ``citation_mapping`` keeps describing the
    report that actually shipped.
    """
    return {
        str(i): {"url": ref.get("url", ""), "title": ref.get("title", "")}
        for i, ref in enumerate(references, 1)
        if (ref.get("url") or "").strip()
    }


def renumbered_references(
    body: str, references: list[dict[str, str]],
) -> list[dict[str, str]]:
    """Predict the reference list ``finalize_report_with_canonical_references``
    will emit for ``body``.

    That finalizer renumbers inline markers by order of first appearance and
    drops references the body never cites, but returns only the rendered
    report — so the caller has no direct way to learn the shipped numbering.
    Reading it back out of the appended block is unreliable: the footer
    parser deliberately ignores a References section that starts in the first
    30% of the text, which is exactly what a short report produces.

    Applying the same rule here keeps ``citation_mapping`` describing the
    report that shipped, on long and short reports alike. Pass the body
    *before* finalization — the markers it renumbers are the ones still
    numbered against ``references``.
    """
    order: list[int] = []
    for num in citation_numbers(body or ""):
        if 1 <= num <= len(references) and num not in order:
            order.append(num)
    return [references[n - 1] for n in order]


def cards_for_url_repair(cards: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flatten ``{"sources": [...]}`` cards into the one-source-per-card
    shape ``citation_url_repair`` indexes.

    Every source becomes its own entry, so a card citing three pages
    contributes all three spellings to the whitelist.
    """
    out: list[dict[str, Any]] = []
    for card in cards or []:
        if not isinstance(card, dict):
            continue
        for src in card.get("sources") or []:
            url = (src or {}).get("url") if isinstance(src, dict) else None
            if not isinstance(url, str) or not url.strip():
                continue
            out.append({
                "source": {"url": url.strip()},
                "source_type": src.get("source_type") or "search",
            })
    return out


def snippet_lookup_for(
    references: list[dict[str, str]],
    cards: list[dict[str, Any]],
) -> Callable[[int], str]:
    """Build the ``index -> supporting text`` lookup the numeric audit needs.

    ``validate_numeric_grounding`` compares each cited number against the
    text behind its citation index. Our best evidence for reference ``[N]``
    is every card quoting or claiming something from that URL, so the
    snippet is those cards' ``quote`` and ``claim`` text concatenated.

    Without this the audit degrades to "is the number cited at all", which
    catches uncited numbers but not fabricated ones attached to a real
    citation — the class we actually care about.
    """
    by_url: dict[str, list[str]] = {}
    for card in cards or []:
        if not isinstance(card, dict):
            continue
        texts = [
            str(card.get("quote") or "").strip(),
            str(card.get("claim") or "").strip(),
        ]
        text = "\n".join(t for t in texts if t)
        if not text:
            continue
        for src in card.get("sources") or []:
            url = (src or {}).get("url") if isinstance(src, dict) else None
            if not isinstance(url, str) or not url.strip():
                continue
            by_url.setdefault(normalize_url(url), []).append(text)

    index_to_url = {
        i: normalize_url(ref.get("url", ""))
        for i, ref in enumerate(references, 1)
    }

    def lookup(index: int) -> str:
        key = index_to_url.get(index, "")
        if not key:
            return ""
        return "\n".join(by_url.get(key, ()))

    return lookup
