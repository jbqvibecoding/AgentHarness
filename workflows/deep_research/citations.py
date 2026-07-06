"""Global citation numbering across parallel sub-agents.

Adapted from Onyx (``backend/onyx/chat/citation_utils.py``:
``collapse_citations`` / ``extract_citation_order_from_text``): each
researcher numbers its sources locally starting at 1; the fan-out node
merges them into one global mapping, deduplicating by normalized URL so
the same source cited by two researchers keeps a single number.

The citation regex mirrors Onyx's deliberately linear pattern
(``\\d+(?:, ?\\d+)*`` inside one bracket pair, unicode variants included)
— written to avoid catastrophic backtracking on adversarial text.
"""

from __future__ import annotations

import re

# {citation_num(str): {"url": ..., "title": ...}} — string keys so the
# mapping survives JSON round-trips through state.json unchanged.
CitationMapping = dict[str, dict[str, str]]

CITATION_PATTERN = re.compile(
    r"([\[【［]{2}\d+[\]】］]{2})|([\[【［]\d+(?:, ?\d+)*[\]】］])"
)


def normalize_url(url: str) -> str:
    """Canonicalize a URL for dedup: strip scheme, trailing slash, fragment."""
    url = (url or "").strip()
    url = re.sub(r"^https?://", "", url, flags=re.IGNORECASE)
    url = url.split("#", 1)[0]
    return url.rstrip("/").lower()


def merge_sources(
    existing: CitationMapping,
    sources: list[dict[str, str]],
) -> tuple[CitationMapping, dict[str, int]]:
    """Merge ``sources`` (``[{url, title}]``) into the global mapping.

    Returns ``(updated_mapping, url_to_num)`` where ``url_to_num`` maps
    each normalized source URL to its global citation number.
    """
    mapping: CitationMapping = dict(existing)
    url_to_num: dict[str, int] = {
        normalize_url(doc.get("url", "")): int(num)
        for num, doc in mapping.items()
        if doc.get("url")
    }
    next_num = max((int(n) for n in mapping), default=0) + 1

    for src in sources:
        url = src.get("url") or ""
        if not url:
            continue
        key = normalize_url(url)
        if key in url_to_num:
            continue
        mapping[str(next_num)] = {
            "url": url,
            "title": (src.get("title") or url)[:300],
        }
        url_to_num[key] = next_num
        next_num += 1
    return mapping, url_to_num


def citation_listing(mapping: CitationMapping) -> str:
    """Render ``[n] Title: URL`` lines in numeric order."""
    lines = []
    for num in sorted(mapping, key=int):
        doc = mapping[num]
        lines.append(f"[{num}] {doc.get('title', '')}: {doc.get('url', '')}")
    return "\n".join(lines)


def extract_cited_numbers(text: str) -> list[int]:
    """Return citation numbers appearing in ``text`` in first-use order."""
    order: list[int] = []
    seen: set[int] = set()
    for m in CITATION_PATTERN.finditer(text or ""):
        for num_str in re.findall(r"\d+", m.group()):
            num = int(num_str)
            if num not in seen:
                seen.add(num)
                order.append(num)
    return order


__all__ = [
    "CitationMapping",
    "CITATION_PATTERN",
    "citation_listing",
    "extract_cited_numbers",
    "merge_sources",
    "normalize_url",
]
