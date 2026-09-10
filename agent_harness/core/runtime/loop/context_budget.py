"""Token estimation for context-window accounting.

Uses ``tiktoken cl100k_base`` when available and falls back to a CJK-aware
heuristic. ``estimate_tokens`` is the only export consumed in OSS — the
historical budget-allocation + truncation helpers (``MODEL_CONTEXT_BUDGETS``,
``RESEARCH_PROMPT_ALLOCATION``, ``allocate_prompt_budget``,
``truncate_to_budget``, ``compact_evidence`` etc.) were re-exported from
``core/runtime/loop/__init__.py`` but had no real callers.

``truncate_text_to_tokens`` is ported from FrontierAgent
(``frontier_agent/core/runtime/loop/context_budget.py``, Apache-2.0); the
finalization-recovery path needs it to replay a damaged history into a
finalize prompt without blowing the context window.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from typing import Literal

logger = logging.getLogger(__name__)


_tokenizer = None
_tokenizer_loaded = False

# Broad CJK regex: Unified Ideographs, Ext-A, radicals, strokes,
# Hiragana, Katakana, CJK compatibility, fullwidth forms.
_CJK_RE = re.compile(
    r"[⺀-⻿　-〿぀-ヿ㐀-䶿"
    r"一-鿿豈-﫿＀-￯]"
)


def _get_tokenizer():
    """Lazily load the tiktoken cl100k_base encoder."""
    global _tokenizer, _tokenizer_loaded  # noqa: PLW0603
    if not _tokenizer_loaded:
        _tokenizer_loaded = True
        try:
            import tiktoken
            _tokenizer = tiktoken.get_encoding("cl100k_base")
        except (ImportError, Exception):
            logger.debug(
                "tiktoken unavailable, using heuristic token estimation"
            )
    return _tokenizer


def estimate_tokens(text: str) -> int:
    """Estimate the token count for a plain text string.

    Uses tiktoken cl100k_base when available; falls back to a CJK-aware
    heuristic (CJK character ≈ 1 token, Latin text ≈ chars / 4).
    """
    if not text:
        return 0

    enc = _get_tokenizer()
    if enc is not None:
        return len(enc.encode(text))

    cjk_count = len(_CJK_RE.findall(text))
    other_count = len(text) - cjk_count
    return cjk_count + (other_count // 4)


def truncate_text_to_tokens(
    text: str,
    max_tokens: int,
    *,
    marker: str = "\n[... older context truncated to fit token budget ...]",
    estimator: Callable[[str], int] = estimate_tokens,
    keep: Literal["head", "tail"] = "head",
) -> str:
    """Keep the largest text prefix — or suffix — that fits a token budget.

    ``estimator`` and ``marker`` are injectable so specialized callers can
    preserve their existing tokenizer and user-facing truncation language.

    ``keep="head"`` (default) drops the newest text and appends the marker —
    right for summaries and tool output, where the opening lines carry the
    identity of the content. ``keep="tail"`` drops the oldest text and
    prepends the marker — right for reasoning traces, where the conclusion
    and the tool-use intent sit at the end.

    Note that ``estimator`` need not be monotonic in the slice length, so the
    binary search returns a near-maximal slice rather than a provably maximal
    one. The budget itself is always respected.
    """
    if keep not in ("head", "tail"):
        raise ValueError(f"keep must be 'head' or 'tail', got {keep!r}")
    if max_tokens <= 0:
        return ""
    if estimator(text) <= max_tokens:
        return text
    # A marker wider than the whole budget would leave room for a single
    # character of real text and still overshoot; drop it instead.
    effective_marker = "" if estimator(marker) >= max_tokens else marker
    target = max(1, max_tokens - estimator(effective_marker))
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        chunk = text[:mid] if keep == "head" else text[-mid:]
        if estimator(chunk) <= target:
            lo = mid
        else:
            hi = mid - 1
    if keep == "head":
        return text[:lo] + effective_marker
    return effective_marker + text[-lo:] if lo else effective_marker
