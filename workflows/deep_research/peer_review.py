"""Anonymized peer review among council members — ballots and aggregation.

Design inspired by karpathy/llm-council's three-stage council (first
opinions → anonymized peer ranking → chairman synthesis). The peer-review
stage is the piece our council lacked: instead of one analyst model
judging every answer, each member model ranks the others' answers blind,
turning "who answered best" into an aggregated vote.

This is an independent implementation that fixes several weaknesses of
that reference design:

* **Per-evaluator label shuffling** — each evaluator sees its own random
  label ordering, so position bias does not stack across ballots.
* **Self-vote exclusion** — an evaluator never ranks its own answer, so
  self-preference cannot inflate a member's score.
* **Structured ballots** — evaluators return JSON (parsed with the
  pipeline's tolerant ``extract_json_block``) instead of a prose tail
  scraped by regex, whose fallback could mistake mention-order for rank.
* **Normalized Borda scoring** — a rank on a short ballot is worth the
  same as on a long one, so a member ranked by fewer evaluators cannot
  top the table by accident; ties break deterministically by name.

All functions here are pure and side-effect free (the node calls the
models); ``rng`` is injectable so tests are deterministic.
"""

from __future__ import annotations

import random
from typing import Any

from workflows.deep_research.prompts import extract_json_block

#: Labels shown to evaluators — anonymizes authorship of each answer.
_LABELS = [chr(ord("A") + i) for i in range(26)]

MIN_MEMBERS_FOR_REVIEW = 2


def build_ballots(
    member_results: list[dict[str, Any]],
    *,
    exclude_self: bool = True,
    rng: random.Random | None = None,
) -> list[dict[str, Any]]:
    """Build one anonymized ballot per successful member.

    Returns ``[{evaluator, evaluator_slug, label_to_member, entries}]``
    where ``entries`` is ``[{label, content}]`` in the (shuffled) order the
    evaluator will see, and ``label_to_member`` maps each label back to the
    member name for de-anonymization after parsing.

    With ``exclude_self`` (default) an evaluator's ballot omits its own
    answer, so it cannot vote for itself. Members with fewer than two
    peers to rank get no ballot.
    """
    rng = rng or random.Random()
    ok = [m for m in member_results if m.get("status") == "ok" and m.get("report")]
    ballots: list[dict[str, Any]] = []

    for evaluator in ok:
        peers = [
            m for m in ok
            if not (exclude_self and m["name"] == evaluator["name"])
        ]
        if len(peers) < MIN_MEMBERS_FOR_REVIEW:
            continue
        shuffled = list(peers)
        rng.shuffle(shuffled)  # independent label ordering per evaluator
        label_to_member = {
            _LABELS[i]: m["name"] for i, m in enumerate(shuffled)
        }
        ballots.append({
            "evaluator": evaluator["name"],
            "evaluator_slug": evaluator.get("slug", ""),
            "label_to_member": label_to_member,
            "entries": [
                {"label": _LABELS[i], "content": m.get("report", "")}
                for i, m in enumerate(shuffled)
            ],
        })
    return ballots


def parse_ballot(raw: str, label_to_member: dict[str, str]) -> dict[str, Any]:
    """Parse an evaluator's JSON reply into a de-anonymized ranking.

    Returns ``{ranking: [member_name, ...], critiques: [...],
    top_reason: str, valid: bool}``. Unknown labels are dropped and
    duplicates ignored; a ballot with fewer than two resolvable entries is
    marked invalid (it contributes nothing to the aggregate).
    """
    data = extract_json_block(raw)
    if not isinstance(data, dict):
        return {"ranking": [], "critiques": [], "top_reason": "", "valid": False}

    ranking: list[str] = []
    seen: set[str] = set()
    for label in data.get("ranking") or []:
        name = label_to_member.get(str(label).strip().upper()[:1])
        if name and name not in seen:
            seen.add(name)
            ranking.append(name)

    critiques: list[dict[str, str]] = []
    for item in data.get("critiques") or []:
        if not isinstance(item, dict):
            continue
        name = label_to_member.get(str(item.get("label", "")).strip().upper()[:1])
        if not name:
            continue
        critiques.append({
            "model": name,
            "strengths": str(item.get("strengths", ""))[:400],
            "weaknesses": str(item.get("weaknesses", ""))[:400],
        })

    return {
        "ranking": ranking,
        "critiques": critiques,
        "top_reason": str(data.get("top_reason", ""))[:400],
        "valid": len(ranking) >= MIN_MEMBERS_FOR_REVIEW,
    }


def aggregate_rankings(
    parsed_ballots: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Aggregate ballots into a peer ranking with normalized Borda scores.

    On a ballot of ``k`` ranked members, position ``p`` (1-indexed) scores
    ``(k - p) / (k - 1)`` — 1.0 for first, 0.0 for last, regardless of
    ballot length, so short and long ballots weigh the same. A member's
    ``peer_score`` is the mean of its normalized scores.

    Returns ``[{model, peer_score, average_rank, ballots_counted}]``
    sorted by descending score, ties broken by model name so the order is
    deterministic.
    """
    scores: dict[str, list[float]] = {}
    positions: dict[str, list[int]] = {}

    for ballot in parsed_ballots:
        if not ballot.get("valid"):
            continue
        ranking = ballot.get("ranking") or []
        k = len(ranking)
        if k < MIN_MEMBERS_FOR_REVIEW:
            continue
        for pos, model in enumerate(ranking, start=1):
            scores.setdefault(model, []).append((k - pos) / (k - 1))
            positions.setdefault(model, []).append(pos)

    out: list[dict[str, Any]] = []
    for model, vals in scores.items():
        pos = positions[model]
        out.append({
            "model": model,
            "peer_score": round(sum(vals) / len(vals), 3),
            "average_rank": round(sum(pos) / len(pos), 2),
            "ballots_counted": len(vals),
        })
    out.sort(key=lambda r: (-r["peer_score"], r["model"]))
    return out


def peer_notes_by_model(
    parsed_ballots: list[dict[str, Any]],
) -> dict[str, list[str]]:
    """Collect what peers said about each model (for the ranking table)."""
    notes: dict[str, list[str]] = {}
    for ballot in parsed_ballots:
        for crit in ballot.get("critiques") or []:
            model = crit.get("model")
            weakness = (crit.get("weaknesses") or "").strip()
            strength = (crit.get("strengths") or "").strip()
            text = weakness or strength
            if model and text:
                notes.setdefault(model, []).append(text)
    return notes


__all__ = [
    "MIN_MEMBERS_FOR_REVIEW",
    "aggregate_rankings",
    "build_ballots",
    "parse_ballot",
    "peer_notes_by_model",
]
