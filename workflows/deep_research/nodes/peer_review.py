"""``peer_review`` node — anonymized cross-ranking among council members.

Sits between the members answering and the analyst/synthesizer: every
member re-reads the others' answers with authorship hidden, critiques
them, and ranks them. Ballots are aggregated into a peer ranking that
feeds both the comparison tables and the synthesizer — so "which answer
is best" comes from an anonymous vote of the whole council rather than
one analyst model's opinion.

Off by default (``peer_review`` config knob): when disabled the node
returns immediately with zero LLM calls, so the default council path
costs exactly what it did before. Shared by ``model_council`` (ranking
direct answers) and ``deep_council_research`` (ranking research-report
excerpts).

Fail-open throughout: an evaluator that errors or returns an unparseable
ballot is dropped with a recorded error; the remaining ballots still
produce a ranking, and no failure here can block the final report.
"""

from __future__ import annotations

import logging
import random
from typing import Any

from agent_harness.models.node_context import NodeContext

from workflows.deep_research.config import get_cfg
from workflows.deep_research.council import (
    chat_with_retries,
    get_llm_for_member,
    load_council_members,
)
from workflows.deep_research.council_prompts import (
    COUNCIL_PEER_REVIEW_SYSTEM,
    build_peer_review_prompt,
)
from workflows.deep_research.nodes.common import profile_name, today
from workflows.deep_research.peer_review import (
    MIN_MEMBERS_FOR_REVIEW,
    aggregate_rankings,
    build_ballots,
    parse_ballot,
)
from workflows.deep_research.profile import load_profile
from workflows.deep_research.prompts import with_date
from workflows.deep_research.subagent import gather_with_limit

logger = logging.getLogger(__name__)

_SKIPPED: dict[str, Any] = {
    "peer_reviews": [],
    "peer_ranking": [],
    "current_phase": "peer_review",
}


async def peer_review_node(
    state: dict[str, Any], ctx: NodeContext,
) -> dict[str, Any]:
    if not bool(get_cfg(state, "peer_review")):
        return dict(_SKIPPED)

    member_results = state.get("member_results") or []
    ok = [m for m in member_results if m.get("status") == "ok" and m.get("report")]
    # Ranking needs an evaluator plus at least two peers to compare.
    if len(ok) <= MIN_MEMBERS_FOR_REVIEW:
        msg = (
            f"peer_review skipped: {len(ok)} member(s) succeeded — at least "
            f"{MIN_MEMBERS_FOR_REVIEW + 1} are needed for blind cross-ranking"
        )
        logger.info(msg)
        return {**_SKIPPED, "errors": [msg]}

    mode = (
        "deep"
        if (state.get("metadata") or {}).get("council_mode") == "deep"
        else "light"
    )
    excerpt_chars = int(get_cfg(state, "peer_review_excerpt_chars"))
    profile = load_profile(profile_name(state))
    members_by_name = {m["name"]: m for m in load_council_members(profile)}
    system = with_date(COUNCIL_PEER_REVIEW_SYSTEM, today())

    ballots = build_ballots(ok, exclude_self=True, rng=random.Random())
    ballots = [b for b in ballots if b["evaluator"] in members_by_name]
    if not ballots:
        return {**_SKIPPED, "errors": ["peer_review: no ballots could be built"]}

    async def review(ballot: dict[str, Any]) -> str:
        return await chat_with_retries(
            get_llm_for_member(members_by_name[ballot["evaluator"]]),
            system_prompt=system,
            user_prompt=build_peer_review_prompt(
                state.get("original_question", ""),
                ballot["entries"], mode, excerpt_chars,
            ),
            timeout_s=600.0,
        )

    logger.info(
        "peer_review (task=%s, mode=%s): %d evaluators ranking %d peers each",
        ctx.task_id, mode, len(ballots), len(ballots[0]["entries"]),
    )
    raw_results = await gather_with_limit(
        [review(b) for b in ballots],
        limit=int(get_cfg(state, "peer_review_parallel")),
    )

    peer_reviews: list[dict[str, Any]] = []
    parsed_ballots: list[dict[str, Any]] = []
    errors: list[str] = []
    for ballot, raw in zip(ballots, raw_results):
        if isinstance(raw, BaseException):
            errors.append(
                f"peer_review: evaluator '{ballot['evaluator']}' failed: "
                f"{raw!r}"[:300],
            )
            continue
        parsed = parse_ballot(raw, ballot["label_to_member"])
        if not parsed["valid"]:
            errors.append(
                f"peer_review: evaluator '{ballot['evaluator']}' returned an "
                f"unusable ballot — discarded",
            )
            continue
        parsed_ballots.append(parsed)
        peer_reviews.append({
            "evaluator": ballot["evaluator"],
            "ranking": parsed["ranking"],
            "critiques": parsed["critiques"],
            "top_reason": parsed["top_reason"],
        })

    ranking = aggregate_rankings(parsed_ballots)
    logger.info(
        "peer_review (task=%s): %d/%d ballots counted; winner=%s",
        ctx.task_id, len(parsed_ballots), len(ballots),
        ranking[0]["model"] if ranking else "n/a",
    )
    return {
        "peer_reviews": peer_reviews,
        "peer_ranking": ranking,
        "errors": errors,
        "current_phase": "peer_review",
    }


__all__ = ["peer_review_node"]
