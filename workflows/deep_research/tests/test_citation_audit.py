"""Citation-integrity tests for the ``citation_audit`` node and its adapters.

These are pure-function tests: no API key, no network. They pin the
contract between this pipeline's evidence shapes and the ported
``workflows/_shared`` citation machinery, and the repair-once-then-disclose
policy the node implements.
"""

from __future__ import annotations

import pytest

from workflows._shared.cited_report_finalizer import (
    parse_trailing_references,
    validate_citation_body,
    validate_numeric_grounding,
)
from workflows._shared.citation_contract import (
    compose_citation_contract,
    finalize_report_with_canonical_references,
)
from workflows._shared.report_clean import clean_report_v3
from workflows.deep_research.references import (
    cards_for_url_repair,
    mapping_from_references,
    references_from_mapping,
    renumbered_references,
    snippet_lookup_for,
)

MAPPING = {
    "1": {"url": "https://a.example/report", "title": "A Report"},
    "2": {"url": "https://b.example/filing", "title": "B Filing"},
    "3": {"url": "https://c.example/blog", "title": "C Blog"},
}

CARDS = [
    {
        "card_id": "E1",
        "claim": "Revenue reached $5 billion in 2024.",
        "quote": "Full-year revenue reached $5 billion.",
        "sources": [{"url": "https://a.example/report", "title": "A Report"}],
    },
    {
        "card_id": "E2",
        "claim": "Operating costs were $1.2 billion.",
        "quote": "Operating costs came to $1.2 billion for the year.",
        "sources": [{"url": "https://b.example/filing"}],
    },
]


# --------------------------------------------------------------------------
# Adapters
# --------------------------------------------------------------------------

def test_references_from_mapping_is_dense_and_ordered():
    refs = references_from_mapping(MAPPING)
    assert [r["url"] for r in refs] == [
        "https://a.example/report",
        "https://b.example/filing",
        "https://c.example/blog",
    ]
    # List position i must equal citation [i+1] — the binding the whole
    # contract depends on.
    assert refs[0]["url"] == MAPPING["1"]["url"]


def test_references_from_mapping_sorts_numerically_not_lexically():
    mapping = {str(n): {"url": f"https://e{n}.example", "title": ""} for n in range(1, 12)}
    refs = references_from_mapping(mapping)
    # Lexical sort would put "10" and "11" before "2".
    assert refs[1]["url"] == "https://e2.example"
    assert refs[9]["url"] == "https://e10.example"


def test_references_from_mapping_skips_url_less_entries():
    refs = references_from_mapping({"1": {"url": "", "title": "ghost"}})
    assert refs == []


def test_mapping_from_references_round_trips():
    refs = references_from_mapping(MAPPING)
    assert mapping_from_references(refs) == MAPPING


def test_cards_for_url_repair_flattens_every_source():
    cards = cards_for_url_repair([
        {"sources": [{"url": "https://x.example"}, {"url": "https://y.example"}]},
        {"sources": []},
        {"no_sources_key": True},
    ])
    assert [c["source"]["url"] for c in cards] == [
        "https://x.example", "https://y.example",
    ]


def test_snippet_lookup_returns_supporting_text_for_the_right_index():
    refs = references_from_mapping(MAPPING)
    lookup = snippet_lookup_for(refs, CARDS)
    assert "$5 billion" in lookup(1)
    assert "$1.2 billion" in lookup(2)
    assert lookup(3) == ""      # no card cites the blog
    assert lookup(99) == ""     # out of range is empty, not an error


# --------------------------------------------------------------------------
# Deterministic checks
# --------------------------------------------------------------------------

def test_validate_citation_body_flags_orphan_index():
    issues = validate_citation_body(
        "Claim one [1]. Claim two [7].", max_ref=3, valid_indices=[1, 2, 3],
    )
    assert issues and "7" in issues[0]


def test_validate_citation_body_accepts_in_whitelist_indices():
    assert validate_citation_body(
        "Claim [1] and claim [3].", max_ref=3, valid_indices=[1, 2, 3],
    ) == []


def test_validate_citation_body_flags_a_report_with_no_citations():
    issues = validate_citation_body("No markers here.", max_ref=3)
    assert issues == ["no inline [N] citations were emitted"]


def test_numeric_grounding_separates_grounded_from_suspect():
    """A figure the source supports is grounded; an inflated one is suspect.

    Both numbers carry a valid citation, so citation validation alone
    cannot tell them apart — this is the check that does.
    """
    refs = references_from_mapping(MAPPING)
    lookup = snippet_lookup_for(refs, CARDS)
    report = validate_numeric_grounding(
        "Revenue reached $5 billion [1]. Costs were $7.4 billion [2].",
        snippet_lookup=lookup,
    )
    grounded = {tok for tok, _ in report.grounded + report.grounded_derived}
    suspect = {tok for tok, _ in report.suspect}
    assert "$5 billion" in grounded
    assert "$7.4 billion" in suspect
    assert 0.0 < report.suspect_ratio <= 1.0


def test_numeric_grounding_matches_across_notations():
    refs = references_from_mapping(MAPPING)
    lookup = snippet_lookup_for(refs, [{
        "card_id": "E9",
        "claim": "",
        "quote": "Revenue was $5,000 million for the year.",
        "sources": [{"url": "https://a.example/report"}],
    }])
    report = validate_numeric_grounding(
        "Revenue reached $5 billion [1].", snippet_lookup=lookup,
    )
    assert report.suspect == [], "$5 billion ↔ $5,000 million must match"


def test_numeric_grounding_without_lookup_does_not_invent_suspects():
    body = "Revenue reached $5 billion [1]."
    report = validate_numeric_grounding(body, snippet_lookup=None)
    assert report.suspect == []


# --------------------------------------------------------------------------
# Contract + finalization
# --------------------------------------------------------------------------

def test_contract_is_empty_without_references():
    # An empty contract must not be appended: it would tell the model to
    # "use one of these indices" with no indices.
    assert compose_citation_contract([]) == ""


def test_contract_lists_every_reference():
    block = compose_citation_contract(references_from_mapping(MAPPING))
    assert "[1]" in block and "https://a.example/report" in block
    assert "https://c.example/blog" in block


def test_finalize_renumbers_by_first_appearance():
    refs = references_from_mapping(MAPPING)
    body = "Second source first [2]. Then the first one [1]."
    once = finalize_report_with_canonical_references(body, references=refs)
    parsed = parse_trailing_references(once)
    # [2] was cited first, so it becomes [1] and the References block agrees.
    assert parsed["1"]["url"] == "https://b.example/filing"
    assert parsed["2"]["url"] == "https://a.example/report"
    # The uncited third source is dropped rather than listed.
    assert "3" not in parsed


def test_finalize_is_idempotent_against_the_shipped_reference_list():
    """Re-running the finalizer must not shuffle numbering a second time.

    It is idempotent given the references the body is *currently* numbered
    against — which after one pass is the renumbered list, not the original
    whitelist. This is exactly why the node tracks the shipped list.
    """
    refs = references_from_mapping(MAPPING)
    body = "Second source first [2]. Then the first one [1]."
    once = finalize_report_with_canonical_references(body, references=refs)
    shipped = renumbered_references(body, refs)
    twice = finalize_report_with_canonical_references(once, references=shipped)
    assert once == twice


def test_renumbered_references_predicts_the_shipped_order():
    refs = references_from_mapping(MAPPING)
    body = "Blog first [3], then filing [2], then blog again [3]."
    shipped = renumbered_references(body, refs)
    assert [r["url"] for r in shipped] == [
        "https://c.example/blog", "https://b.example/filing",
    ]


def test_renumbered_references_ignores_out_of_range_markers():
    refs = references_from_mapping(MAPPING)
    assert renumbered_references("Only [99] here.", refs) == []


def test_finalize_leaves_an_uncited_report_alone():
    refs = references_from_mapping(MAPPING)
    body = "A report with no citation markers at all."
    assert finalize_report_with_canonical_references(body, references=refs) == body


def test_short_report_footer_is_not_parseable_so_prediction_is_required():
    """Pins the reason ``renumbered_references`` exists.

    ``parse_trailing_references`` ignores a References block starting in the
    first 30% of the text, so on a short report it returns nothing at all —
    reading the shipped numbering back out of the report would silently
    fall back to the wrong (pre-renumber) mapping.
    """
    refs = references_from_mapping(MAPPING)
    body = "Only the blog [3]."
    final = finalize_report_with_canonical_references(body, references=refs)
    assert "https://c.example/blog" in final
    assert parse_trailing_references(final) == {}
    shipped = mapping_from_references(renumbered_references(body, refs))
    assert shipped == {"1": {"url": "https://c.example/blog", "title": "C Blog"}}


# --------------------------------------------------------------------------
# Report cleaning
# --------------------------------------------------------------------------

def test_clean_report_v3_scrubs_internal_ids_outside_code():
    body = "The claim holds [E1] per the filing."
    cleaned, stats = clean_report_v3(body, internal_node_ids={"E1"})
    assert "E1" not in cleaned
    assert stats["node_ids_scrubbed"] >= 1


def test_clean_report_v3_leaves_code_fences_untouched():
    body = "Prose here.\n\n```python\narr[0] = E1\n```\n"
    cleaned, _stats = clean_report_v3(body, internal_node_ids={"E1"})
    assert "arr[0] = E1" in cleaned, "code fences must never be rewritten"


def test_clean_report_v3_without_ids_is_a_noop_scrub():
    body = "The claim holds [E1]."
    cleaned, stats = clean_report_v3(body, internal_node_ids=set())
    assert cleaned.strip() == body.strip()
    assert stats["node_ids_scrubbed"] == 0


# --------------------------------------------------------------------------
# Node behaviour
# --------------------------------------------------------------------------

class _Ctx:
    node_id = "citation_audit"
    role_id = "dr_verifier"
    task_id = "t-test"


def _state(**over):
    state = {
        "task_id": "t-test",
        "original_question": "How large is the market?",
        "metadata": {"deep_research": {"citation_repair_rounds": 0}},
        "citation_mapping": dict(MAPPING),
        "evidence_cards": list(CARDS),
        "fact_check_results": [],
        "report": "Revenue reached $5 billion [1]. Costs were $1.2 billion [2].",
    }
    state.update(over)
    return state


@pytest.mark.asyncio
async def test_node_audits_and_appends_canonical_references():
    from workflows.deep_research.nodes.citation_audit import citation_audit_node

    out = await citation_audit_node(_state(), _Ctx())
    assert out["report"], "terminal output must stay non-empty"
    assert "## References" in out["report"] or "参考" in out["report"]
    assert out["numeric_grounding"]["total"] >= 2
    assert out["citation_audit"]["reference_count"] == 3
    # The mapping must describe the report that shipped, not the pre-audit one.
    assert set(out["citation_mapping"]) == {"1", "2"}


@pytest.mark.asyncio
async def test_node_discloses_orphan_citations_in_the_appendix():
    from workflows.deep_research.nodes.citation_audit import citation_audit_node

    state = _state(report="A bold claim [9].")
    out = await citation_audit_node(state, _Ctx())
    assert "Verification appendix" in out["report"]
    assert "9" in out["report"]
    assert out["citation_audit"]["issues"]


@pytest.mark.asyncio
async def test_node_passes_the_report_through_when_disabled():
    from workflows.deep_research.nodes.citation_audit import citation_audit_node

    state = _state(metadata={"deep_research": {"citation_contract": False}})
    out = await citation_audit_node(state, _Ctx())
    assert "report" not in out


@pytest.mark.asyncio
async def test_node_never_loses_the_report_when_a_step_explodes(monkeypatch):
    from workflows.deep_research.nodes import citation_audit as mod

    def boom(*_a, **_kw):
        raise RuntimeError("audit exploded")

    monkeypatch.setattr(mod, "repair_citation_urls", boom)
    out = await mod.citation_audit_node(_state(), _Ctx())
    # Node returns no report key, so the incoming one survives the merge.
    assert "report" not in out
    assert any("citation_audit" in e for e in out["errors"])


# --------------------------------------------------------------------------
# Writer-side contract injection
# --------------------------------------------------------------------------

def test_writer_prompt_swaps_its_own_references_bullet_for_the_contract():
    from workflows.deep_research.nodes.draft import _writer_system

    system = _writer_system({
        "metadata": {},
        "citation_mapping": {"1": {"url": "https://a.example", "title": "A"}},
    })
    assert "Deterministic Citation Contract" in system
    # Both instructions at once is what produces two reference lists.
    assert 'End with a "## References" section' not in system


def test_writer_prompt_unchanged_when_the_contract_is_off():
    from workflows.deep_research.nodes.draft import _writer_system

    system = _writer_system({
        "metadata": {"deep_research": {"citation_contract": False}},
        "citation_mapping": {"1": {"url": "https://a.example"}},
    })
    assert "Deterministic Citation Contract" not in system
    assert 'End with a "## References" section' in system


def test_writer_prompt_keeps_its_own_bullet_when_there_are_no_sources():
    """An empty whitelist must not activate the contract.

    Telling a model to "use one of these indices" with no indices makes it
    either refuse to cite or invent numbers.
    """
    from workflows.deep_research.nodes.draft import _writer_system

    system = _writer_system({"metadata": {}, "citation_mapping": {}})
    assert "Deterministic Citation Contract" not in system
    assert 'End with a "## References" section' in system
