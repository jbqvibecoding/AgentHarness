"""Standalone CLI runner for the ``deep_research`` pipeline.

    uv run python -m workflows.deep_research.run \
        --question "..." [--out DIR] [--depth quick|standard|deep] \
        [--profile default] [--max-iterations N] [--max-revisions N] \
        [--max-parallel N] [--metadata-json '{...}'] [--dry-run]

Output contract (consumed by the Hermes ``deep_research`` tool):

* ``<out>/report.md``      — the final report (only on success).
* ``<out>/result.json``    — written on success AND failure:
  ``{status, question, report_path, summary, sub_question_count,
  evidence_count, iterations, errors, started_at, finished_at, error?}``.
* ``<out>/state.json``     — full final pipeline state.
* ``<out>/progress.jsonl`` — one line per event: run_start / node_done /
  run_end.

Logging goes to stderr; stdout stays clean. Exit code 0 iff status=="ok".
``--dry-run`` builds and validates the graph without any LLM/tool calls.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

logger = logging.getLogger("deep_research.run")


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m workflows.deep_research.run",
        description="Run the multi-agent deep_research pipeline once.",
    )
    parser.add_argument("--question", help="Research question")
    parser.add_argument(
        "--question-file", help="Read the question from a file instead",
    )
    parser.add_argument(
        "--out",
        default=f"runs/deep_research/{time.strftime('%Y%m%d-%H%M%S')}",
        help="Output directory (default: runs/deep_research/<timestamp>)",
    )
    parser.add_argument(
        "--pipeline",
        choices=["deep_research", "deep_council_research", "model_council"],
        default="deep_research",
        help=(
            "deep_research: single-model multi-agent research. "
            "deep_council_research: full research once per COUNCIL_MODEL_* "
            "member plus comparison tables. model_council: lightweight "
            "multi-model answers plus comparison tables (no web research)."
        ),
    )
    parser.add_argument(
        "--depth", choices=["quick", "standard", "deep"], default="standard",
    )
    parser.add_argument("--profile", default="default")
    parser.add_argument("--max-iterations", type=int, default=None)
    parser.add_argument("--max-revisions", type=int, default=None)
    parser.add_argument("--max-parallel", type=int, default=None)
    parser.add_argument(
        "--metadata-json", default=None,
        help="Extra JSON merged into metadata['deep_research']",
    )
    parser.add_argument(
        "--no-vault", action="store_true",
        help="Disable the evidence vault (sources are not stored/reused)",
    )
    parser.add_argument(
        "--peer-review", action="store_true",
        help=(
            "Council pipelines: add an anonymized peer-review round where "
            "members rank each other's answers blind (one extra LLM call "
            "per member; off by default)"
        ),
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Validate registration + graph build; no LLM calls",
    )
    args = parser.parse_args(argv)

    if args.question_file:
        args.question = Path(args.question_file).read_text(
            encoding="utf-8",
        ).strip()
    if not args.question and not args.dry_run:
        parser.error("--question (or --question-file) is required")
    return args


def _dr_overrides(args: argparse.Namespace) -> dict[str, Any]:
    dr: dict[str, Any] = {"depth": args.depth}
    if args.max_iterations is not None:
        dr["max_research_iterations"] = args.max_iterations
    if args.max_revisions is not None:
        dr["max_revisions"] = args.max_revisions
    if args.max_parallel is not None:
        dr["max_parallel_subagents"] = args.max_parallel
    if args.peer_review:
        dr["peer_review"] = True
    if args.metadata_json:
        dr.update(json.loads(args.metadata_json))
    return dr


def _dry_run() -> int:
    """Static wiring check: spec, graph, reducers, routing. No LLM calls."""
    from agent_harness.core.runtime.dag.graph_builder import DynamicGraphBuilder
    from agent_harness.core.runtime.dag.minidag import extract_reducers

    from workflows.deep_research.conditions import (
        route_after_conflict_check,
        route_after_review,
    )
    from workflows.deep_research.spec import DEEP_RESEARCH_SPEC
    from workflows.deep_research.state import (
        REDUCED_LIST_FIELDS,
        DeepResearchState,
    )

    dag = DynamicGraphBuilder().build(DEEP_RESEARCH_SPEC, DeepResearchState)
    node_ids = {n.node_id for n in DEEP_RESEARCH_SPEC.resolved_nodes}
    assert DEEP_RESEARCH_SPEC.entry_point in node_ids
    assert len(dag.nodes) == 9, f"expected 9 nodes, got {len(dag.nodes)}"
    assert "polish" in node_ids and DEEP_RESEARCH_SPEC.resolved_terminal_nodes == ["polish"]
    # The citation audit must sit between the verifier and the terminal
    # node — before polish so its Verification appendix gets cleaned too,
    # after final_verify so it audits the report that actually ships.
    assert "citation_audit" in dag.nodes
    assert dag.edges.get("final_verify") == "citation_audit"
    assert dag.edges.get("citation_audit") == "polish"

    # hyperresearch patch/citation engine sanity (pure functions, no LLM).
    from workflows.deep_research.patch import (
        apply_edit_hunks,
        strip_filler,
        validate_citations,
    )
    _t, _log = apply_edit_hunks(
        "A 30% rise [1].", [{"old": "30%", "new": "25%", "reason": "fix"}],
        citation_mapping={"1": {}},
    )
    assert "25%" in _t and _log[0]["applied"]
    assert apply_edit_hunks(
        "x [1]", [{"old": "x", "new": "x [9]", "reason": ""}],
        citation_mapping={"1": {}},
    )[1][0]["status"] == "breaks-citations"
    assert validate_citations("a [2]", {"1": {}}) == [2]
    assert strip_filler("Importantly, X.", ("Importantly, ",))[1] == 1

    # Council pipelines: graphs build, shared synthesis node resolves, and
    # the table renderers produce sane output for a sample council dict.
    from workflows.deep_research.council_tables import (
        normalize_council,
        render_council_html,
        render_markdown_tables,
        render_ranking_table,
    )
    from workflows.deep_research.spec import DEEP_COUNCIL_RESEARCH_SPEC
    from workflows.deep_research.state import CouncilState
    from workflows.model_council.spec import MODEL_COUNCIL_SPEC

    for spec in (DEEP_COUNCIL_RESEARCH_SPEC, MODEL_COUNCIL_SPEC):
        cdag = DynamicGraphBuilder().build(spec, CouncilState)
        assert len(cdag.nodes) == 3, f"{spec.pipeline_id}: bad node count"
        assert "peer_review" in cdag.nodes, spec.pipeline_id

    names = ["Model A", "Model B"]
    sample = normalize_council({
        "agreements": [{"finding": "f1", "models": names,
                        "evidence": "e1", "citations": ["src"]}],
        "disagreements": [{"topic": "t1",
                           "positions": {"Model A": "yes"},
                           "why_differ": "risk tolerance"}],
        "unique": [{"model": "Model B", "finding": "u1",
                    "why_it_matters": "matters"}],
    }, names)
    md = render_markdown_tables(sample, names)
    html_page = render_council_html(sample, names, "q?", "light")
    assert "Where Models Agree" in md and "✓" in md
    assert "Not addressed" in md, "missing member must be filled in"
    assert "Model B" in html_page and "✓" in html_page
    assert "<style>" in html_page  # self-contained: inline CSS only
    assert "src=" not in html_page and "href=" not in html_page

    # Peer review: ballots are per-evaluator shuffled and self-excluded;
    # aggregation is normalized Borda with deterministic tie-breaks.
    import random as _random

    from workflows.deep_research.peer_review import (
        aggregate_rankings,
        build_ballots,
        parse_ballot,
    )
    _members = [
        {"name": n, "slug": n.lower(), "status": "ok", "report": f"a{n}"}
        for n in ("Alpha", "Beta", "Gamma")
    ]
    _ballots = build_ballots(_members, rng=_random.Random(0))
    assert len(_ballots) == 3
    for _b in _ballots:
        assert _b["evaluator"] not in _b["label_to_member"].values()
    assert parse_ballot("garbage", {"A": "Beta"})["valid"] is False
    _agg = aggregate_rankings([
        {"valid": True, "ranking": ["Beta", "Gamma"]},
        {"valid": True, "ranking": ["Beta", "Alpha"]},
    ])
    assert _agg[0]["model"] == "Beta" and _agg[0]["peer_score"] == 1.0
    _ranking_md = render_ranking_table(_agg, [])
    assert "Peer Review Ranking" in _ranking_md and "Beta" in _ranking_md
    assert render_ranking_table([], []) == ""  # no ranking → no empty shell
    _html_pr = render_council_html(
        sample, names, "q?", "light", peer_ranking=_agg, peer_reviews=[],
    )
    assert "Peer Review Ranking" in _html_pr and "src=" not in _html_pr

    reducers = extract_reducers(DeepResearchState)
    for field in REDUCED_LIST_FIELDS:
        assert field in reducers, f"missing reducer for {field}"

    # Routing: both branches + bound enforcement, on synthetic states.
    meta = {"deep_research": {"depth": "standard"}}
    assert route_after_conflict_check({
        "gap_questions": [{"question": "g"}],
        "research_iteration": 1, "metadata": meta,
    }) == "research_fanout"
    assert route_after_conflict_check({
        "gap_questions": [{"question": "g"}],
        "research_iteration": 2, "metadata": meta,
    }) == "draft"
    assert route_after_conflict_check({
        "gap_questions": [], "research_iteration": 1, "metadata": meta,
    }) == "draft"
    assert route_after_review({
        "review_verdict": "revise", "revision_count": 1, "metadata": meta,
    }) == "draft"
    assert route_after_review({
        "review_verdict": "revise", "revision_count": 2, "metadata": meta,
    }) == "final_verify"
    assert route_after_review({
        "review_verdict": "approve", "revision_count": 0, "metadata": meta,
    }) == "final_verify"

    # Citation-integrity engine (pure functions, no LLM): the whitelist
    # binding, orphan detection, and the numeric audit that separates a
    # supported figure from a fabricated one carrying a valid citation.
    from workflows._shared.cited_report_finalizer import (
        validate_citation_body,
        validate_numeric_grounding,
    )
    from workflows._shared.citation_contract import (
        compose_citation_contract,
        finalize_report_with_canonical_references,
    )
    from workflows.deep_research.references import (
        mapping_from_references,
        references_from_mapping,
        renumbered_references,
        snippet_lookup_for,
    )

    _map = {
        "1": {"url": "https://a.example/r", "title": "A"},
        "2": {"url": "https://b.example/r", "title": "B"},
    }
    _refs = references_from_mapping(_map)
    assert mapping_from_references(_refs) == _map
    assert compose_citation_contract([]) == ""
    assert "[2]" in compose_citation_contract(_refs)
    assert validate_citation_body("x [9]", max_ref=2, valid_indices=[1, 2])
    assert validate_citation_body("x [1]", max_ref=2, valid_indices=[1, 2]) == []
    _cards = [{
        "card_id": "E1", "claim": "", "quote": "Revenue was $5 billion.",
        "sources": [{"url": "https://a.example/r"}],
    }]
    _audit = validate_numeric_grounding(
        "Revenue was $5 billion [1]. Costs were $9.9 billion [2].",
        snippet_lookup=snippet_lookup_for(_refs, _cards),
    )
    assert [t for t, _ in _audit.suspect] == ["$9.9 billion"], _audit.suspect
    _body = "Second first [2], then first [1]."
    assert [r["url"] for r in renumbered_references(_body, _refs)] == [
        "https://b.example/r", "https://a.example/r",
    ]
    assert "[1] B" in finalize_report_with_canonical_references(
        _body, references=_refs,
    )

    print(
        f"deep_research dry-run OK: 3 pipelines build "
        f"(deep_research={len(dag.nodes)} nodes), reducers wired, routing "
        f"sane, council renderers sane, patch/citation engine sane, "
        f"citation contract + numeric audit sane",
    )
    return 0


async def _execute(args: argparse.Namespace, out_dir: Path) -> dict[str, Any]:
    """Bootstrap the kernel, stream the pipeline, return the final state."""
    from benchmarks.core.kernel_adapter import BenchmarkSession

    progress_path = out_dir / "progress.jsonl"

    def emit(event: dict[str, Any]) -> None:
        event = {"ts": _utc_now(), **event}
        with progress_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
        print(f"PHASE {json.dumps(event, ensure_ascii=False)}", file=sys.stderr)

    async with BenchmarkSession() as session:
        task = await session.pm.create_task(args.question)
        # Seed dict mirrors BenchmarkSession.run (kernel_adapter) — list
        # fields MUST pre-exist as [] for the state reducers to apply.
        input_data: dict[str, Any] = {
            "task_id": str(task.id),
            "original_question": args.question,
            "metadata": {
                "profile": args.profile,
                "deep_research": _dr_overrides(args),
            },
            "sub_questions": [],
            "evidence_cards": [],
            "research_notes": [],
            "fact_check_results": [],
            "conflicts": [],
            "gap_questions": [],
            "contradiction_graph": [],
            "consensus_claims": [],
            "loci": [],
            "review_feedback": [],
            "citation_mapping": {},
            "research_iteration": 0,
            "revision_count": 0,
            "patch_log": [],
            "polish_log": [],
            "report": None,
            "final_content": "",
            "current_phase": "",
            "errors": [],
        }
        # Evidence vault lives under the run's output dir (unless disabled).
        if not args.no_vault:
            input_data["vault_dir"] = str(out_dir)
        else:
            input_data["metadata"]["deep_research"]["use_vault"] = False
        if args.pipeline != "deep_research":
            input_data["metadata"]["council_mode"] = (
                "deep" if args.pipeline == "deep_council_research"
                else "light"
            )
            input_data.update({
                "members": [],
                "member_results": [],
                "peer_reviews": [],
                "peer_ranking": [],
                "council": {},
                "council_tables_md": "",
                "synthesis": "",
            })

        emit({"event": "run_start", "question": args.question,
              "pipeline": args.pipeline,
              "depth": args.depth, "profile": args.profile})
        async for _mode, chunk in session.scheduler.execute(
            task.id, input_data, pipeline_id=args.pipeline,
        ):
            if isinstance(chunk, dict) and chunk:
                node = next(iter(chunk))
                delta = chunk[node] if isinstance(chunk[node], dict) else {}
                emit({
                    "event": "node_done",
                    "node": node,
                    "iteration": delta.get("research_iteration"),
                })
        state = await session.scheduler.get_state(task.id) or {}
        emit({"event": "run_end"})
        return state


def _write_council_artifacts(
    args: argparse.Namespace, out_dir: Path, state: dict[str, Any],
) -> dict[str, Any]:
    """Write per-member outputs + council.json + council.html; return the
    result.json fields describing them."""
    from workflows.deep_research.council_tables import render_council_html

    mode = "deep" if args.pipeline == "deep_council_research" else "light"
    artifact_dir = out_dir / ("papers" if mode == "deep" else "answers")
    artifact_dir.mkdir(parents=True, exist_ok=True)

    member_results = state.get("member_results") or []
    member_papers: dict[str, str] = {}
    for res in member_results:
        if res.get("status") != "ok" or not res.get("report"):
            continue
        path = artifact_dir / f"{res['slug']}.md"
        path.write_text(res["report"], encoding="utf-8")
        member_papers[res["name"]] = str(path.resolve())

    member_names = [
        r["name"] for r in member_results if r.get("status") == "ok"
    ]
    council = state.get("council") or {}

    peer_ranking = state.get("peer_ranking") or []
    peer_reviews = state.get("peer_reviews") or []

    council_json_path = out_dir / "council.json"
    council_json_path.write_text(
        json.dumps({
            "question": state.get("original_question", ""),
            "mode": args.pipeline,
            "members": member_names,
            "synthesis": state.get("synthesis", ""),
            **council,
            "peer_ranking": peer_ranking,
            "peer_reviews": peer_reviews,
        }, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    council_html_path = out_dir / "council.html"
    council_html_path.write_text(
        render_council_html(
            council, member_names,
            state.get("original_question", ""), mode,
            peer_ranking=peer_ranking, peer_reviews=peer_reviews,
        ),
        encoding="utf-8",
    )

    return {
        "members": member_names,
        "member_papers": member_papers,
        "council_json_path": str(council_json_path.resolve()),
        "council_html_path": str(council_html_path.resolve()),
        "peer_review_enabled": bool(args.peer_review),
        "peer_ranking": peer_ranking,
    }


def _write_enhancement_artifacts(
    out_dir: Path, state: dict[str, Any],
) -> dict[str, Any]:
    """Write patch-log / polish-log / vault stats; return result.json fields."""
    fields: dict[str, Any] = {}
    patch_log = state.get("patch_log") or []
    if patch_log:
        p = out_dir / "patch-log.json"
        p.write_text(json.dumps(patch_log, ensure_ascii=False, indent=2),
                     encoding="utf-8")
        fields["patch_log_path"] = str(p.resolve())
    polish_log = state.get("polish_log") or []
    if polish_log:
        p = out_dir / "polish-log.json"
        p.write_text(json.dumps(polish_log, ensure_ascii=False, indent=2),
                     encoding="utf-8")
        fields["polish_log_path"] = str(p.resolve())
    vault_dir = state.get("vault_dir")
    if vault_dir:
        notes = Path(vault_dir) / "vault" / "research" / "notes"
        fields["vault_dir"] = str((Path(vault_dir) / "vault").resolve())
        fields["vault_source_count"] = (
            len(list(notes.glob("*.md"))) if notes.exists() else 0
        )
    audit = state.get("citation_audit") or {}
    numeric = state.get("numeric_grounding") or {}
    if audit or numeric:
        p = out_dir / "citation-audit.json"
        p.write_text(
            json.dumps(
                {"citation_audit": audit, "numeric_grounding": numeric},
                ensure_ascii=False, indent=2,
            ),
            encoding="utf-8",
        )
        fields["citation_audit_path"] = str(p.resolve())
        # Keep result.json itself to a scannable summary — the full token
        # lists live in the artifact above.
        fields["citation_audit"] = {
            "reference_count": audit.get("reference_count", 0),
            "issue_count": len(audit.get("issues") or []),
            "repaired": bool(audit.get("repaired")),
            "numbers_checked": numeric.get("total", 0),
            "numbers_suspect": numeric.get("suspect", 0),
            "suspect_ratio": numeric.get("suspect_ratio", 0.0),
        }
    return fields


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stderr,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    load_dotenv()
    args = _parse_args(argv)

    if args.dry_run:
        return _dry_run()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    started_at = _utc_now()

    result: dict[str, Any] = {
        "status": "error",
        "mode": args.pipeline,
        "question": args.question,
        "report_path": "",
        "summary": "",
        "sub_question_count": 0,
        "evidence_count": 0,
        "iterations": 0,
        "errors": [],
        "started_at": started_at,
        "finished_at": "",
    }

    try:
        state = asyncio.run(_execute(args, out_dir))

        report = state.get("report") or state.get("final_content") or ""
        if not report:
            raise RuntimeError("pipeline finished without a report")

        report_path = out_dir / "report.md"
        report_path.write_text(report, encoding="utf-8")
        (out_dir / "state.json").write_text(
            json.dumps(state, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        result.update({
            "status": "ok",
            "report_path": str(report_path.resolve()),
            "summary": state.get("verification_summary", ""),
            "sub_question_count": len(state.get("sub_questions") or []),
            "evidence_count": len(state.get("evidence_cards") or []),
            "iterations": int(state.get("research_iteration", 0)),
            "errors": list(state.get("errors") or []),
            "loci_count": len(state.get("loci") or []),
            "contradiction_clusters": len(state.get("contradiction_graph") or []),
        })
        result.update(_write_enhancement_artifacts(out_dir, state))
        if args.pipeline != "deep_research":
            result.update(_write_council_artifacts(args, out_dir, state))
    except Exception as exc:  # noqa: BLE001 — result.json carries the error
        logger.exception("deep_research run failed")
        result["error"] = f"{type(exc).__name__}: {exc}"

    result["finished_at"] = _utc_now()
    (out_dir / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("result.json written to %s (status=%s)",
                out_dir, result["status"])
    return 0 if result["status"] == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
