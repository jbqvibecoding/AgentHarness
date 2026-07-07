"""Renderers for the council comparison tables (markdown + HTML).

The council dict shape (produced by the analyst, normalized by
``normalize_council``):

    {"agreements":    [{finding, models: [name], evidence, citations: [str]}],
     "disagreements": [{topic, positions: {name: str}, why_differ}],
     "unique":        [{model, finding, why_it_matters, citations: [str]}]}

``render_council_html`` emits a self-contained page (inline CSS, no
external resources, everything escaped) with the three card-style tables
mirroring the reference design: rounded cards, grey header row, check
marks per member column, small evidence tags.
"""

from __future__ import annotations

import html
from typing import Any

_MAX_CELL = 500


def _cell(text: Any, limit: int = _MAX_CELL) -> str:
    out = str(text or "").strip()
    if len(out) > limit:
        out = out[:limit] + "…"
    return out


def _md_escape(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def normalize_council(
    council: Any, member_names: list[str],
) -> dict[str, list[dict[str, Any]]]:
    """Validate/trim the analyst JSON into the canonical council shape.

    Unknown member names are dropped; missing members in a disagreement
    row become "Not addressed"; agreements backed by fewer than 2 valid
    members are discarded.
    """
    known = set(member_names)
    out: dict[str, list[dict[str, Any]]] = {
        "agreements": [], "disagreements": [], "unique": [],
    }
    if not isinstance(council, dict):
        return out

    for row in council.get("agreements") or []:
        if not isinstance(row, dict) or not row.get("finding"):
            continue
        models = [m for m in (row.get("models") or []) if m in known]
        if len(models) < 2:
            continue
        out["agreements"].append({
            "finding": _cell(row["finding"]),
            "models": models,
            "evidence": _cell(row.get("evidence")),
            "citations": [
                _cell(c, 120) for c in (row.get("citations") or [])
            ][:4],
        })

    for row in council.get("disagreements") or []:
        if not isinstance(row, dict) or not row.get("topic"):
            continue
        raw_positions = row.get("positions") or {}
        positions = {
            name: _cell(raw_positions.get(name) or "Not addressed", 400)
            for name in member_names
        }
        out["disagreements"].append({
            "topic": _cell(row["topic"], 200),
            "positions": positions,
            "why_differ": _cell(row.get("why_differ")),
        })

    for row in council.get("unique") or []:
        if not isinstance(row, dict) or not row.get("finding"):
            continue
        model = str(row.get("model") or "")
        if model not in known:
            continue
        out["unique"].append({
            "model": model,
            "finding": _cell(row["finding"]),
            "why_it_matters": _cell(row.get("why_it_matters")),
            "citations": [
                _cell(c, 120) for c in (row.get("citations") or [])
            ][:4],
        })
    return out


def degraded_council(
    member_results: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Fallback tables when the analyst fails: no cross-analysis, but each
    member's headline finding survives as a unique row so the deliverable
    never ships empty."""
    unique = []
    for res in member_results:
        if res.get("status") != "ok":
            continue
        headline = (res.get("report") or "").strip().split("\n\n")[0]
        unique.append({
            "model": res.get("name", "?"),
            "finding": _cell(headline or "(no output)"),
            "why_it_matters": "Analyst unavailable — headline of this member's output.",
            "citations": [],
        })
    return {"agreements": [], "disagreements": [], "unique": unique}


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------

def render_markdown_tables(
    council: dict[str, Any], member_names: list[str],
) -> str:
    parts: list[str] = []

    parts.append("## Where Models Agree\n")
    if council.get("agreements"):
        header = "| Finding | " + " | ".join(member_names) + " | Evidence |"
        sep = "|" + "---|" * (len(member_names) + 2)
        rows = [header, sep]
        for row in council["agreements"]:
            marks = " | ".join(
                "✓" if name in row["models"] else ""
                for name in member_names
            )
            evidence = _md_escape(row["evidence"])
            if row.get("citations"):
                evidence += " `" + "` `".join(row["citations"]) + "`"
            rows.append(
                f"| {_md_escape(row['finding'])} | {marks} | {evidence} |"
            )
        parts.append("\n".join(rows))
    else:
        parts.append("*No cross-model agreements identified.*")

    parts.append("\n## Where Models Disagree\n")
    if council.get("disagreements"):
        header = "| Topic | " + " | ".join(member_names) + " | Why They Differ |"
        sep = "|" + "---|" * (len(member_names) + 2)
        rows = [header, sep]
        for row in council["disagreements"]:
            positions = " | ".join(
                _md_escape(row["positions"].get(name, "Not addressed"))
                for name in member_names
            )
            rows.append(
                f"| {_md_escape(row['topic'])} | {positions} | "
                f"{_md_escape(row['why_differ'])} |"
            )
        parts.append("\n".join(rows))
    else:
        parts.append("*No material disagreements identified.*")

    parts.append("\n## Unique Discoveries\n")
    if council.get("unique"):
        rows = [
            "| Model | Unique Finding | Why It Matters |",
            "|---|---|---|",
        ]
        for row in council["unique"]:
            finding = _md_escape(row["finding"])
            if row.get("citations"):
                finding += " `" + "` `".join(row["citations"]) + "`"
            rows.append(
                f"| {row['model']} | {finding} | "
                f"{_md_escape(row['why_it_matters'])} |"
            )
        parts.append("\n".join(rows))
    else:
        parts.append("*No unique single-model discoveries identified.*")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------

_CSS = """
body { margin: 0; padding: 32px 16px; background: #10141a;
       font-family: -apple-system, 'Segoe UI', Roboto, 'PingFang SC',
       'Microsoft YaHei', sans-serif; color: #1d2b33; }
.wrap { max-width: 1080px; margin: 0 auto; }
h1.q { color: #e8ecef; font-size: 20px; font-weight: 600; margin: 0 0 6px; }
p.meta { color: #93a1ab; font-size: 13px; margin: 0 0 24px; }
.card { background: #fbfaf6; border-radius: 18px; padding: 28px 32px;
        margin-bottom: 28px; box-shadow: 0 10px 30px rgba(0,0,0,.35); }
.card h2 { font-size: 22px; font-weight: 650; margin: 0 0 18px; }
table { width: 100%; border-collapse: collapse; font-size: 14.5px; }
th, td { text-align: left; vertical-align: top; padding: 14px 14px;
         border: 1px solid #e6e3da; line-height: 1.5; }
th { background: #f2f0e9; font-weight: 600; color: #33424c; }
td.mark { text-align: center; width: 56px; color: #1d2b33;
          font-size: 16px; }
th.model { text-align: center; white-space: nowrap; }
.tag { display: inline-block; background: #ecebe4; color: #5a6570;
       border-radius: 6px; padding: 1px 8px; font-size: 12px;
       font-family: ui-monospace, 'SF Mono', Menlo, monospace;
       margin-left: 6px; white-space: nowrap; }
.model-name { font-weight: 600; white-space: nowrap; }
"""


def _tags(citations: list[str]) -> str:
    return "".join(
        f'<span class="tag">{html.escape(c)}</span>' for c in citations
    )


def render_council_html(
    council: dict[str, Any],
    member_names: list[str],
    question: str,
    mode: str,
) -> str:
    esc = html.escape
    member_ths = "".join(
        f'<th class="model">{esc(n)}</th>' for n in member_names
    )

    agree_rows = "".join(
        "<tr>"
        f"<td>{esc(row['finding'])}</td>"
        + "".join(
            f'<td class="mark">{"✓" if n in row["models"] else ""}</td>'
            for n in member_names
        )
        + f"<td>{esc(row['evidence'])}{_tags(row.get('citations') or [])}</td>"
        "</tr>"
        for row in council.get("agreements") or []
    ) or (
        f'<tr><td colspan="{len(member_names) + 2}">'
        "No cross-model agreements identified.</td></tr>"
    )

    disagree_rows = "".join(
        "<tr>"
        f"<td>{esc(row['topic'])}</td>"
        + "".join(
            f"<td>{esc(row['positions'].get(n, 'Not addressed'))}</td>"
            for n in member_names
        )
        + f"<td>{esc(row['why_differ'])}</td>"
        "</tr>"
        for row in council.get("disagreements") or []
    ) or (
        f'<tr><td colspan="{len(member_names) + 2}">'
        "No material disagreements identified.</td></tr>"
    )

    unique_rows = "".join(
        "<tr>"
        f'<td class="model-name">{esc(row["model"])}</td>'
        f"<td>{esc(row['finding'])}{_tags(row.get('citations') or [])}</td>"
        f"<td>{esc(row['why_it_matters'])}</td>"
        "</tr>"
        for row in council.get("unique") or []
    ) or '<tr><td colspan="3">No unique single-model discoveries.</td></tr>'

    mode_label = (
        "Deep Council Research" if mode == "deep" else "Model Council"
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Council Report</title>
<style>{_CSS}</style>
</head>
<body>
<div class="wrap">
<h1 class="q">{esc(question)}</h1>
<p class="meta">{esc(mode_label)} · Members: {esc(", ".join(member_names))}</p>

<div class="card">
<h2>Where Models Agree</h2>
<table>
<tr><th>Finding</th>{member_ths}<th>Evidence</th></tr>
{agree_rows}
</table>
</div>

<div class="card">
<h2>Where Models Disagree</h2>
<table>
<tr><th>Topic</th>{member_ths}<th>Why They Differ</th></tr>
{disagree_rows}
</table>
</div>

<div class="card">
<h2>Unique Discoveries</h2>
<table>
<tr><th>Model</th><th>Unique Finding</th><th>Why It Matters</th></tr>
{unique_rows}
</table>
</div>

</div>
</body>
</html>
"""


__all__ = [
    "degraded_council",
    "normalize_council",
    "render_council_html",
    "render_markdown_tables",
]
