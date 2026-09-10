"""Deliverable-file policy — inert in this repo.

``assign_task`` (ported from FrontierAgent) appends prompt blocks telling a
sub-agent where to write deliverable *files*: a sandboxed ``/outputs`` tree
with a workspace root, path quotas and retirement rules. That whole model
belongs to FrontierAgent's sandbox, which this repo does not have.

Our sub-agents return structured findings (evidence cards, verdicts,
report bodies) through ``submit_report`` and the agent bus, not files on a
mounted volume. So the five functions ``assign_task`` calls are provided
here as honest no-ops rather than porting 1500 lines of path policy for a
filesystem layout nothing here creates.

They are deliberately *not* stubs that raise: a caller reaching one is
doing something reasonable, and the correct answer in this repo is "there
is no deliverable path to assign", not a crash.

If a file-producing workflow lands here later, this is the seam to fill:
implement these five against whatever output root that workflow defines,
and ``assign_task`` starts emitting the directives with no change.
"""

from __future__ import annotations

from collections.abc import Iterable

__all__ = [
    "normalise_output_paths",
    "output_write_directives",
    "render_publish_assignment",
    "render_retirement_note",
    "render_workspace_assignment",
]


def normalise_output_paths(paths: Iterable[str]) -> tuple[str, ...]:
    """De-duplicate an output manifest, preserving order.

    No path validation: without an output root there is nothing to
    validate against, and rejecting paths a caller supplied would be a
    guess rather than a rule.
    """
    seen: list[str] = []
    for raw in paths or ():
        path = str(raw).strip()
        if path and path not in seen:
            seen.append(path)
    return tuple(seen)


def output_write_directives(text: str) -> tuple[str, ...]:  # noqa: ARG001
    """Deliverable paths ``text`` directs a write to — always none here."""
    return ()


def render_publish_assignment(paths: Iterable[str]) -> str:  # noqa: ARG001
    """Prompt block for a publishing task — empty: nothing to publish to."""
    return ""


def render_retirement_note(paths: Iterable[str]) -> str:  # noqa: ARG001
    """Prompt block retiring superseded deliverables — nothing to retire."""
    return ""


def render_workspace_assignment(
    inherited_paths: Iterable[str] = (),  # noqa: ARG001
    publisher_declared: bool = True,  # noqa: ARG001, FBT001, FBT002
) -> str:
    """Prompt block describing the workspace — empty: there is no workspace."""
    return ""
