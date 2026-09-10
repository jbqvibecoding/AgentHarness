"""Workflow-scoped default lookup."""

from __future__ import annotations

import importlib
import logging
from typing import Any

logger = logging.getLogger(__name__)


# pipeline_id → dotted path of the workflow's defaults module.
# Add a row when a new workflow ships a ``defaults.py``.
_PIPELINE_TO_DEFAULTS: dict[str, str] = {
    # deep_research budgets its own research phases and must not have the
    # write-up, verification and citation audit cancelled by the graph-wide
    # ceiling — the finalization reserve is the whole point.
    "deep_research": "workflows.deep_research.defaults",
    "deep-research": "workflows.deep_research.defaults",
    "deep_council_research": "workflows.deep_research.defaults",
    "deep-council-research": "workflows.deep_research.defaults",
}


def get_workflow_default(pipeline_id: str | None, attr: str) -> Any:
    """Look up ``attr`` from ``pipeline_id``'s workflow defaults module.

    Returns ``None`` when:

    - ``pipeline_id`` is falsy or unrecognised;
    - the registered defaults module is missing or fails to import;
    - the module does not define ``attr``.

    Callers should treat ``None`` as "no workflow default" and continue
    their existing fallback chain (env var, then no-deadline / default).
    """
    if not pipeline_id:
        return None
    module_path = _PIPELINE_TO_DEFAULTS.get(pipeline_id)
    if module_path is None:
        return None
    try:
        module = importlib.import_module(module_path)
    except ImportError as e:
        logger.debug(
            "workflow_defaults: failed to import %s (pipeline=%s): %s",
            module_path, pipeline_id, e,
        )
        return None
    return getattr(module, attr, None)
