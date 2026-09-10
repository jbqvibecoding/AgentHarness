"""Re-export of the shared duplicate-query rollback observer.

The implementation moved to
``agent_harness.components.observers.duplicate_query_rollback`` when
``deep_research`` started using it too — it has no react_base coupling.
This module stays so existing imports keep working.
"""

from __future__ import annotations

from agent_harness.components.observers.duplicate_query_rollback import (
    DuplicateQueryRollbackObserver,
)

__all__ = ["DuplicateQueryRollbackObserver"]
