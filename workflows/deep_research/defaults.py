"""Workflow-scoped scheduler defaults for ``deep_research``.

Read by :mod:`agent_harness.scheduling.workflow_defaults`, which the
Scheduler consults when resolving the graph-wide wall-time ceiling.

``TASK_WALL_TIME_MODE = "soft_research"`` says: this workflow budgets its
own research phases and needs the scheduler's ceiling to sit *above* that
budget rather than equal to it.

Without it the failure mode is specific and bad. The research fan-out and
its re-research loop will happily consume the entire wall — that is what
the iteration budget is for — and the scheduler would then cancel the task
in `draft`, `final_verify` or `citation_audit`, i.e. after paying for all
the research and before producing anything a user can read. The scheduler
requires a non-empty terminal ``report``, so that cancellation costs the
run its whole deliverable.

Splitting the wall means the research stages share
``soft_wall_deadline_s(total, RESEARCH_FINALIZE_RESERVE_S)`` and the
write-up stages get what is left.
"""

from __future__ import annotations

# Tells the scheduler this workflow owns its research deadline.
TASK_WALL_TIME_MODE = "soft_research"

# Graph-wide hard ceiling. Deliberately generous: the real pacing comes
# from the research deadline below plus the per-branch subagent timeouts,
# and this only exists to stop a wedged run from living forever.
TASK_WALL_TIME_S = 5400

# Wall-clock reserved for everything after research: draft, review,
# final_verify, citation_audit, polish. Sized for a full revision loop plus
# one citation-repair round on a large report.
#
# ``soft_wall_deadline_s`` floors the research share at half the wall, so on
# a short wall the reserve that actually survives is smaller than this —
# which is why phase timeouts are clamped through
# ``remaining_phase_budget_s`` rather than assuming the full reserve.
RESEARCH_FINALIZE_RESERVE_S = 900.0

__all__ = [
    "RESEARCH_FINALIZE_RESERVE_S",
    "TASK_WALL_TIME_MODE",
    "TASK_WALL_TIME_S",
]
