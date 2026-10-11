"""Pathfind: a skill gap, and (ADR-0008) later a set of bridge routes.

Deterministic — set arithmetic and a depth-capped graph traversal, no model
call. See ``docs/decisions/0008-pathfind-contract.md`` for the contract this
package implements.
"""

from __future__ import annotations

from talent_angels.skills.pathfind.gap import SkillGap, compute_gap

__all__ = ["SkillGap", "compute_gap"]
