"""Pathfind: a skill gap, and a set of bridge routes (ADR-0008).

Deterministic — set arithmetic and a depth-capped graph traversal, no model
call. See ``docs/decisions/0008-pathfind-contract.md`` for the contract this
package implements.
"""

from __future__ import annotations

from talent_angels.skills.pathfind.dispatch import pathfind_one
from talent_angels.skills.pathfind.gap import SkillGap, compute_gap
from talent_angels.skills.pathfind.routes import BridgeRoute, BridgeRoutes, find_routes

__all__ = ["BridgeRoute", "BridgeRoutes", "SkillGap", "compute_gap", "find_routes", "pathfind_one"]
