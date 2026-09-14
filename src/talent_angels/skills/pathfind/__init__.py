"""Pathfind (Compose): routes between two resolved nodes.

Chains neighbor traversal in code (depth-capped, cycle-free) — composition
lives here, not in agent-to-agent calls.
"""

from talent_angels.skills.pathfind.compose import pathfind

__all__ = ["pathfind"]
