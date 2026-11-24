"""Named, versioned ranking policies. A change means a new name, never an edit (ADR-0011)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RouteRankPolicy:
    """route-rank-v1: ``score = w_strength*strength + w_specificity*specificity + w_length*length``.

    strength: mean over route edges of essential=1.0, optional/transferable=0.5, untyped=0.25.
    specificity: mean over inner (bridge) nodes of ``1/log2(2+degree)``; 1.0 with no bridge.
    length: ``1/number_of_edges``.
    """

    name: str = "route-rank-v1"
    w_strength: float = 0.5
    w_specificity: float = 0.35
    w_length: float = 0.15


@dataclass(frozen=True)
class MapRankPolicy:
    """map-rank-v1: re-rank a pool of ``min(pool_factor*breadth, pool_cap)`` neighbours.

    Sort key: relation, strength, importance, hot flag, then ``-specificity``
    (``1/log2(2+degree)``), label, id. At most ``degree_budget`` degree lookups
    per walk; past it the old key order is kept for the rest.
    """

    name: str = "map-rank-v1"
    pool_factor: int = 5
    pool_cap: int = 30
    degree_budget: int = 2000


ROUTE_RANK_V1 = RouteRankPolicy()
MAP_RANK_V1 = MapRankPolicy()
