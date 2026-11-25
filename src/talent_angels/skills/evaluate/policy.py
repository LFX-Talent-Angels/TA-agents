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


@dataclass(frozen=True)
class LearnOrderPolicy:
    """learn-order-v1: the order of a roadmap (ADR-0013). Same data in, same order out.

    Steps: 1 knowledge; 2 skills of reuse level ``reuse_order[0:2]`` (and O*NET skills);
    3 sector-specific; 4 occupation-specific. A stated ESCO essential-for link puts the
    needed skill first; within a step: level jump, importance, label, node ID.
    """

    name: str = "learn-order-v1"
    meaning: str = (
        "ordered by learn-order-v1 over official taxonomy data; only items marked "
        "stated_prerequisite are prerequisites stated by a taxonomy"
    )
    step_names: tuple[str, ...] = ("Foundations", "Core", "Field", "Job-specific")
    reuse_order: tuple[str, ...] = (
        "transversal",
        "cross-sector",
        "sector-specific",
        "occupation-specific",
    )
    reuse_steps: tuple[int, ...] = (2, 2, 3, 4)  # step per ``reuse_order`` entry
    default_step: int = 2  # a skill whose reuse level the graph does not give
    min_importance: float = 3.0  # O*NET: B's importance at least this
    min_level_jump: float = 0.5  # O*NET: B's level above A's by at least this
    level_digits: int = 2
    lookup_budget: int = 300  # extra get_neighbors calls to read a node's own properties
    related_pool: int = 30  # related skills whose reuse level is read per node (successors)


ROUTE_RANK_V1 = RouteRankPolicy()
MAP_RANK_V1 = MapRankPolicy()
LEARN_ORDER_V1 = LearnOrderPolicy()
