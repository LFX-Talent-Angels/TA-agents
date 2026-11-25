"""map-rank-v1: pick which neighbours a Graph-Mapper step keeps (ADR-0011).

The walker passes candidates with its own sort key; this returns the chosen
order. Generic hubs lose to specific neighbours at equal relation and strength.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from math import log2

from talent_angels.skills.evaluate.policy import MAP_RANK_V1, MapRankPolicy

# Walker key: (relation_rank, strength_rank, -importance, hot_flag, label, id).
Key = tuple[int, int, float, int, str, str]


def rerank_neighbours(
    fresh: Sequence[tuple[str, Key]],
    breadth: int,
    degree_of: Callable[[str], int | None],
    policy: MapRankPolicy = MAP_RANK_V1,
) -> list[str]:
    """Top ``breadth`` ids of ``fresh`` (already in key order) after specificity re-rank.

    ``degree_of`` returns None once the lookup budget is spent; such a member
    gets specificity 0, so it falls back to its plain key order.
    """
    pool = fresh[: max(breadth, min(policy.pool_factor * breadth, policy.pool_cap))]
    scored: list[tuple[tuple[object, ...], str]] = []
    for node_id, key in pool:
        degree = degree_of(node_id)
        specificity = 1 / log2(2 + degree) if degree is not None else 0.0
        scored.append(((*key[:4], -specificity, *key[4:]), node_id))
    scored.sort(key=lambda item: item[0])  # type: ignore[arg-type,return-value]
    return [node_id for _, node_id in scored[:breadth]]
