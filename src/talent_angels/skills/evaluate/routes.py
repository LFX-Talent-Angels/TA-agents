"""rank_routes: order Pathfind routes under route-rank-v1. Reads routes, never changes them."""

from __future__ import annotations

from dataclasses import dataclass
from math import log2

from talent_angels.contracts import AgentResult, Route
from talent_angels.skills.connect.reveal import ConnectableSuite
from talent_angels.skills.evaluate.policy import ROUTE_RANK_V1, RouteRankPolicy

_STRENGTH = {"essential": 1.0, "optional": 0.5, "transferable": 0.5}
_UNTYPED = 0.25


@dataclass(frozen=True)
class RankedRoute:
    route: Route
    score: float
    reasons: list[str]


def rank_routes(
    suite: ConnectableSuite, result: AgentResult, policy: RouteRankPolicy = ROUTE_RANK_V1
) -> list[RankedRoute]:
    labels = {n.id: n.pref_label for n in result.nodes}
    degrees: dict[str, int] = {}

    def degree(node_id: str) -> int:
        if node_id not in degrees:
            degrees[node_id] = len(suite.get_neighbors(node_id).edges)
        return degrees[node_id]

    scored: list[tuple[float, str, tuple[str, ...], RankedRoute]] = []
    for route in result.routes:
        steps = max(1, len(route.node_ids) - 1)
        values = [
            _STRENGTH.get(str(e.properties.get("relation_type")), _UNTYPED) for e in route.edges
        ]
        strength = sum(values) / len(values) if values else _UNTYPED
        inner = route.node_ids[1:-1]
        specs = [1 / log2(2 + degree(i)) for i in inner]
        specificity = sum(specs) / len(specs) if specs else 1.0
        score = round(
            policy.w_strength * strength
            + policy.w_specificity * specificity
            + policy.w_length / steps,
            4,
        )
        essential = sum(1 for v in values if v == 1.0)
        reasons = [f"{essential}/{len(values)} links essential"] if values else []
        if inner:
            reasons.append(f"bridge used by {max(degree(i) for i in inner)} items")
        reasons.append(f"{steps} step{'s' if steps != 1 else ''}")
        joined = " > ".join(labels.get(i, i) for i in route.node_ids)
        scored.append((-score, joined, tuple(route.node_ids), RankedRoute(route, score, reasons)))
    return [item[3] for item in sorted(scored, key=lambda s: s[:3])]
