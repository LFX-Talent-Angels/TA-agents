"""Roadmap A -> B: what B needs that A does not, in the order of policy learn-order-v1 (ADR-0013).

Per taxonomy where A and B both resolve with the Graph-Mapper's start rule (a clear match,
else candidates; node IDs accepted). The gap, per taxonomy: ESCO B's essential skills minus A's;
JobTech B's skills minus A's; O*NET B's skills and knowledge with importance >= 3.0 whose level
for B exceeds A's by >= 0.5; BLS B's entry education, experience and training as credentials.
Then the order: a stated ESCO essential-for link first, then the step, then the level jump,
importance, label and node ID. No model call, no inferred edge: same input and graph data give
the same JSON.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from talent_angels.contracts import AgentResult
from talent_angels.skills.connect import ConnectRequest, connect
from talent_angels.skills.connect.federated import _select
from talent_angels.skills.connect.neighbourhood import _node_ref
from talent_angels.skills.connect.reveal import TaxonomyNode
from talent_angels.skills.evaluate.policy import LEARN_ORDER_V1 as P
from talent_angels.skills.roadmap.common import Graph, Json, id_suite, ref, resolve_side
from talent_angels.skills.roadmap.order import Item, order_items, step_of_reuse
from talent_angels.suites import SuiteRegistry

if TYPE_CHECKING:
    from talent_angels.skills.pathfind.gap import SkillGap

MAX_PER_STEP = 50
CREDENTIALS = ("entry_education", "work_experience", "job_training")


@dataclass
class SuitePlan:
    """One taxonomy's share of a roadmap."""

    status: str  # ok | ambiguous | not_found | no_order_data | error:<Type>
    start: Json | None = None
    target: Json | None = None
    candidates: list[Json] = field(default_factory=list)
    items: list[Item] = field(default_factory=list)  # in learning order
    credentials: Json | None = None
    job_zone: Json | None = None
    warnings: list[str] = field(default_factory=list)


def _gap(graph: Graph, name: str, a: TaxonomyNode, b: TaxonomyNode) -> SkillGap:
    from talent_angels.skills.pathfind.gap import compute_gap  # pathfind imports the MCP edge

    schema = graph.suite.suite_schema

    def skills(node: TaxonomyNode) -> AgentResult:
        return connect(
            graph.suite,
            name,
            _node_ref(node, name),
            request=ConnectRequest(subject=node.label, rel_types=schema.skill_rel_types),
            confidence=None,
            locate_evidence=[],
            optional_rel_values=schema.optional_rel_values,
        )

    return compute_gap(skills(a), skills(b), optional_rel_values=schema.optional_rel_values)


def _esco(graph: Graph, name: str, a: TaxonomyNode, b: TaxonomyNode, plan: SuitePlan) -> None:
    gap = _gap(graph, name, a, b)
    if gap.gap_too_large:
        plan.warnings.append("gap_too_large")
    ids = {n.id for n in gap.essential_missing}
    items: list[Item] = []
    links: list[tuple[str, str]] = []
    for n in gap.essential_missing:
        props = graph.props(n.id)  # skill_type and reuse_level live on the node's own call
        knowledge = props is not None and props.get("skill_type") == "knowledge"
        reuse = props.get("reuse_level") if props is not None else None
        why = "esco:knowledge" if knowledge else "esco:skill"
        known = reuse in P.reuse_order
        step = 1 if knowledge else step_of_reuse(reuse)
        items.append(
            Item(
                n.id, n.pref_label, n.kind, n.source_id, step,
                [why, f"esco:reuse:{reuse}" if known else "esco:reuse:unknown"],
            )
        )  # fmt: skip
        if props is not None:  # a stated link: this skill needs another skill of the roadmap
            links += [
                (n.id, e.to_id)
                for e in graph.neighbors(n.id).edges
                if e.type == "RELATED_TO"
                and e.properties.get("relation_type") == "essential"
                and e.from_id == n.id
                and e.to_id in ids
            ]
    plan.items, found = order_items(items, links)
    plan.warnings += found


def _jobtech(graph: Graph, name: str, a: TaxonomyNode, b: TaxonomyNode, plan: SuitePlan) -> None:
    gap = _gap(graph, name, a, b)
    skills = [*gap.essential_missing, *gap.optional_missing]
    items = [
        Item(n.id, n.pref_label, n.kind, n.source_id, P.default_step, ["jobtech:skill"])
        for n in skills
    ]
    plan.items, _ = order_items(items, [])  # JobTech states no order: every skill is Core


def _rated(graph: Graph, node: TaxonomyNode) -> dict[str, Any]:
    return {
        e.to_id: e.properties
        for e in graph.neighbors(node.id).edges
        if e.type in ("HAS_SKILL", "HAS_KNOWLEDGE") and e.from_id == node.id
    }


def _num(value: object) -> float | None:
    return float(value) if isinstance(value, int | float) and not isinstance(value, bool) else None


def _onet(graph: Graph, name: str, a: TaxonomyNode, b: TaxonomyNode, plan: SuitePlan) -> None:
    mine, wanted = _rated(graph, a), _rated(graph, b)
    kinds = {n.id: n for n in graph.neighbors(b.id).nodes}
    items: list[Item] = []
    suppressed = 0
    for sid in sorted(wanted):
        node, props = kinds.get(sid), wanted[sid]
        level, importance = _num(props.get("level")), _num(props.get("importance"))
        if node is None or level is None or importance is None:
            continue
        if props.get("recommend_suppress") == "Y":  # O*NET: too few raters to publish
            suppressed += 1
            continue
        before = _num(mine.get(sid, {}).get("level"))
        jump = round(level - (before or 0.0), P.level_digits)
        if importance < P.min_importance or jump < P.min_level_jump:
            continue
        frm = "n/a" if before is None else f"{before:.2f}"
        items.append(
            Item(
                sid, node.label, node.kind, node.source_id,
                1 if node.kind == "Knowledge" else 2,
                [
                    f"onet:{node.kind.casefold()}",
                    f"onet:level {frm}→{level:.2f}",
                    f"onet:importance {importance:.2f}",
                ],
                levels={
                    "from": None if before is None else round(before, P.level_digits),
                    "to": round(level, P.level_digits),
                    "jump": jump,
                    "importance": round(importance, P.level_digits),
                },
                jump=jump,
                importance=importance,
            )
        )  # fmt: skip
    plan.items, _ = order_items(items, [])
    if suppressed:
        plan.warnings.append(f"suppressed:{suppressed}")

    def zone(node: TaxonomyNode) -> Json | None:
        res = graph.neighbors(node.id)
        for e in res.edges:
            if e.type == "HAS_JOB_ZONE" and e.from_id == node.id:
                z = next((n for n in res.nodes if n.id == e.to_id), None)
                last = e.to_id.rsplit(":", 1)[-1]
                if z is not None:
                    return {**ref(z), "zone": int(last) if last.isdigit() else None}
        return None

    left, right = zone(a), zone(b)
    if left or right:
        plan.job_zone = {"from": left, "to": right}


def _bls(graph: Graph, name: str, a: TaxonomyNode, b: TaxonomyNode, plan: SuitePlan) -> None:
    props = graph.props(b.id) or {}
    plan.credentials = {"suite": name, **ref(b), **{k: props.get(k) for k in CREDENTIALS}}


_GAPS = {"esco": _esco, "jobtech": _jobtech, "onet": _onet, "bls": _bls}


def roadmap_suite(
    suite: Any, name: str, a: str, b: str, *, owners: tuple[str | None, str | None] = (None, None)
) -> SuitePlan:
    """One taxonomy's plan. ``a`` and ``b`` are names or node IDs; ``owners`` are the
    taxonomies their IDs belong to (an ID is never searched as text in another one)."""
    if name not in _GAPS:
        return SuitePlan("no_order_data")
    graph = Graph(suite)
    own_a, own_b = owners
    left = resolve_side(graph, name, a, own_a, "from")
    right = resolve_side(graph, name, b, own_b, "to")
    if "not_found" in (left.status, right.status):
        return SuitePlan("not_found")
    if "ambiguous" in (left.status, right.status):
        return SuitePlan("ambiguous", candidates=left.candidates + right.candidates)
    assert left.node is not None and right.node is not None
    plan = SuitePlan("ok", start=ref(left.node), target=ref(right.node))
    if left.node.id == right.node.id:
        plan.warnings.append("same_start_and_target")
    elif not left.node.kind == right.node.kind == "Occupation":
        plan.status = "no_order_data"
        plan.warnings.append(f"not_an_occupation:{left.node.kind}/{right.node.kind}")
    else:
        _GAPS[name](graph, name, left.node, right.node, plan)
    plan.warnings += graph.warnings
    return plan


def roadmap(
    registry: SuiteRegistry,
    a: str,
    b: str,
    *,
    suites: Iterable[str] | None = None,
    per_step: int = 10,
) -> Json:
    """The learning order from ``a`` to ``b`` in every selected taxonomy, merged and cited."""
    if not 1 <= per_step <= MAX_PER_STEP:
        raise ValueError(f"per_step must be 1..{MAX_PER_STEP}, got {per_step}")
    names = _select(registry, suites)
    owners = (id_suite(registry, a), id_suite(registry, b))
    plans: dict[str, SuitePlan] = {}
    for name in names:
        try:
            with registry.open(name) as runtime:
                plans[name] = roadmap_suite(runtime.suite, name, a, b, owners=owners)
        except Exception as exc:  # one broken taxonomy must not hide the others
            plans[name] = SuitePlan(f"error:{type(exc).__name__}")
    steps: list[Json] = []
    for number, title in enumerate(P.step_names, start=1):
        shown: list[Json] = []
        more = 0
        for name, plan in plans.items():
            here = [i for i in plan.items if i.step == number]
            shown += [i.to_dict(name) for i in here[:per_step]]
            more += max(0, len(here) - per_step)
        if shown or more:
            steps.append({"step": number, "name": title, "items": shown, "more": more})
    zones = [p.job_zone for p in plans.values() if p.job_zone]
    return {
        "policy": P.name,
        "meaning": P.meaning,
        "from": a,
        "to": b,
        "per_step": per_step,
        "suites": {
            name: {
                "status": p.status,
                "start": p.start,
                "target": p.target,
                "candidates": p.candidates,
                "items": len(p.items),
                "warnings": p.warnings,
            }
            for name, p in plans.items()
        },
        "steps": steps,
        "credentials": [p.credentials for p in plans.values() if p.credentials],
        "job_zone": zones[0] if zones else None,
        "warnings": [f"{n}:{w}" for n, p in plans.items() for w in p.warnings],
    }
