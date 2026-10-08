"""Deterministic Locate rerank and ISCO grouping. No LLM pick. Not query-specific."""

from __future__ import annotations

from talent_angels.contracts import AgentResult, EdgeRef, NodeRef
from talent_angels.skills.connect.reveal import ConnectableSuite
from talent_angels.skills.locate.resolve import _node_ref


def _near_pref_label(query: str, node: NodeRef) -> bool:
    q = query.casefold().strip()
    pref = node.pref_label.casefold().strip()
    if pref == q:
        return True
    if pref.endswith("s") and pref[:-1] == q:
        return True
    if q.endswith("s") and q[:-1] == pref:
        return True
    return False


def lexical_rank(query: str, node: NodeRef) -> tuple[int, int, str]:
    """Lower is better: exact pref, pref token, pref substring, exact alt, alt substring."""
    q = query.casefold().strip()
    pref = node.pref_label.casefold()
    alts = [alt.casefold() for alt in node.alt_labels]
    tokens = pref.replace("/", " ").replace("-", " ").split()
    if pref == q:
        tier = 0
    elif q in tokens:
        tier = 1
    elif q and q in pref:
        tier = 2
    elif q in alts:
        tier = 3
    elif q and any(q in alt for alt in alts):
        tier = 4
    else:
        tier = 5
    return (tier, len(pref), node.id)


def _edge_ends(edge: object) -> tuple[str, str, str]:
    edge_type = str(getattr(edge, "type", "") or "")
    source = str(getattr(edge, "from_id", None) or getattr(edge, "source_node_id", "") or "")
    target = str(getattr(edge, "to_id", None) or getattr(edge, "target_node_id", "") or "")
    return edge_type, source, target


def classified_under_parent(
    suite: ConnectableSuite, node: NodeRef, *, suite_name: str
) -> NodeRef | None:
    if node.kind.casefold() != "occupation":
        return None
    hop = suite.get_neighbors(node.id, rel_types=["CLASSIFIED_UNDER"])
    by_id = {item.id: item for item in hop.nodes}
    for edge in hop.edges:
        edge_type, source, target = _edge_ends(edge)
        if edge_type != "CLASSIFIED_UNDER" or source != node.id:
            continue
        parent = by_id.get(target)
        if parent is None:
            continue
        if str(getattr(parent, "kind", "")).casefold() not in {"iscogroup", "isco group"}:
            # Still use it as a heading if the suite marked it as the CLASSIFIED_UNDER target.
            pass
        label = getattr(parent, "pref_label", None) or getattr(parent, "label", None)
        if isinstance(parent, NodeRef):
            return parent
        return _node_ref(parent, suite_name) if label else None
    return None


def group_and_sort_locate(
    suite: ConnectableSuite,
    result: AgentResult,
    query: str,
    *,
    suite_name: str,
) -> AgentResult:
    """Reorder Locate hits; attach CLASSIFIED_UNDER edges; mark multi-hit as ambiguous."""
    if len(result.nodes) <= 1:
        return result

    parents: dict[str, NodeRef] = {}
    for node in result.nodes:
        parent = classified_under_parent(suite, node, suite_name=suite_name)
        if parent is not None:
            parents[node.id] = parent

    def group_id(node: NodeRef) -> str:
        parent = parents.get(node.id)
        return parent.id if parent is not None else ""

    def group_score(gid: str) -> tuple[int, int, str, str]:
        members = [node for node in result.nodes if group_id(node) == gid]
        best = min(lexical_rank(query, node) for node in members)
        label = (
            parents[members[0].id].pref_label.casefold()
            if gid and members[0].id in parents
            else "\uffff"
        )
        return (*best[:2], label, gid)

    group_order = {
        gid: i for i, gid in enumerate(sorted({group_id(n) for n in result.nodes}, key=group_score))
    }
    ordered = sorted(
        result.nodes,
        key=lambda node: (group_order[group_id(node)], lexical_rank(query, node)),
    )
    top_tier = lexical_rank(query, ordered[0])[0]
    second_tier = lexical_rank(query, ordered[1])[0]
    # Exact preferred label (0) or exact alias (3) that beats the next hit
    # is unique enough — extra full-text noise is not a picker.
    unique_enough = (
        top_tier in {0, 3} or _near_pref_label(query, ordered[0])
    ) and top_tier < second_tier
    if unique_enough:
        extra = len(ordered) - 1
        winner = ordered[0]
        parent = parents.get(winner.id)
        edges = (
            [
                EdgeRef(
                    type="CLASSIFIED_UNDER",
                    suite=suite_name,
                    source_node_id=winner.id,
                    target_node_id=parent.id,
                    properties={"group_label": parent.pref_label},
                )
            ]
            if parent is not None
            else []
        )
        warnings = [item for item in result.warnings if item != "ambiguous"]
        if extra:
            warnings.append(f"also_matched:{extra}")
        confidence = 0.95 if top_tier == 0 else 0.90
        return result.model_copy(
            update={
                "nodes": [winner],
                "edges": edges,
                "warnings": warnings,
                "confidence": confidence,
            }
        )

    edges = [
        EdgeRef(
            type="CLASSIFIED_UNDER",
            suite=suite_name,
            source_node_id=node.id,
            target_node_id=parents[node.id].id,
            properties={"group_label": parents[node.id].pref_label},
        )
        for node in ordered
        if node.id in parents
    ]
    warnings = list(result.warnings)
    if "ambiguous" not in warnings:
        warnings.append("ambiguous")
    return result.model_copy(update={"nodes": ordered, "edges": edges, "warnings": warnings})
