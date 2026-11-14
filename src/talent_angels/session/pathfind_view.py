"""Deterministic text for a pathfind result: the skill gap, routes, source.

Essential/optional/software are not separate AgentResult fields (ADR-0008:
no new contract type beyond ``routes``) — reconstructed here from the
target's own edges, the same way ``synthesize._connect_line`` already
splits software from skills. A node in ``result.nodes[2:]`` with no edge
to the target is a route-bridge node, not part of the gap.
"""

from __future__ import annotations

from talent_angels.contracts import AgentResult, NodeRef, Route
from talent_angels.session.i18n import t
from talent_angels.skills.locate.rank import published_facts

#: Names shown per gap bucket before "+N more".
GAP_PREVIEW = 5
#: Routes shown before "+N more routes".
ROUTES_PREVIEW = 3
#: find_routes' own default, never re-tuned (ADR-0008 decision 4).
DEFAULT_MAX_DEPTH = 4


def _gap_buckets(result: AgentResult) -> tuple[list[NodeRef], list[NodeRef], list[NodeRef]]:
    """Essential, optional and software-missing nodes, target's own order."""
    target = result.nodes[1]
    relation: dict[str, tuple[str, object]] = {}
    for edge in result.edges:
        other = None
        if edge.source_node_id == target.id:
            other = edge.target_node_id
        elif edge.target_node_id == target.id:
            other = edge.source_node_id
        if other is not None:
            relation[other] = (edge.type, edge.properties.get("relation_type"))

    essential: list[NodeRef] = []
    optional: list[NodeRef] = []
    software: list[NodeRef] = []
    for node in result.nodes[2:]:
        found = relation.get(node.id)
        if found is None:
            continue  # a route-bridge node, not part of the gap
        edge_type, rel = found
        if edge_type == "USES_SOFTWARE":
            software.append(node)
        elif rel in (None, "essential"):
            essential.append(node)
        else:
            optional.append(node)
    return essential, optional, software


def _gap_line(title: str, nodes: list[NodeRef], *, empty_key: str) -> str:
    if not nodes:
        return f"**{title}** (0): {t(empty_key)}"
    shown = ", ".join(node.pref_label for node in nodes[:GAP_PREVIEW])
    extra = len(nodes) - GAP_PREVIEW
    more = f" {t('pathfind_more', count=extra)}" if extra > 0 else ""
    return f"**{title}** ({len(nodes)}): {shown}{more}"


def _route_line(number: int, route: Route, labels: dict[str, str]) -> str:
    names = [labels.get(node_id, node_id) for node_id in route.node_ids]
    if len(names) > 2:
        names = [names[0], *(f"*{name}*" for name in names[1:-1]), names[-1]]
    return t("pathfind_route", number=number, chain=" → ".join(names))


def _render_bls_facts(start: NodeRef, target: NodeRef) -> str:
    """BLS's own skill taxonomy is ~17 broad skills shared by every
    occupation (confirmed live, PF-6): the essential/optional gap and
    routes are always trivially empty, never a real signal. BLS's actual
    distinguishing data — wage, outlook, education — is shown instead.
    """
    lines = [t("pathfind_head", start=start.pref_label, target=target.pref_label), ""]
    for node in (start, target):
        facts = published_facts(node)
        lines.append(f"**{node.pref_label}**")
        if facts:
            lines.extend(f"- {fact}" for fact in facts)
        else:
            lines.append(f"- {t('pathfind_no_facts')}")
        lines.append("")
    return "\n".join(lines).strip()


def render_pathfind(result: AgentResult) -> str:
    start, target = result.nodes[0], result.nodes[1]
    if result.suite == "bls":
        return _render_bls_facts(start, target)
    essential, optional, software = _gap_buckets(result)
    labels = {node.id: node.pref_label for node in result.nodes}

    lines = [
        t("pathfind_head", start=start.pref_label, target=target.pref_label),
        "",
        _gap_line(t("pathfind_essential"), essential, empty_key="pathfind_gap_none"),
        _gap_line(t("pathfind_optional"), optional, empty_key="pathfind_gap_none"),
        _gap_line(t("pathfind_software"), software, empty_key="pathfind_gap_none"),
    ]
    if "gap_too_large" in result.warnings:
        lines += ["", t("pathfind_full_retrain")]

    lines.append("")
    if result.routes:
        shown_routes = result.routes[:ROUTES_PREVIEW]
        lines += [_route_line(i, route, labels) for i, route in enumerate(shown_routes, start=1)]
        extra = len(result.routes) - len(shown_routes)
        if extra > 0:
            lines.append(t("pathfind_more_routes", count=extra))
    elif "no_path" in result.warnings:
        lines.append(t("pathfind_no_route", steps=DEFAULT_MAX_DEPTH))

    return "\n".join(lines).strip()
