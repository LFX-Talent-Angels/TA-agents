"""Deterministic Locate rerank and ISCO grouping. No LLM pick. Not query-specific."""

from __future__ import annotations

import re

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


#: Highest lexical tier that may be auto-selected (3 = exact alternative label).
_MAX_AUTO_SELECT_TIER = 3


def _plain(query: str) -> str:
    """The words of a search, without end punctuation or a leading article."""
    text = re.sub(r"[\s?.!,;:]+$", "", query.strip())
    return re.sub(r"^(?:an?|the)\s+", "", text, flags=re.IGNORECASE)


def _word_end(query: str) -> str:
    """A whole-word end for queries too short to be a prefix of a real title."""
    text = query.strip()
    return r"(?![a-z0-9])" if text.isalpha() and len(text) <= 4 else ""


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
    elif q and re.search(rf"(?<![a-z0-9]){re.escape(q)}{_word_end(query)}", pref):
        # From the start of a word: "it manager" is not inside "credit manager".
        # A short word or an acronym must be a whole word: "swe" is not "sweep".
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
    suite: ConnectableSuite,
    node: NodeRef,
    *,
    suite_name: str,
    group_rel_type: str | None = "CLASSIFIED_UNDER",
    group_node_kinds: frozenset[str] = frozenset({"ISCOGroup", "isco group"}),
) -> NodeRef | None:
    if group_rel_type is None:
        return None
    if node.kind.casefold() != "occupation":
        return None
    hop = suite.get_neighbors(node.id, rel_types=[group_rel_type])
    by_id = {item.id: item for item in hop.nodes}
    _group_kinds_lower = frozenset(k.casefold() for k in group_node_kinds)
    for edge in hop.edges:
        edge_type, source, target = _edge_ends(edge)
        if edge_type != group_rel_type or source != node.id:
            continue
        parent = by_id.get(target)
        if parent is None:
            continue
        if str(getattr(parent, "kind", "")).casefold() not in _group_kinds_lower:
            # Still use it as a heading if the suite marked it as the group target.
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
    group_rel_type: str | None = "CLASSIFIED_UNDER",
    group_node_kinds: frozenset[str] = frozenset({"ISCOGroup", "isco group"}),
) -> AgentResult:
    """Reorder Locate hits; attach group edges; mark multi-hit as ambiguous."""
    if (
        len(result.nodes) == 1
        and lexical_rank(_plain(query), result.nodes[0])[0] > _MAX_AUTO_SELECT_TIER
    ):
        # One hit found only inside an alias ("qa tester" -> "localiser") or by
        # meaning is offered for the user to confirm, never selected for them.
        if "ambiguous" not in result.warnings:
            return result.model_copy(update={"warnings": [*result.warnings, "ambiguous"]})
        return result
    if len(result.nodes) <= 1:
        return result

    parents: dict[str, NodeRef] = {}
    for node in result.nodes:
        parent = classified_under_parent(
            suite,
            node,
            suite_name=suite_name,
            group_rel_type=group_rel_type,
            group_node_kinds=group_node_kinds,
        )
        if parent is not None:
            parents[node.id] = parent

    # Hits with no lexical match came from the suite's meaning search; its
    # order is the relevance signal, so it replaces label length for them.
    position = {node.id: index for index, node in enumerate(result.nodes)}

    def sort_key(node: NodeRef) -> tuple[int, int, str]:
        tier, length, node_id = lexical_rank(query, node)
        return (tier, position[node.id] if tier == 5 else length, node_id)

    def group_id(node: NodeRef) -> str:
        parent = parents.get(node.id)
        return parent.id if parent is not None else ""

    def group_score(gid: str) -> tuple[int, int, str, str]:
        members = [node for node in result.nodes if group_id(node) == gid]
        best = min(sort_key(node) for node in members)
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
        key=lambda node: (group_order[group_id(node)], sort_key(node)),
    )
    top_tier = lexical_rank(query, ordered[0])[0]
    # The runner-up is the best of the rest, not the next node in group order:
    # a weak member of the winning group must not hide an equal hit elsewhere.
    second_tier = min(lexical_rank(query, node)[0] for node in ordered[1:])
    # A hit that names the query in its preferred label (tiers 0-2) or as an
    # exact alias (3), and beats the next hit's tier, is unique enough — extra
    # full-text noise is not a picker. An alias *substring* (4) or a hit with no
    # lexical match at all (5) is never auto-selected: "nurse" must not resolve
    # to "chief executive officer" because one alias is "senior nurse manager".
    # A truncated pool may have cut an equal hit, so only an exact title wins it.
    # The suite flagged an alias its meaning search disagrees with ("AI
    # engineer" as an alias of an insemination technician): the user picks.
    truncated = "truncated" in result.warnings
    unique_enough = (
        top_tier <= _MAX_AUTO_SELECT_TIER
        and top_tier < second_tier
        and not (truncated and top_tier > 0)
        and "alias_unconfirmed" not in result.warnings
    )
    if unique_enough:
        extra = len(ordered) - 1
        winner = ordered[0]
        parent = parents.get(winner.id)
        edges = (
            [
                EdgeRef(
                    type=group_rel_type or "CLASSIFIED_UNDER",
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
            type=group_rel_type or "CLASSIFIED_UNDER",
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


_ONET_CODE = re.compile(r"^onet:occupation:(\d{2}-\d{4}\.\d{2})$")


def code_facts(result: AgentResult, node: NodeRef) -> list[str]:
    """Codes and group of ``node`` that the graph result itself carries.

    So a card can answer "what ISCO group is a chef in?" from data rather than
    the model's memory: the ISCO group comes from the node's CLASSIFIED_UNDER
    edge (ids are ``esco:isco:<code>``), the O*NET-SOC code from its id.
    """
    facts: list[str] = []
    match = _ONET_CODE.match(node.id)
    if match:
        facts.append(f"O*NET-SOC code: {match.group(1)}")
    for edge in result.edges:
        label = edge.properties.get("group_label")
        if edge.source_node_id != node.id or not label:
            continue
        code = edge.target_node_id.rsplit(":", 1)[-1]
        if code.isdigit():
            facts.append(f"ISCO-08 group: {code} {label}")
        else:
            facts.append(f"group: {label}")
        break
    return facts
