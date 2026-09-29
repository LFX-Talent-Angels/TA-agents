"""Tests for talent_angels.session.followup — relation tags and the expand list."""

from __future__ import annotations

from talent_angels.contracts import AgentResult, EdgeRef, NodeRef
from talent_angels.session.followup import relation_tags, render_connect_list


def _occ(suite: str = "esco") -> NodeRef:
    return NodeRef(
        id=f"{suite}:occupation:1",
        suite=suite,
        source=suite,
        source_id="1",
        kind="Occupation",
        pref_label="software developer",
    )


def _skill(node_id: str, label: str, suite: str = "esco") -> NodeRef:
    return NodeRef(
        id=node_id, suite=suite, source=suite, source_id=node_id, kind="Skill", pref_label=label
    )


def test_relation_tags_reads_essential_and_optional() -> None:
    center = _occ()
    edges = [
        EdgeRef(
            type="HAS_SKILL",
            suite="esco",
            source_node_id=center.id,
            target_node_id="esco:skill:1",
            properties={"relation_type": "essential"},
        ),
        EdgeRef(
            type="HAS_SKILL",
            suite="esco",
            source_node_id="esco:skill:2",
            target_node_id=center.id,
            properties={"relation_type": "optional"},
        ),
    ]
    tags = relation_tags(edges, center.id)
    assert tags == {"esco:skill:1": "essential", "esco:skill:2": "optional"}


def test_relation_tags_falls_back_to_tool_for_untagged_software_edges() -> None:
    """GAP C: an edge with no relation_type is not 'unknown' — USES_SOFTWARE
    never carries one, so it gets a real tag instead of disappearing."""
    center = _occ("onet")
    edges = [
        EdgeRef(
            type="USES_SOFTWARE",
            suite="onet",
            source_node_id=center.id,
            target_node_id="onet:tool:1",
            properties={},
        )
    ]
    assert relation_tags(edges, center.id) == {"onet:tool:1": "tool"}


def test_relation_tags_leaves_genuinely_unknown_edges_untagged() -> None:
    """An edge type with no fallback mapping stays untagged — no invented signal."""
    center = _occ()
    edges = [
        EdgeRef(
            type="SOME_OTHER_EDGE",
            suite="esco",
            source_node_id=center.id,
            target_node_id="esco:node:1",
            properties={},
        )
    ]
    assert relation_tags(edges, center.id) == {}


def test_render_connect_list_shows_tool_tag_alongside_essential_and_optional() -> None:
    center = _occ("onet")
    essential = _skill("onet:skill:reading", "reading comprehension", "onet")
    tool = _skill("onet:tool:js", "JavaScript", "onet")
    result = AgentResult(
        capability="connect",
        suite="onet",
        nodes=[center, essential, tool],
        edges=[
            EdgeRef(
                type="HAS_SKILL",
                suite="onet",
                source_node_id=center.id,
                target_node_id=essential.id,
                properties={"relation_type": "essential"},
            ),
            EdgeRef(
                type="USES_SOFTWARE",
                suite="onet",
                source_node_id=center.id,
                target_node_id=tool.id,
                properties={},
            ),
        ],
    )
    text = render_connect_list(result)
    assert "reading comprehension (essential)" in text
    assert "JavaScript (tool)" in text


def test_render_connect_list_reports_no_stored_list_before_any_connect() -> None:
    empty = AgentResult(capability="connect", suite="esco", nodes=[], edges=[])
    text = render_connect_list(empty)
    assert "Name a job title first" in text
