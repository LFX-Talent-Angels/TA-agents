"""Deterministic text for a compare result: what two occupations share and don't."""

from __future__ import annotations

from collections.abc import Sequence

from talent_angels.contracts import AgentResult, NodeRef
from talent_angels.session.followup import relation_tags
from talent_angels.skills.connect.compare import skill_overlap

#: Names shown per group before "+N more".
COMPARE_PREVIEW = 8

_TAG_ORDER = {"essential": 0, "optional": 1, "tool": 3}


def _line(title: str, nodes: Sequence[NodeRef], tags: dict[str, str]) -> str:
    if not nodes:
        return f"**{title}** (0): none"
    # Essential first, then optional, then the rest; graph order within a tag.
    ranked = sorted(nodes, key=lambda node: _TAG_ORDER.get(tags.get(node.id, ""), 2))
    shown = ", ".join(node.pref_label for node in ranked[:COMPARE_PREVIEW])
    extra = len(ranked) - COMPARE_PREVIEW
    more = f" (+{extra} more)" if extra > 0 else ""
    return f"**{title}** ({len(nodes)}): {shown}{more}"


def render_compare(result: AgentResult) -> str:
    overlap = skill_overlap(result)
    a, b = overlap.a, overlap.b
    tags_a = relation_tags(result.edges, a.id)
    tags_b = relation_tags(result.edges, b.id)
    shared = len(overlap.shared)
    noun = "skill" if shared == 1 else "skills"
    return "\n".join(
        [
            f"**{a.pref_label}** vs **{b.pref_label}** — {shared} shared {noun}.",
            "",
            _line("Shared", overlap.shared, tags_a),
            _line(f"Only {a.pref_label}", overlap.only_a, tags_a),
            _line(f"Only {b.pref_label}", overlap.only_b, tags_b),
            "",
            "These are graph neighbours, not a recommendation.",
        ]
    )
