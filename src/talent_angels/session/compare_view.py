"""Deterministic text for a compare result: what two occupations share and don't."""

from __future__ import annotations

from collections.abc import Sequence

from talent_angels.contracts import AgentResult, NodeRef
from talent_angels.session.followup import relation_tags
from talent_angels.session.i18n import plural, t
from talent_angels.skills.connect.compare import skill_overlap

#: Names shown per group before "+N more".
COMPARE_PREVIEW = 8

_TAG_ORDER = {"essential": 0, "optional": 1, "tool": 3}


def _line(title: str, nodes: Sequence[NodeRef], tags: dict[str, str]) -> str:
    if not nodes:
        return f"**{title}** (0): {t('compare_none')}"
    # Essential first, then optional, then the rest; graph order within a tag.
    ranked = sorted(nodes, key=lambda node: _TAG_ORDER.get(tags.get(node.id, ""), 2))
    shown = ", ".join(node.pref_label for node in ranked[:COMPARE_PREVIEW])
    extra = len(ranked) - COMPARE_PREVIEW
    more = f" {t('compare_more', count=extra)}" if extra > 0 else ""
    return f"**{title}** ({len(nodes)}): {shown}{more}"


def render_compare(result: AgentResult) -> str:
    overlap = skill_overlap(result)
    a, b = overlap.a, overlap.b
    tags_a = relation_tags(result.edges, a.id)
    tags_b = relation_tags(result.edges, b.id)
    tags = {**tags_b, **tags_a}
    tools = sum(1 for node in overlap.shared if tags.get(node.id) == "tool")
    skills = len(overlap.shared) - tools
    shared = f"{skills} {plural(skills, 'shared_skill', 'shared_skills_noun')}"
    if tools:
        shared += f" {t('and')} {tools} {plural(tools, 'shared_tool', 'shared_tools_noun')}"
    return "\n".join(
        [
            t("compare_head", first=a.pref_label, second=b.pref_label, shared=shared),
            "",
            # A blank line between groups: single newlines are soft wraps in
            # Markdown, and ran the three groups into one paragraph.
            _line(t("compare_shared"), overlap.shared, tags_a),
            "",
            _line(t("compare_only", title=a.pref_label), overlap.only_a, tags_a),
            "",
            _line(t("compare_only", title=b.pref_label), overlap.only_b, tags_b),
            "",
            t("compare_foot"),
        ]
    )
