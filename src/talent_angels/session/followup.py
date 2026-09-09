"""Session follow-ups resolved in code (binding + last result), not by a new search."""

from __future__ import annotations

import re

from talent_angels.contracts import AgentResult, NodeRef

_BARE_YES = re.compile(r"^\s*yes\s*[.!]?\s*$", re.IGNORECASE)

_EXPAND = re.compile(
    r"(complete|full|entire|whole)\s+(list|set)|"
    r"list(\s+down)?(\s+the)?\s+(complete|full|all)|"
    r"show(\s+me)?(\s+the)?\s+(complete|full|all|rest)|"
    r"list\s+them(\s+all)?|"
    r"the\s+rest|"
    r"all\s+(the\s+)?skills|"
    r"deeper\s+search|"
    r"tell\s+me\s+more",
    re.IGNORECASE,
)

CONNECT_LIST_CAP = 60

_SKILL_MENTION = re.compile(
    r"(?:tell\s+me\s+about|what(?:'s|s|\s+is)?)\s+skill\s*#?\s*(\d+)|"
    r"\bskill\s*#?\s*(\d+)\b",
    re.IGNORECASE,
)


def is_bare_yes(text: str) -> bool:
    return bool(_BARE_YES.match(text.strip()))


_CATALOGUE_OBJECT = re.compile(
    r"\b(jobs?|occupations?|roles?|titles?|careers?)\b",
    re.IGNORECASE,
)


def is_expand_list(text: str) -> bool:
    """True when the user wants the rest of the last Connect list."""
    stripped = text.strip()
    if is_bare_yes(stripped):
        return False
    # "list all jobs" is a catalogue ask, not "expand the last skill list".
    if _CATALOGUE_OBJECT.search(stripped) and not re.search(r"\bskills?\b", stripped, re.I):
        return False
    return bool(_EXPAND.search(stripped))


def parse_skill_mention(text: str) -> int | None:
    """1-based skill index from 'skill 11' / 'tell me about skill 11', else None."""
    match = _SKILL_MENTION.search(text.strip())
    if match is None:
        return None
    raw = match.group(1) or match.group(2)
    return int(raw)


def skill_index_by_label(result: AgentResult, text: str) -> int | None:
    """If the user names a skill already on the last Connect list, return its 1-based index."""
    cleaned = re.sub(r"[?.!]+$", "", text.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(
        r"^(tell\s+me\s+about|what\s+is|what\s+about)\s+",
        "",
        cleaned,
        flags=re.IGNORECASE,
    ).strip()
    if len(cleaned) < 4:
        return None
    key = cleaned.casefold()
    for index, node in enumerate(result.nodes[1:], start=1):
        if node.pref_label.casefold() == key:
            return index
    return None


def skill_from_connect(result: AgentResult, number: int) -> NodeRef:
    """Return last_result.nodes[number] where nodes[0] is the occupation."""
    neighbors = result.nodes[1:]
    if number < 1 or number > len(neighbors):
        raise ValueError(f"skill {number} is not in 1..{len(neighbors)}")
    return neighbors[number - 1]


def can_expand_connect(result: AgentResult | None) -> bool:
    if result is None or result.capability != "connect":
        return False
    return len(result.nodes) > 1


def can_expand_locate(result: AgentResult | None) -> bool:
    """True when the last turn was an ambiguous occupation picker we can widen."""
    if result is None or result.capability != "locate":
        return False
    return "ambiguous" in result.warnings and bool(result.nodes)


def render_connect_list(result: AgentResult, *, cap: int = CONNECT_LIST_CAP) -> str:
    """Deterministic full-ish neighbor list. Not a model-authored curriculum."""
    if not result.nodes:
        return "I don't have a stored skill list for this session yet. Name a job title first."
    center = result.nodes[0]
    neighbors = result.nodes[1:]
    shown = neighbors[:cap]
    omitted = len(neighbors) - len(shown)
    lines = [
        f"**{center.pref_label}** — {len(neighbors)} skills on the map:",
        "",
    ]
    relations = {}
    for edge in result.edges:
        rel = edge.properties.get("relation_type")
        if not rel:
            continue
        if edge.target_node_id != center.id:
            relations[edge.target_node_id] = str(rel)
        if edge.source_node_id != center.id:
            relations[edge.source_node_id] = str(rel)
    for index, node in enumerate(shown, start=1):
        tag = relations.get(node.id)
        suffix = f" ({tag})" if tag else ""
        lines.append(f"{index}. {node.pref_label}{suffix}")
    lines.append("")
    if omitted:
        lines.append(f"{omitted} more are in query details.")
    lines.append("These are graph neighbors, not a study plan.")
    return "\n".join(lines)
