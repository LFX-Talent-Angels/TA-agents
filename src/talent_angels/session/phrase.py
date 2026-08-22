"""LLM phrasing for the TUI. Rails (ids, numbers, graph facts) stay in code.

Uses a fact card, never the growing transcript. Falls back to copy when the
client is a stub or the call fails.
"""

from __future__ import annotations

import re

from talent_angels.contracts import AgentResult, NodeRef
from talent_angels.llm import LLMClient, Message

_CHAT_SYSTEM = """You are Talent Angels, a concise assistant in a terminal.
Warm and useful. You look up occupations and skills on a taxonomy map.
Do not call yourself an ESCO desk or any other taxonomy's chatbot.
Do not invent job titles or skills. Do not give personal career advice.
Keep replies to 2–4 short sentences unless listing facts you were given."""

_MAP_SYSTEM = """You are Talent Angels. Phrase the FACT CARD for a terminal user.
Rules:
- Cite only titles and skills written in the card. Do not invent any.
- These are map titles, not a guess about a person. Never say "the person is".
- Do not brand yourself as a single taxonomy. You may mention the suite once.
- 2–4 short sentences. Offer a useful next step (skills of this title, or another search).
- Do not number options. Do not pick rank 1.
- Do not suggest related job titles that are not in the FACT CARD."""


def uses_chat_phrasing(client: LLMClient | None) -> bool:
    if client is None:
        return False
    provider = str(getattr(client, "provider", "none") or "none").lower()
    return provider not in {"none", "stub", "test"}


def phrase_chat(
    client: LLMClient | None,
    *,
    user_text: str,
    fallback: str,
    hint: str,
    mode: str = "chat",
) -> str:
    """Phrase a non-map line (greeting, help intro, advice, miss)."""
    if not uses_chat_phrasing(client):
        return fallback
    assert client is not None
    messages = [
        Message(role="system", content=_CHAT_SYSTEM + "\n" + hint),
        Message(role="user", content=user_text),
    ]
    try:
        result = client.complete(messages)
    except (RuntimeError, OSError, ValueError):
        return fallback
    text = (result.text or "").strip()
    if not text:
        return fallback
    if mode == "intro" and _looks_like_numbered_list(text):
        return fallback
    if mode == "miss" and _looks_like_invented_miss(text):
        return fallback
    return text


def phrase_map(
    client: LLMClient | None,
    *,
    question: str,
    result: AgentResult,
    fallback: str,
    card: str,
) -> str:
    if not uses_chat_phrasing(client):
        return fallback
    assert client is not None
    messages = [
        Message(role="system", content=_MAP_SYSTEM),
        Message(role="user", content=f"User: {question}\n\nFACT CARD:\n{card}"),
    ]
    try:
        llm_result = client.complete(messages)
    except (RuntimeError, OSError, ValueError):
        return fallback
    text = (llm_result.text or "").strip()
    if not text:
        return fallback
    if _looks_like_numbered_list(text):
        return fallback
    if result.capability == "locate" and _has_extra_job_title(text, result):
        return fallback
    return text


def locate_card(result: AgentResult) -> str:
    if not result.nodes:
        return f"warnings: {', '.join(result.warnings) or 'not_found'}"
    top = result.nodes[0]
    conf = f"{result.confidence:.0%}" if result.confidence is not None else "unknown"
    return (
        f"unique occupation title: {top.pref_label}\n"
        f"kind: {top.kind}\n"
        f"confidence: {conf}\n"
        "next: user may ask for essential or optional skills"
    )


def connect_card(result: AgentResult, *, shown: int = 5) -> str:
    if not result.nodes:
        return "no neighbors"
    center = result.nodes[0]
    skills = [node.pref_label for node in result.nodes[1 : shown + 1]]
    extra = max(0, len(result.nodes) - 1 - shown)
    lines = [
        f"occupation: {center.pref_label}",
        f"connections: {len(result.edges)}",
        "shown skills: " + "; ".join(skills) if skills else "shown skills: none",
        "tag each skill essential or optional when known",
    ]
    if extra:
        lines.append(f"{extra} more skills live in query details, not in this reply")
    lines.append("this is the map, not a study plan")
    return "\n".join(lines)


def skill_card(
    skill: NodeRef,
    *,
    number: int,
    occupation: str,
    neighbors: list[NodeRef],
    shown: int = 5,
) -> str:
    occupations = [node.pref_label for node in neighbors if node.kind.casefold() == "occupation"]
    pool = occupations or [node.pref_label for node in neighbors]
    extra = max(0, len(pool) - shown)
    lines = [
        f"Skill {number} on {occupation}: {skill.pref_label} (id {skill.id})",
        f"kind: {skill.kind}",
        "shown occupations: " + ("; ".join(pool[:shown]) if pool[:shown] else "none"),
    ]
    if extra:
        lines.append(f"{extra} more neighbors live in query details")
    return "\n".join(lines)


def ambiguous_intro_card(question: str, labels: list[str], *, omitted: int) -> str:
    lines = [
        f'search: "{question}"',
        f"matches shown: {len(labels)}",
        "A numbered list will be appended in code. Write only a one-sentence "
        "intro asking the user to pick. Do not list titles.",
    ]
    if omitted:
        lines.append(f"{omitted} further matches omitted")
    return "\n".join(lines)


_SAFE_BOLDS = {
    "talent angels",
    "essential skills",
    "optional skills",
    "query details",
    "the map",
    "job title",
    "occupation title",
}


def _has_extra_job_title(text: str, result: AgentResult) -> bool:
    if len(result.nodes) != 1:
        return False
    allowed = result.nodes[0].pref_label.casefold()
    for bold in re.findall(r"\*\*(.+?)\*\*", text):
        key = bold.casefold()
        if key == allowed or key in _SAFE_BOLDS or "skill" in key:
            continue
        if len(key.split()) >= 2:
            return True
    return False


def _looks_like_numbered_list(text: str) -> bool:
    return bool(re.search(r"(?m)^\s*\d+\.\s+\S", text)) or "```" in text


def _looks_like_invented_miss(text: str) -> bool:
    lowered = text.casefold()
    if any(marker in lowered for marker in ("linked to", "appears as", "occupations such as")):
        return True
    if re.search(r"(?m)^\s*[-*]\s+\S", text):
        return True
    return False
