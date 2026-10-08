"""LLM phrasing for the TUI. Rails (ids, numbers, graph facts) stay in code.

Uses a fact card, never the growing transcript. Falls back to copy when the
client is a stub or the call fails.
"""

from __future__ import annotations

import re

from talent_angels.assistant.answer import CONNECT_PREVIEW_CAP
from talent_angels.assistant.llm_call import measure_complete
from talent_angels.assistant.prose import prose_only
from talent_angels.contracts import AgentResult, NodeRef
from talent_angels.env import episode_retriever
from talent_angels.llm import LLMClient, Message
from talent_angels.llm.protocol import uses_chat_phrasing
from talent_angels.memory.agent_notes import notes_prefix
from talent_angels.memory.profile import profile_prefix
from talent_angels.memory.retrieval import recall_prefix
from talent_angels.session.followup import relation_tags
from talent_angels.skills.locate.rank import code_facts

_CHAT_SYSTEM = """You are LFX Talent Angels, a concise assistant in a terminal.
Warm and useful. You look up occupations and skills on a taxonomy map.
Do not call yourself an ESCO desk or any other taxonomy's chatbot.
Do not invent job titles or skills. Do not give personal career advice.
Keep replies to 2–4 short sentences unless listing facts you were given.
Write in the same language as the user's message."""

_MAP_SYSTEM = """You are LFX Talent Angels. Phrase the FACT CARD for a terminal user.
Rules:
- First reply to the user's own words in one sentence, then the facts; 2-4
  sentences in total. If they only named a title, just describe it. Never
  assume a goal or a wish the user did not state.
- Cite only titles, skills, and description written in the card. Do not invent any.
- Do not invent people, names, demand, pay, outlook, or study advice. Say nothing
  about the user unless the profile block above states it.
- The profile may already hold what the user says in this message. Never claim
  something was noted or said earlier unless a past-turns block shows it.
- Write plain sentences only: no tables, lists, headings, or code. The app shows
  the full list itself; name at most three skills as examples.
- If a description is on the card, paraphrase it in 1-2 sentences. Do not add duties.
- Do not list skills unless they are on the card. Locate cards have no skills.
- Never write the product name (not "LFX", not "Talent Angels").
- These are map titles, not a guess about a person. Never say "the person is".
- Do not number options. Do not pick rank 1.
- Do not suggest related job titles that are not in the FACT CARD.
- Do not repeat the id/confidence block; that is printed under your text.
- A fact the card does not hold (a code, a group, pay, outlook) is not known:
  say the card does not include it. Never state it from memory, and never
  correct yourself mid-answer.
- ESCO and O*NET are separate taxonomies with no official link here: never say
  a record in one maps to, equals, or is the equivalent of one in the other.
- Write in the same language as the user's message; keep job and skill titles
  exactly as the card writes them."""


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
        Message(
            role="system",
            # Recall sits between the memory blocks and the instructions: it is
            # context about the user, like the profile and the notes, and not
            # part of the brief. `""` unless a retriever is configured, so the
            # prompt is byte-identical to before for everyone not opted in.
            content=(
                profile_prefix()
                + notes_prefix()
                + recall_prefix(user_text, retriever=episode_retriever())
                + _CHAT_SYSTEM
                + "\n"
                + hint
            ),
        ),
        Message(role="user", content=user_text),
    ]
    try:
        result, _ = measure_complete(client, messages, stage="phrase")
    except RuntimeError:
        return fallback
    text = (result.text or "").strip()
    if not text:
        return fallback
    if mode == "intro" and (_looks_like_numbered_list(text) or _too_long_for_intro(text)):
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
        Message(
            role="system",
            content=(
                profile_prefix()
                + notes_prefix()
                + recall_prefix(question, retriever=episode_retriever())
                + _MAP_SYSTEM
            ),
        ),
        Message(role="user", content=f"User: {question}\n\nFACT CARD:\n{card}"),
    ]
    try:
        llm_result, _ = measure_complete(client, messages, stage="phrase")
    except RuntimeError:
        return fallback
    text = prose_only(llm_result.text or "")
    if not text:
        return fallback
    if _looks_like_numbered_list(text):
        return fallback
    if result.capability == "locate" and _has_extra_job_title(text, result):
        return fallback
    if _names_the_product(text):
        return fallback
    return text


def locate_card(result: AgentResult) -> str:
    if not result.nodes:
        return f"warnings: {', '.join(result.warnings) or 'not_found'}"
    top = result.nodes[0]
    conf = f"{result.confidence:.0%}" if result.confidence is not None else "unknown"
    lines = [
        f"unique occupation title: {top.pref_label}",
        f"kind: {top.kind}",
        f"confidence: {conf}",
    ]
    if top.description:
        lines.append(f"description: {top.description}")
    lines.extend(code_facts(result, top))
    lines.append("next: user may ask for essential or optional skills")
    return "\n".join(lines)


def connect_card(result: AgentResult, *, shown: int = CONNECT_PREVIEW_CAP) -> str:
    if not result.nodes:
        return "no neighbors"
    center = result.nodes[0]
    neighbors = result.nodes[1 : shown + 1]
    tags = relation_tags(result.edges, center.id)
    skills = [
        f"{node.pref_label} ({tags[node.id]})" if node.id in tags else node.pref_label
        for node in neighbors
    ]
    extra = max(0, len(result.nodes) - 1 - shown)
    lines = [
        f"occupation: {center.pref_label}",
        f"connections: {len(result.edges)}",
    ]
    if center.description:
        lines.append(f"description: {center.description}")
    lines.extend(
        [
            "shown skills: " + "; ".join(skills) if skills else "shown skills: none",
            "the app lists these skills itself; do not list them",
            "tags in parentheses (essential/optional/tool) come from the graph — "
            "repeat them as given, do not invent a tag for a skill that has none",
        ]
    )
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


def _names_the_product(text: str) -> bool:
    lowered = text.casefold()
    return "lfx" in lowered or "talent angels" in lowered


#: A pick-list intro is one sentence; a longer one has started answering for
#: the user (live: an invented side-by-side table of "typical tasks").
_INTRO_MAX_CHARS = 200


def _too_long_for_intro(text: str) -> bool:
    return len(text) > _INTRO_MAX_CHARS or "|" in text or "\n" in text.strip()


def _looks_like_numbered_list(text: str) -> bool:
    return bool(re.search(r"(?m)^\s*\d+\.\s+\S", text)) or "```" in text


def _looks_like_invented_miss(text: str) -> bool:
    lowered = text.casefold()
    if any(marker in lowered for marker in ("linked to", "appears as", "occupations such as")):
        return True
    if re.search(r"(?m)^\s*[-*]\s+\S", text):
        return True
    return False
