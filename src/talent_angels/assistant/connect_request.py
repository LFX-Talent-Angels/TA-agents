"""Deterministic request extraction for the thin Connect MVP."""

from __future__ import annotations

import re

from talent_angels.contracts import NodeRef
from talent_angels.skills.connect.models import ConnectRequest


class UnsupportedConnectQuery(ValueError):
    """Raised when the zero-token MVP cannot safely identify a subject."""


_SKILLS_DOES = re.compile(
    r"(?:essential\s+|optional\s+)?skills\s+does\s+"
    r"(?:(?:an?|the)\s+)?(.+?)\s+need[?.!]*$",
    re.IGNORECASE,
)
_SKILLS_FOR = re.compile(
    r"skills\s+for\s+(?:(?:an?|the)\s+)?(.+?)[?.!]*$",
    re.IGNORECASE,
)
_SKILLS_NEED_TO_BE = re.compile(
    r"skills\s+(?:(?:do\s+)?i\s+|does\s+one\s+)?need\s+to\s+(?:become|be)\s+"
    r"(?:(?:an?|the)\s+)?(.+?)[?.!]*$",
    re.IGNORECASE,
)
_NEIGHBORS_OF = re.compile(
    r"neighbou?rs\s+of\s+(?:(?:an?|the)\s+)?(.+?)[?.!]*$",
    re.IGNORECASE,
)


def extract_connect_request(question: str) -> ConnectRequest:
    """Extract only the explicitly supported zero-token Connect query forms."""
    cleaned = question.strip()
    lowered = cleaned.lower()
    relation_kind = None
    if "essential skill" in lowered:
        relation_kind = "essential"
    elif "optional skill" in lowered:
        relation_kind = "optional"

    match = (
        _SKILLS_DOES.search(cleaned)
        or _SKILLS_FOR.search(cleaned)
        or _SKILLS_NEED_TO_BE.search(cleaned)
    )
    if match:
        return ConnectRequest(
            subject=match.group(1).strip(),
            rel_types=("HAS_SKILL",),
            relation_kind=relation_kind,
        )

    match = _NEIGHBORS_OF.search(cleaned)
    if match:
        return ConnectRequest(subject=match.group(1).strip())

    raise UnsupportedConnectQuery("could not extract a Connect subject from the question")


def _normalized_followup(question: str) -> str:
    return re.sub(r"[?.!]+$", "", question.strip()).strip().casefold()


_FOLLOWUP_ESSENTIAL = frozenset(
    {
        "essential skills",
        "what are the essential skills",
        "what do i actually need",
        "essential",
    }
)
_FOLLOWUP_OPTIONAL = frozenset(
    {
        "optional skills",
        "what are the optional skills",
        "optional",
    }
)
_FOLLOWUP_SKILLS = frozenset(
    {
        "list the skills",
        "those skills",
        "the skills",
        "list skills",
        "show me the skills",
        "what are the skills",
        "its skills",
        "skills",
    }
)
_FOLLOWUP_NEIGHBORS = frozenset(
    {
        "neighbors",
        "neighbours",
        "neighbors of that",
        "neighbours of that",
    }
)


_DESCRIBE_RE = re.compile(
    r"what\s+does\s+.+\s+do|"
    r"what\s+do\s+they\s+do|"
    r"what\s+is\s+this(\s+job|\s+occupation)?|"
    r"\bduties\b|"
    r"\bdefinition\b|"
    r"describe\s+(this|the|that|it|the\s+job)",
    re.IGNORECASE,
)


def is_describe_followup(question: str, bindings: dict[str, NodeRef]) -> bool:
    """True when the user asks what the already-bound occupation does."""
    if not bindings:
        return False
    if not _DESCRIBE_RE.search(question.strip()):
        return False
    q = f" {question.casefold()} "
    if any(token in q for token in (" they ", " this ", " that ", " it ", " the job ")):
        return True
    for node in bindings.values():
        label = node.pref_label.casefold().strip()
        stem = label.rstrip("s")
        if label and (label in q or (stem and stem in q)):
            return True
    return False


def followup_connect_request(question: str, bound: NodeRef) -> ConnectRequest | None:
    """Build a Connect request for a short follow-up around an already-bound node.

    Returns None when the user names a different occupation than ``bound.pref_label``
    so the existing locate-then-connect path can run.
    """
    normalized = _normalized_followup(question)
    if normalized in _FOLLOWUP_ESSENTIAL:
        return ConnectRequest(
            subject=bound.pref_label,
            rel_types=("HAS_SKILL",),
            relation_kind="essential",
        )
    if normalized in _FOLLOWUP_OPTIONAL:
        return ConnectRequest(
            subject=bound.pref_label,
            rel_types=("HAS_SKILL",),
            relation_kind="optional",
        )
    if normalized in _FOLLOWUP_SKILLS:
        return ConnectRequest(subject=bound.pref_label, rel_types=("HAS_SKILL",))
    if normalized in _FOLLOWUP_NEIGHBORS:
        return ConnectRequest(subject=bound.pref_label)
    if re.search(r"\b(become|becoming)\s+(an?\s+)?(one|that|this|it)\b", normalized):
        return ConnectRequest(
            subject=bound.pref_label,
            rel_types=("HAS_SKILL",),
            relation_kind="essential",
        )
    if re.search(r"how\s+(do\s+i\s+|can\s+i\s+|to\s+)become\s*$", normalized):
        return ConnectRequest(
            subject=bound.pref_label,
            rel_types=("HAS_SKILL",),
            relation_kind="essential",
        )

    try:
        request = extract_connect_request(question)
    except UnsupportedConnectQuery:
        return _bound_skills_followup(normalized, bound)
    if request.subject.casefold() != bound.pref_label.casefold():
        return None
    return request


def _bound_skills_followup(normalized: str, bound: NodeRef) -> ConnectRequest | None:
    """Map a skills question with no extracted subject onto the bound occupation."""
    if re.search(r"\bessential\s+skills?\b", normalized) or (
        re.search(r"\bskills?\b", normalized) and re.search(r"\bneed\b", normalized)
    ):
        return ConnectRequest(
            subject=bound.pref_label,
            rel_types=("HAS_SKILL",),
            relation_kind="essential",
        )
    if re.search(r"\boptional\s+skills?\b", normalized):
        return ConnectRequest(
            subject=bound.pref_label,
            rel_types=("HAS_SKILL",),
            relation_kind="optional",
        )
    if re.search(r"\bskills\b", normalized):
        return ConnectRequest(subject=bound.pref_label, rel_types=("HAS_SKILL",))
    return None
