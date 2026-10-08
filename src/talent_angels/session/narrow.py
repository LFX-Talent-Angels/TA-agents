"""Narrowing an open pick list: an area letter, or a hint about the work.

While a list is on screen, "the ones that build bridges" or "something
outdoors" is about that list, not a new search. The model may only point at
what is shown (option numbers, area letters); code checks every pointer and
builds the shorter list itself. A hint never binds a title on its own.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

from talent_angels.assistant.llm_call import measure_complete
from talent_angels.llm import LLMClient, Message
from talent_angels.session.i18n import plural, t
from talent_angels.session.models import AreaChoice, PendingChoice
from talent_angels.session.phrase import uses_chat_phrasing
from talent_angels.skills.locate.areas import Area

_LETTERS = "ABCDEFGHIJKLMNOP"

NARROW_SYSTEM = """The user is choosing from a numbered list of job titles, and
possibly from lettered areas (occupation groups). Read their new message and
return ONLY a JSON object:
{"action": "narrow" | "new", "options": [numbers], "areas": [letters]}

- narrow: the message describes, filters or chooses among what is shown ("the
  ones that build bridges", "something outdoors", "not the technician ones",
  "the civil ones"). List only the option numbers and area letters whose own
  title clearly fits the description; leave out titles that merely might. Use
  [] for a list where nothing clearly fits. A message that only adds a detail
  to the same topic, naming no new job ("something with children", "outdoors
  work"), is narrow even when nothing on the list fits.
- new: the message asks for something else (another title, a different
  question, a command).

Use only the numbers and letters shown. Never add titles."""

_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)
_AREA_LETTER = re.compile(r"^\s*(?:area\s+)?([a-p])\s*[.)!]?\s*$", re.IGNORECASE)
_WORD = re.compile(r"[a-z0-9]+")
#: Without a model, only text shaped like a hint narrows; anything else is a
#: new search ("what skills does a civil engineer need?" shares a word too).
_HINT_SHAPE = re.compile(
    r"^\s*(?:the\s+(?:\w+\s+)?ones?|ones?\s+(?:that|who|with|in)|something|only|just|"
    r"not\s+the|i\s+mean|more\s+like|those|in\s+\w+)\b",
    re.IGNORECASE,
)
_STOP = frozenset(
    "a an and any are as at be but by do for from in is it its me my of on one ones "
    "or so some something that the them these they this those to want with who".split()
)


@dataclass(frozen=True)
class Narrowing:
    action: Literal["narrow", "new"]
    options: list[int] = field(default_factory=list)
    areas: list[str] = field(default_factory=list)


def area_choices(
    areas_by_suite: dict[str, list[Area]], *, query: str, kind: str | None
) -> list[AreaChoice]:
    """Letter the areas of the first suite that reported any (one list, short)."""
    for areas in areas_by_suite.values():
        if len(areas) >= 2:
            return [
                AreaChoice(
                    letter=_LETTERS[index],
                    suite=area.suite,
                    code=area.code,
                    label=area.label,
                    count=area.count,
                    query=query,
                    kind=kind,
                )
                for index, area in enumerate(areas[: len(_LETTERS)])
            ]
    return []


def render_areas(choices: Sequence[AreaChoice]) -> str:
    """The "which area?" block printed under a pick list."""
    if not choices:
        return ""
    lines = [t("areas_head"), ""]
    for choice in choices:
        noun = plural(choice.count, "area_title", "area_titles")
        lines.append(f"{choice.letter}. {choice.label} ({choice.count} {noun})")
    lines.extend(["", t("areas_foot")])
    return "\n".join(lines)


def area_by_letter(text: str, choices: Sequence[AreaChoice]) -> AreaChoice | None:
    """ "B" or "area b" names an offered area."""
    m = _AREA_LETTER.match(text)
    if not m:
        return None
    letter = m.group(1).upper()
    return next((choice for choice in choices if choice.letter == letter), None)


def _stems(text: str) -> set[str]:
    words = (w for w in _WORD.findall(text.casefold()) if w not in _STOP and len(w) > 2)
    return {w[:5] for w in words}


def _overlap_narrowing(
    text: str, options: Sequence[PendingChoice], areas: Sequence[AreaChoice]
) -> Narrowing:
    """No model: keep what shares a word stem with the hint, else a new search."""
    if not _HINT_SHAPE.match(text):
        return Narrowing("new")
    stems = _stems(text)
    picked = [c.number for c in options if stems & _stems(c.node.pref_label)]
    letters = [a.letter for a in areas if stems & _stems(a.label)]
    if picked or letters:
        return Narrowing("narrow", picked, letters)
    return Narrowing("new")


def _menu(options: Sequence[PendingChoice], areas: Sequence[AreaChoice]) -> str:
    lines = ["Options:", *(f"{c.number}. {c.node.pref_label}" for c in options)]
    if areas:
        lines.extend(["Areas:", *(f"{a.letter}. {a.label}" for a in areas)])
    return "\n".join(lines)


def _parse(text: str, options: Sequence[PendingChoice], areas: Sequence[AreaChoice]) -> Narrowing:
    match = _JSON_OBJECT.search(text)
    if match is None:
        raise ValueError("no JSON object")
    payload = json.loads(match.group(0))
    if not isinstance(payload, dict) or payload.get("action") not in ("narrow", "new"):
        raise ValueError("bad action")
    if payload["action"] == "new":
        return Narrowing("new")
    numbers = {c.number for c in options}
    letters = {a.letter for a in areas}
    picked = [n for n in payload.get("options") or [] if isinstance(n, int) and n in numbers]
    chosen = [
        str(x).upper()
        for x in payload.get("areas") or []
        if isinstance(x, str) and str(x).upper() in letters
    ]
    return Narrowing("narrow", list(dict.fromkeys(picked)), list(dict.fromkeys(chosen)))


def narrow_decision(
    client: LLMClient | None,
    text: str,
    options: Sequence[PendingChoice],
    areas: Sequence[AreaChoice],
) -> Narrowing:
    """Is ``text`` about the list on screen, and which parts of it?"""
    if not uses_chat_phrasing(client):
        return _overlap_narrowing(text, options, areas)
    assert client is not None
    messages = [
        Message(role="system", content=NARROW_SYSTEM),
        Message(role="user", content=f"{_menu(options, areas)}\n\nMessage: {text}"),
    ]
    try:
        result, _ = measure_complete(client, messages, stage="narrow")
        return _parse(result.text or "", options, areas)
    except (RuntimeError, ValueError, json.JSONDecodeError):
        return _overlap_narrowing(text, options, areas)
