"""Choose which attached taxonomies a turn should search.

The registry lists every suite wired into the process. This module decides
the subset for one question: an explicit override wins; a named source in
the question may narrow; otherwise every attached suite is selected.
Health filtering happens at dispatch (skip unreachable), not here.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from talent_angels.assistant.merge import suite_heading
from talent_angels.suites.registry import UnknownSuiteError

# Aliases a user might type. Keys are registry names.
SUITE_ALIASES: dict[str, tuple[str, ...]] = {
    "esco": ("esco",),
    "onet": ("onet", "o*net", "o-net"),
    "sfia": ("sfia",),
    "bls": ("bls",),
}


def _aliases(name: str) -> tuple[str, ...]:
    return SUITE_ALIASES.get(name, (name,))


def resolve_show_token(token: str, available: Sequence[str]) -> str | None:
    """Map 'all' / 'O*NET' / 'sfia' to a registry name. Unknown → None."""
    text = token.casefold().strip()
    if text == "all":
        return "all"
    for name in available:
        labels = (name, suite_heading(name), *_aliases(name))
        if text in {item.casefold() for item in labels}:
            return name
    return None


def _normalize_mentions(question: str) -> str:
    """Fold O*NET spellings to the token 'onet' without substring traps."""
    text = question.casefold()
    text = text.replace("o*net", " onet ").replace("o-net", " onet ")
    text = re.sub(r"\bo\s+net\b", " onet ", text)
    return text


def _alias_token(alias: str) -> str:
    return alias.casefold().replace("o*net", "onet").replace("o-net", "onet")


def _token_in(text: str, token: str) -> bool:
    if not token:
        return False
    return re.search(rf"(?<![a-z0-9]){re.escape(token)}(?![a-z0-9])", text) is not None


def named_suites(question: str, available: Sequence[str]) -> tuple[str, ...]:
    """Return suites the user named as whole tokens, in `available` order."""
    text = _normalize_mentions(question)
    hits: list[str] = []
    for name in available:
        for alias in _aliases(name):
            if _token_in(text, _alias_token(alias)):
                hits.append(name)
                break
    return tuple(hits)


def named_unattached(question: str, available: Sequence[str]) -> tuple[str, ...]:
    """Suite aliases the user named that are not in the attached registry."""
    known = tuple(dict.fromkeys([*SUITE_ALIASES.keys(), *available]))
    return tuple(name for name in named_suites(question, known) if name not in available)


def select_suites(
    *,
    available: Sequence[str],
    override: str | None = None,
    question: str = "",
) -> tuple[str, ...]:
    """Resolve the suite list for one turn.

    * ``override`` (CLI ``--suite`` / API ``suite``) must be attached.
    * Else any taxonomy names in ``question`` that are attached.
    * Else every attached suite.
    """
    attached = tuple(available)
    if not attached:
        raise ValueError("no suites attached")
    if override is not None:
        if override not in attached:
            listed = ", ".join(attached)
            raise UnknownSuiteError(f"unknown suite {override!r}; available: {listed}")
        return (override,)
    named = named_suites(question, attached)
    if named:
        return named
    return attached
