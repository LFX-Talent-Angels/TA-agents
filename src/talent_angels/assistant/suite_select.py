"""Choose which attached taxonomies a turn should search.

The registry lists every suite wired into the process. This module decides
the subset for one question: an explicit override wins; a named source in
the question may narrow; otherwise every attached suite is selected.
Health filtering happens at dispatch (skip unreachable), not here.
"""

from __future__ import annotations

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


def named_suites(question: str, available: Sequence[str]) -> tuple[str, ...]:
    """Return attached suites the user named, in `available` order."""
    text = question.casefold()
    hits: list[str] = []
    for name in available:
        for alias in _aliases(name):
            if alias.casefold() in text:
                hits.append(name)
                break
    return tuple(hits)


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
