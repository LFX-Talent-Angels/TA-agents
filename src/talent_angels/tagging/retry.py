"""Context-widening helpers for a reviewer-triggered retry (`review.RetryRequest`).

A retry climbs the *standard's own tree* for more context instead of just
re-asking the same question: attempt 1 adds the parent heading, attempt 2
also adds the grandparent and the unit's sibling lines — the same ladder the
pilot's automatic widened attempt uses (`runner._widen_pending`), just driven
by the reviewer instead of a `none` fit.
"""

from __future__ import annotations

from talent_angels.tagging.models import TagUnit
from talent_angels.tagging.profile import Profile
from talent_angels.tagging.rewrite import Rewrite

_MAX_SIBLINGS = 4


def siblings_for(unit: TagUnit, units: list[TagUnit]) -> list[str]:
    """Up to `_MAX_SIBLINGS` other units sharing `unit.group_key`, in order."""
    out = [
        other.clean_text
        for other in units
        if other.group_key == unit.group_key and other.unit_key != unit.unit_key
    ]
    return out[:_MAX_SIBLINGS]


def extra_queries_for(unit: TagUnit, rewrite: Rewrite, profile: Profile, attempt: int) -> list[str]:
    """Context-climbing queries a retry adds on top of the unit's own ones.

    Attempt 1: description + parent heading (same shape as the automatic
    widened attempt). Attempt 2: also description + grandparent heading.
    """
    description = rewrite.description or unit.clean_text
    path = unit.context_path
    extra: list[str] = []
    if path:
        parent = profile.clean(path[-1])
        if parent:
            extra.append(f"{description} {parent}".strip())
    if attempt >= 2 and len(path) >= 2:
        grandparent = profile.clean(path[-2])
        if grandparent:
            extra.append(f"{description} {grandparent}".strip())
    return extra


def choose_hint_for(review_hint: str, attempt: int, siblings: list[str]) -> str:
    """The `choose` hint for a retry: the reviewer's hint, plus sibling lines at attempt >= 2."""
    lines = [review_hint] if review_hint else []
    if attempt >= 2 and siblings:
        lines.append("Sibling lines: " + "; ".join(siblings))
    return "\n".join(lines)
