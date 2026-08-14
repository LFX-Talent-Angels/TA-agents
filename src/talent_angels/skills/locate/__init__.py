"""Locate (Resolve): free text -> suite node candidates + confidence.

Resolution order (owned by the suite tool, not this skill): exact match,
alias/label match, case-insensitive/contains fallback. Confidence attaches to
every result and crosses all later steps.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from talent_angels.skills.locate.resolve import SearchableSuite, locate

ESCO_SUITE_NAME = "esco"


@contextmanager
def open_esco_suite() -> Iterator[SearchableSuite]:
    """Open the optional concrete ESCO adapter without importing it eagerly."""
    from talent_angels.skills.locate.esco import open_esco_suite as _open_esco_suite

    with _open_esco_suite() as suite:
        yield suite


__all__ = ["ESCO_SUITE_NAME", "SearchableSuite", "locate", "open_esco_suite"]
