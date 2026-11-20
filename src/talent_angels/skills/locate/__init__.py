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
    """Compatibility helper backed by the central lazy suite registry."""
    from talent_angels.suites import default_suite_registry

    with default_suite_registry().open(ESCO_SUITE_NAME) as runtime:
        yield runtime.suite


__all__ = ["ESCO_SUITE_NAME", "SearchableSuite", "locate", "open_esco_suite"]
