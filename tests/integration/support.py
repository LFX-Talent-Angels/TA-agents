"""Shared availability probes for live integration tests."""

from __future__ import annotations

from functools import lru_cache

from talent_angels.suites import default_suite_registry


@lru_cache(maxsize=1)
def neo4j_reachable() -> bool:
    """Return whether the configured Neo4j instance accepts a connection."""
    try:
        with default_suite_registry().open() as runtime:
            return runtime.is_reachable()
    except Exception:
        return False
