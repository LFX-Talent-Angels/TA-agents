"""Shared availability probes for live integration tests."""

from __future__ import annotations

from ta_taxonomies.suites.esco.db import neo4j_driver


def neo4j_reachable() -> bool:
    """Return whether the configured Neo4j instance accepts a connection."""
    try:
        with neo4j_driver() as (driver, _database):
            driver.verify_connectivity()
        return True
    except Exception:
        return False
