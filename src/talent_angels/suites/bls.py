"""Concrete BLS suite lifecycle, the same shape as the ESCO adapter.

The suite reads its own ``BLS_NEO4J_*`` overrides before the shared
``NEO4J_*``, so one suite can be pointed somewhere else without moving the
others. All four coexist in one graph, so the usual case is that every adapter
resolves to the same connection.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from ta_taxonomies.suites.bls.db import neo4j_driver
from ta_taxonomies.suites.bls.tools import BlsSuite

from talent_angels.suites.registry import SuiteRuntime

BLS_SUITE_NAME = "bls"


@contextmanager
def open_bls_runtime() -> Iterator[SuiteRuntime]:
    """Open one BLS suite and close its Neo4j driver with the context."""
    with neo4j_driver() as (driver, database):
        suite = BlsSuite(driver, database=database)

        def health_check() -> bool:
            try:
                driver.verify_connectivity()
                return True
            except Exception:
                return False

        yield SuiteRuntime(name=BLS_SUITE_NAME, suite=suite, health_check=health_check)
