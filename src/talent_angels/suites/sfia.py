"""Concrete SFIA suite lifecycle, the same shape as the ESCO adapter.

The suite reads its own ``SFIA_NEO4J_*`` overrides before the shared
``NEO4J_*``, so one suite can be pointed somewhere else without moving the
others. All four coexist in one graph, so the usual case is that every adapter
resolves to the same connection.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from ta_taxonomies.suites.sfia.db import neo4j_driver
from ta_taxonomies.suites.sfia.tools import SfiaSuite

from talent_angels.suites.registry import SuiteRuntime

SFIA_SUITE_NAME = "sfia"


@contextmanager
def open_sfia_runtime() -> Iterator[SuiteRuntime]:
    """Open one SFIA suite and close its Neo4j driver with the context."""
    with neo4j_driver() as (driver, database):
        suite = SfiaSuite(driver, database=database)

        def health_check() -> bool:
            try:
                driver.verify_connectivity()
                return True
            except Exception:
                return False

        yield SuiteRuntime(name=SFIA_SUITE_NAME, suite=suite, health_check=health_check)
