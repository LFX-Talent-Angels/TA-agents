"""Concrete O*NET suite lifecycle, the same shape as the ESCO adapter.

The suite reads its own ``ONET_NEO4J_*`` overrides before the shared
``NEO4J_*``, so one suite can be pointed somewhere else without moving the
others. All four coexist in one graph, so the usual case is that every adapter
resolves to the same connection.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from ta_taxonomies.suites.onet.db import neo4j_driver
from ta_taxonomies.suites.onet.tools import OnetSuite

from talent_angels.suites.registry import SuiteRuntime

ONET_SUITE_NAME = "onet"


@contextmanager
def open_onet_runtime() -> Iterator[SuiteRuntime]:
    """Open one O*NET suite and close its Neo4j driver with the context."""
    with neo4j_driver() as (driver, database):
        suite = OnetSuite(driver, database=database)

        def health_check() -> bool:
            try:
                driver.verify_connectivity()
                return True
            except Exception:
                return False

        yield SuiteRuntime(name=ONET_SUITE_NAME, suite=suite, health_check=health_check)
