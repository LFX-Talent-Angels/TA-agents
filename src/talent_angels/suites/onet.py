"""Concrete O*NET suite lifecycle; opens the sibling TA-taxonomies adapter."""

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
