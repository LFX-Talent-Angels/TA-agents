"""Concrete BLS suite lifecycle; opens the sibling TA-taxonomies adapter."""

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
