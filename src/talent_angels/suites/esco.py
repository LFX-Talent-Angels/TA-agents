"""Concrete ESCO suite lifecycle; the only TA-taxonomies infrastructure edge."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from ta_taxonomies.suites.esco.db import neo4j_driver
from ta_taxonomies.suites.esco.tools import EscoSuite

from talent_angels.suites.registry import SuiteRuntime

ESCO_SUITE_NAME = "esco"


@contextmanager
def open_esco_runtime() -> Iterator[SuiteRuntime]:
    """Open one ESCO suite and close its Neo4j driver with the context."""
    with neo4j_driver() as (driver, database):
        suite = EscoSuite(driver, database=database)

        def health_check() -> bool:
            try:
                driver.verify_connectivity()
                return True
            except Exception:
                return False

        yield SuiteRuntime(name=ESCO_SUITE_NAME, suite=suite, health_check=health_check)
