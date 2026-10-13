"""Concrete SFIA suite lifecycle; opens the sibling TA-taxonomies adapter."""

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
