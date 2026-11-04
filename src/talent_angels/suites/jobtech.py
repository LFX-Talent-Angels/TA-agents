"""Concrete JobTech suite lifecycle; opens the sibling TA-taxonomies adapter."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from ta_taxonomies.suites.jobtech.db import neo4j_driver
from ta_taxonomies.suites.jobtech.tools import JobtechSuite

from talent_angels.suites.registry import SuiteRuntime

JOBTECH_SUITE_NAME = "jobtech"


@contextmanager
def open_jobtech_runtime() -> Iterator[SuiteRuntime]:
    """Open one JobTech suite and close its Neo4j driver with the context."""
    with neo4j_driver() as (driver, database):
        suite = JobtechSuite(driver, database=database)

        def health_check() -> bool:
            try:
                driver.verify_connectivity()
                return True
            except Exception:
                return False

        yield SuiteRuntime(name=JOBTECH_SUITE_NAME, suite=suite, health_check=health_check)
