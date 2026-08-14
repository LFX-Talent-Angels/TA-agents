"""Build a live ESCO suite from env — one Neo4j driver per `with` block.

`neo4j_driver()` (TA-taxonomies) reads `NEO4J_URI`/`NEO4J_USER`/
`NEO4J_PASSWORD` from the same `.env` this repo uses. The FastAPI edge opens
one of these for the app lifetime (lifespan); the CLI opens one per command.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from ta_taxonomies.suites.esco.db import neo4j_driver
from ta_taxonomies.suites.esco.tools import EscoSuite


@contextmanager
def open_esco_suite() -> Iterator[EscoSuite]:
    with neo4j_driver() as (driver, database):
        yield EscoSuite(driver, database=database)
