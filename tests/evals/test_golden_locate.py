"""Golden Locate eval — hit@1 accuracy against the ESCO fixture.

Skipped (not failed) when Neo4j isn't reachable, per CONTRIBUTING.md.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from ta_taxonomies.suites.esco.db import neo4j_driver
from ta_taxonomies.suites.esco.tools import EscoSuite

from talent_angels.skills.locate import ESCO_SUITE_NAME, locate

GOLDEN_PATH = Path(__file__).parent / "golden_locate.json"


def _neo4j_reachable() -> bool:
    try:
        with neo4j_driver() as (driver, _database):
            driver.verify_connectivity()
        return True
    except Exception:
        return False


def _load_cases() -> list[dict]:
    data = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    return data["cases"]


pytestmark = pytest.mark.skipif(
    not _neo4j_reachable(), reason="Neo4j not reachable; see TA-taxonomies NOTES.md"
)


@pytest.mark.parametrize("case", _load_cases(), ids=lambda c: c["question"])
def test_golden_locate_case(case: dict) -> None:
    with neo4j_driver() as (driver, database):
        suite = EscoSuite(driver, database=database)
        outcome = locate(suite, ESCO_SUITE_NAME, case["question"], kind=case["kind"])

    assert outcome.warnings == case["expected_warnings"]
    assert outcome.confidence == case["expected_confidence"]
    if case["expected_top_id"] is None:
        assert outcome.nodes == []
    else:
        assert outcome.nodes[0].id == case["expected_top_id"]


def test_golden_locate_hit_at_1_accuracy() -> None:
    cases = [c for c in _load_cases() if c["expected_top_id"] is not None]
    hits = 0
    with neo4j_driver() as (driver, database):
        suite = EscoSuite(driver, database=database)
        for case in cases:
            outcome = locate(suite, ESCO_SUITE_NAME, case["question"], kind=case["kind"])
            if outcome.nodes and outcome.nodes[0].id == case["expected_top_id"]:
                hits += 1

    accuracy = hits / len(cases)
    assert accuracy == 1.0, f"hit@1 accuracy {accuracy:.0%} on {len(cases)} golden questions"
