"""Golden Locate eval — hit@1 accuracy against the ESCO fixture.

Skipped (not failed) when Neo4j isn't reachable, per CONTRIBUTING.md.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip(
    "ta_taxonomies",
    reason="TA-taxonomies is not installed; install the sibling package for integration tests",
)

from talent_angels.evals import LocateMetrics  # noqa: E402
from talent_angels.skills.locate import locate  # noqa: E402
from talent_angels.suites import default_suite_registry  # noqa: E402
from tests.integration.support import neo4j_reachable  # noqa: E402

GOLDEN_PATH = Path(__file__).parents[1] / "evals" / "golden_locate.json"


def _load_cases() -> list[dict]:
    data = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    return data["cases"]


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not neo4j_reachable(), reason="Neo4j is not reachable"),
]


@pytest.mark.parametrize("case", _load_cases(), ids=lambda c: c["question"])
def test_golden_locate_case(case: dict) -> None:
    with default_suite_registry().open() as runtime:
        outcome = locate(runtime.suite, runtime.name, case["question"], kind=case["kind"])

    assert outcome.warnings == case["expected_warnings"]
    assert outcome.confidence == case["expected_confidence"]
    if case["expected_top_id"] is None:
        assert outcome.nodes == []
    elif case.get("metric", "hit_at_1") == "candidate_recall":
        assert case["expected_top_id"] in [node.id for node in outcome.nodes]
    else:
        assert outcome.nodes[0].id == case["expected_top_id"]


def test_golden_locate_metrics() -> None:
    metrics = LocateMetrics()
    with default_suite_registry().open() as runtime:
        for case in _load_cases():
            outcome = locate(runtime.suite, runtime.name, case["question"], kind=case["kind"])
            metrics.observe(
                case.get("metric", "hit_at_1"),
                case["expected_top_id"],
                [node.id for node in outcome.nodes],
            )

    result = metrics.as_dict()
    assert result["hit_at_1_accuracy"] == 1.0
    assert result["hit_at_1_questions"] == 10
    assert result["candidate_recall"] == 1.0
    assert result["candidate_recall_questions"] == 1
