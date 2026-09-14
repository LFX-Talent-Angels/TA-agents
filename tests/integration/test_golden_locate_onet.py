"""Golden Locate eval against the loaded O*NET graph (fixture or full 31.0)."""

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

GOLDEN_PATH = Path(__file__).parents[1] / "evals" / "golden_locate_onet.json"


def _load_cases() -> list[dict]:
    data = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    return data["cases"]


def _onet_loaded() -> bool:
    if not neo4j_reachable():
        return False
    try:
        with default_suite_registry().open("onet") as runtime:
            outcome = locate(runtime.suite, runtime.name, "Software Developers", kind="occupation")
        return any(node.id == "onet:occupation:15-1252.00" for node in outcome.nodes)
    except Exception:
        return False


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not neo4j_reachable(), reason="Neo4j is not reachable"),
    pytest.mark.skipif(not _onet_loaded(), reason="O*NET graph is not loaded"),
]


@pytest.mark.parametrize("case", _load_cases(), ids=lambda c: c["question"])
def test_golden_locate_onet_case(case: dict) -> None:
    with default_suite_registry().open("onet") as runtime:
        outcome = locate(runtime.suite, runtime.name, case["question"], kind=case["kind"])

    actual_warnings = set(outcome.warnings)
    assert set(case["expected_warnings"]).issubset(actual_warnings)
    assert actual_warnings - set(case["expected_warnings"]) <= {
        "truncated",
        "match_count_capped",
        "ambiguous",
    }
    if case["expected_top_id"] is None:
        assert outcome.nodes == []
        return
    assert outcome.nodes[0].id == case["expected_top_id"]
    assert outcome.confidence == case["expected_confidence"]


def test_golden_locate_onet_metrics() -> None:
    metrics = LocateMetrics()
    with default_suite_registry().open("onet") as runtime:
        for case in _load_cases():
            outcome = locate(runtime.suite, runtime.name, case["question"], kind=case["kind"])
            metrics.observe(
                "hit_at_1",
                case["expected_top_id"],
                [node.id for node in outcome.nodes],
            )
    result = metrics.as_dict()
    assert result["hit_at_1_accuracy"] == 1.0
    # not_found cases are excluded from hit@1 (same as ESCO golden metrics).
    assert result["hit_at_1_questions"] == 10
