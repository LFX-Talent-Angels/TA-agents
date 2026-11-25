"""Golden learning order on the live taxonomies (ADR-0013); skipped without Neo4j.

Values: tests/evals/golden_learning_order.json.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip(
    "ta_taxonomies",
    reason="TA-taxonomies is not installed; install the sibling package for integration tests",
)

from talent_angels.skills.roadmap import roadmap, successors  # noqa: E402
from talent_angels.suites import SuiteRegistry, default_suite_registry  # noqa: E402
from tests.integration.support import neo4j_reachable  # noqa: E402

ROOT = Path(__file__).parents[2]
GOLDEN = json.loads(
    (ROOT / "tests" / "evals" / "golden_learning_order.json").read_text(encoding="utf-8")
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not neo4j_reachable(), reason="Neo4j is not reachable"),
]


@pytest.fixture(scope="module")
def live() -> SuiteRegistry:
    return default_suite_registry(neighbor_cache=False)  # a warm cache must not flatter a rerun


def _roadmap(live: SuiteRegistry, case: dict[str, Any]) -> dict[str, Any]:
    return roadmap(live, case["from"], case["to"], suites=case["suites"], per_step=50)


def _items(data: dict[str, Any]) -> list[dict[str, Any]]:
    return [i for s in data["steps"] for i in s["items"]]


def test_onet_roadmap_keeps_both_levels_and_puts_knowledge_first(live: SuiteRegistry) -> None:
    case = GOLDEN["roadmap"][0]
    data = _roadmap(live, case)

    assert data["suites"]["onet"]["status"] == "ok"
    for want in case["items"]:
        found = [
            i for i in _items(data) if i["label"] == want["label"] and i["step"] == want["step"]
        ]
        assert found, want
        assert found[0]["levels"]["from"] == want["levels"]["from"]
        assert found[0]["levels"]["to"] == want["levels"]["to"]
        assert found[0]["stated_prerequisite"] is False
    zone = data["job_zone"]
    assert (zone["from"]["zone"], zone["to"]["zone"]) == (
        case["job_zone"]["from"],
        case["job_zone"]["to"],
    )


def test_esco_roadmap_starts_with_knowledge(live: SuiteRegistry) -> None:
    case = GOLDEN["roadmap"][1]
    data = _roadmap(live, case)

    assert data["suites"]["esco"]["status"] == "ok"
    first = next(s for s in data["steps"] if s["step"] == 1)
    assert first["name"] == "Foundations"
    assert all(case["step1_reason"] in i["reasons"] for i in first["items"])
    assert all(i["cite"].startswith("http://data.europa.eu/esco/") for i in _items(data))


def test_an_esco_skill_with_essential_for_links_has_stated_children(live: SuiteRegistry) -> None:
    case = GOLDEN["successors"][0]
    part = successors(live, case["x"], suites=[case["suite"]], depth=1, breadth=10)["suites"][
        case["suite"]
    ]

    assert part["status"] == "ok"
    assert part["start"]["label"] == case["label"]
    stated = [e for e in part["edges"] if e["kind"] == "stated"]
    assert len(stated) >= case["min_stated_children"]
    kinds = [e["kind"] for e in part["edges"]]
    assert kinds == sorted(kinds, key=lambda k: k != "stated")  # stated first


def test_sfia_levels_follow_in_order(live: SuiteRegistry) -> None:
    case = GOLDEN["successors"][1]
    part = successors(live, case["x"], suites=[case["suite"]], depth=case["depth"])["suites"][
        case["suite"]
    ]

    assert [n["id"] for n in part["nodes"][1:]] == case["children"]


def test_reruns_are_byte_identical(live: SuiteRegistry) -> None:
    def runs() -> tuple[str, ...]:
        return (
            json.dumps(_roadmap(live, GOLDEN["roadmap"][0])),
            json.dumps(_roadmap(live, GOLDEN["roadmap"][1])),
            json.dumps(successors(live, GOLDEN["successors"][0]["x"], suites=["esco"], depth=2)),
        )

    assert runs() == runs()
