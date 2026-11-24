"""Golden Graph-Mapper maps on the live taxonomies (ESCO, SFIA, JobTech, O*NET, BLS).

Skipped when Neo4j isn't reachable. Values: tests/evals/golden_graph_mapper.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip(
    "ta_taxonomies",
    reason="TA-taxonomies is not installed; install the sibling package for integration tests",
)

from talent_angels.skills.connect.neighbourhood import map_from_text, map_route  # noqa: E402
from talent_angels.skills.pathfind.dispatch import pathfind_one  # noqa: E402
from talent_angels.suites import default_suite_registry  # noqa: E402
from tests.integration.support import neo4j_reachable  # noqa: E402

GOLDEN_PATH = Path(__file__).parents[1] / "evals" / "golden_graph_mapper.json"
CASES = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))["cases"]

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not neo4j_reachable(), reason="Neo4j is not reachable"),
]


def _run(case: dict) -> dict:
    with default_suite_registry().open(case["suite"]) as runtime:
        suite, name = runtime.suite, runtime.name
        if "route" in case:
            ids = case.get("expect_route_ids")
            found, _ = pathfind_one(suite, name, *case["route"])
            assert found.routes, "pathfind found no route"
            assert found.routes[0].node_ids == ids
            return map_route(
                suite, name, ids, depth=case["depth"], breadth=case["breadth"]
            ).to_dict()
        return map_from_text(
            suite, name, case["start"], depth=case["depth"], breadth=case["breadth"]
        ).to_dict()


@pytest.mark.parametrize(
    "case", CASES, ids=lambda c: f"{c['suite']}:{c.get('start') or c['route']}"
)
def test_golden_graph_mapper_case(case: dict) -> None:
    first = _run(case)
    second = _run(case)

    assert json.dumps(first, sort_keys=False) == json.dumps(second, sort_keys=False)
    assert first["warnings"] == case["expect_warnings"]
    assert [n["label"] for n in first["nodes"] if n["level"] == 1] == case["expect_level1_labels"]
    assert len(first["nodes"]) >= case["expect_min_nodes"]
    assert max(n["level"] for n in first["nodes"]) <= case["depth"]
    assert all(p["kept"] <= case["breadth"] for p in first["pruned"])
    if "route" in case:
        assert len(first["route"]) == case["expect_route_stops"]
        assert [n["id"] for n in first["nodes"] if n["on_route"]] == first["route"]
