"""Golden Pathfind eval — real occupation pairs against full ESCO/O*NET.

Skipped (not failed) when Neo4j isn't reachable, per CONTRIBUTING.md. This
is also the set ADR-0008 decision 5 names as the source of truth for
MAX_REALISTIC_GAP_SKILLS — see tests/evals/golden_pathfind.json's note and
skills/pathfind/gap.py's module comment for how these numbers were used.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip(
    "ta_taxonomies",
    reason="TA-taxonomies is not installed; install the sibling package for integration tests",
)

from talent_angels.skills.connect import ConnectRequest, connect  # noqa: E402
from talent_angels.skills.locate import locate  # noqa: E402
from talent_angels.skills.locate.rank import group_and_sort_locate  # noqa: E402
from talent_angels.skills.pathfind.gap import compute_gap  # noqa: E402
from talent_angels.skills.pathfind.routes import find_routes, supports_enumerate_paths  # noqa: E402
from talent_angels.suites import default_suite_registry  # noqa: E402
from tests.integration.support import neo4j_reachable  # noqa: E402

GOLDEN_PATH = Path(__file__).parents[1] / "evals" / "golden_pathfind.json"


def _load_cases() -> list[dict]:
    data = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    return data["cases"]


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not neo4j_reachable(), reason="Neo4j is not reachable"),
]


def _case_id(case: dict) -> str:
    return f"{case['suite']}:{case['start']}->{case['target']}"


@pytest.mark.parametrize("case", _load_cases(), ids=_case_id)
def test_golden_pathfind_case(case: dict) -> None:
    """Locate+Connect both ends live, same as pathfind_one, kept unmerged
    here so essential/optional/route counts can be asserted separately."""
    with default_suite_registry().open(case["suite"]) as runtime:
        suite, suite_name = runtime.suite, runtime.name
        schema = suite.suite_schema

        start_found = locate(suite, suite_name, case["start"], kind="occupation")
        start_located = group_and_sort_locate(
            suite,
            start_found,
            case["start"],
            suite_name=suite_name,
            group_rel_type=schema.group_rel_type,
            group_node_kinds=schema.group_node_kinds,
        )
        assert start_located.nodes, f"start {case['start']!r} did not resolve"
        assert start_located.nodes[0].id == case["start_id"]
        start_result = connect(
            suite,
            suite_name,
            start_located.nodes[0],
            request=ConnectRequest(subject=case["start"], rel_types=schema.skill_rel_types),
            confidence=start_located.confidence,
            locate_evidence=start_located.evidence,
            optional_rel_values=schema.optional_rel_values,
        )

        target_found = locate(suite, suite_name, case["target"], kind="occupation")
        target_located = group_and_sort_locate(
            suite,
            target_found,
            case["target"],
            suite_name=suite_name,
            group_rel_type=schema.group_rel_type,
            group_node_kinds=schema.group_node_kinds,
        )
        assert target_located.nodes, f"target {case['target']!r} did not resolve"
        assert target_located.nodes[0].id == case["target_id"]
        target_result = connect(
            suite,
            suite_name,
            target_located.nodes[0],
            request=ConnectRequest(subject=case["target"], rel_types=schema.skill_rel_types),
            confidence=target_located.confidence,
            locate_evidence=target_located.evidence,
            optional_rel_values=schema.optional_rel_values,
        )

        gap = compute_gap(
            start_result, target_result, optional_rel_values=schema.optional_rel_values
        )
        assert supports_enumerate_paths(suite), "live suite must support enumerate_paths"
        routes = find_routes(
            suite, suite_name, start_located.nodes[0].id, target_located.nodes[0].id
        )

    assert len(gap.essential_missing) == case["expected_essential_missing"]
    assert len(gap.optional_missing) == case["expected_optional_missing"]
    assert gap.gap_too_large == case["expected_gap_too_large"]
    assert (len(routes.routes) > 0) == case["expected_has_route"]
    assert sorted(routes.warnings) == case["expected_route_warnings"]
