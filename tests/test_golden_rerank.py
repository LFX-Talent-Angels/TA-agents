"""Golden eval for Locate's rerank/auto-select/ISCO-grouping layer.

Fully offline — see tests/evals/golden_rerank.json for what this eval exists
to catch and how the cases were derived and verified.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from talent_angels.contracts import AgentResult, NodeRef
from talent_angels.evals import RerankMetrics
from talent_angels.skills.locate.rank import group_and_sort_locate
from tests.fakes.taxonomy import FakeEdge, FakeNode, FakeToolResult

GOLDEN_PATH = Path(__file__).parent / "evals" / "golden_rerank.json"


def _load_cases() -> list[dict]:
    data = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    return data["cases"]


def _node(candidate: dict) -> NodeRef:
    suite = candidate["id"].split(":", 1)[0]
    return NodeRef(
        id=candidate["id"],
        suite=suite,
        source=suite,
        source_id=candidate["id"].rsplit(":", 1)[-1],
        kind="Occupation",
        pref_label=candidate["label"],
        alt_labels=list(candidate.get("alts") or []),
    )


class _GroupedSuite:
    """get_neighbors(CLASSIFIED_UNDER) — returns the case's declared ISCO parent, if any."""

    def __init__(self, candidates: list[dict], groups: dict[str, str]) -> None:
        self._parent_of = {c["id"]: c.get("group") for c in candidates}
        self._groups = groups

    def get_neighbors(self, node_id: str, rel_types: list[str] | None = None) -> FakeToolResult:
        assert rel_types == ["CLASSIFIED_UNDER"]
        group_key = self._parent_of.get(node_id)
        if group_key is None:
            return FakeToolResult()
        isco = FakeNode(
            id=f"isco:{group_key}",
            kind="ISCOGroup",
            label=self._groups[group_key],
            source="isco",
            source_id=group_key,
        )
        return FakeToolResult(
            nodes=[
                FakeNode(id=node_id, kind="Occupation", label="x", source="x", source_id="n"),
                isco,
            ],
            edges=[FakeEdge(type="CLASSIFIED_UNDER", from_id=node_id, to_id=isco.id)],
        )


def _run(case: dict) -> AgentResult:
    nodes = [_node(c) for c in case["candidates"]]
    suite_name = nodes[0].suite
    result = AgentResult(capability="locate", suite=suite_name, nodes=nodes, confidence=0.7)
    suite = _GroupedSuite(case["candidates"], case["groups"])
    return group_and_sort_locate(suite, result, case["query"], suite_name=suite_name)


def _group_label(ranked: AgentResult, top_id: str) -> str | None:
    for edge in ranked.edges:
        if edge.source_node_id == top_id:
            label = edge.properties.get("group_label")
            return str(label) if label is not None else None
    return None


@pytest.mark.parametrize("case", _load_cases(), ids=lambda c: c["name"])
def test_golden_rerank_case(case: dict) -> None:
    ranked = _run(case)
    actual_ambiguous = "ambiguous" in ranked.warnings

    assert actual_ambiguous == case["expected_ambiguous"], case["name"]

    if case["expected_ambiguous"]:
        expected_ordered = case.get("expected_ordered_ids")
        if expected_ordered is not None:
            assert [n.id for n in ranked.nodes] == expected_ordered, case["name"]
        return

    assert len(ranked.nodes) == 1, case["name"]
    assert ranked.nodes[0].id == case["expected_top_id"], case["name"]
    assert ranked.confidence == case["expected_confidence"], case["name"]
    extra_warning = f"also_matched:{case['expected_extra']}"
    assert extra_warning in ranked.warnings, case["name"]
    assert _group_label(ranked, ranked.nodes[0].id) == case["expected_group_label"], case["name"]


def test_golden_rerank_metrics() -> None:
    """The aggregate regression gate: every decision and every winner must be right."""
    metrics = RerankMetrics()
    for case in _load_cases():
        ranked = _run(case)
        actual_ambiguous = "ambiguous" in ranked.warnings
        actual_top_id = ranked.nodes[0].id if not actual_ambiguous and ranked.nodes else None
        actual_group_label = (
            _group_label(ranked, actual_top_id) if actual_top_id is not None else None
        )
        metrics.observe(
            expected_ambiguous=case["expected_ambiguous"],
            actual_ambiguous=actual_ambiguous,
            expected_top_id=case["expected_top_id"],
            actual_top_id=actual_top_id,
            expected_group_label=case["expected_group_label"],
            actual_group_label=actual_group_label,
        )

    result = metrics.as_dict()
    assert result["decision_accuracy"] == 1.0
    assert result["decision_questions"] == 6
    assert result["winner_accuracy"] == 1.0
    assert result["winner_questions"] == 3
    assert result["group_label_accuracy"] == 1.0
    assert result["group_label_questions"] == 1
