"""Merged answers keep suite identity; they never blend node ids."""

from __future__ import annotations

from talent_angels.assistant.merge import merge_answers, suite_heading
from talent_angels.contracts import AgentResult, NodeRef


def _node(*, suite: str, node_id: str, label: str) -> NodeRef:
    return NodeRef(
        id=node_id,
        suite=suite,
        source=suite,
        source_id=node_id,
        kind="Occupation",
        pref_label=label,
    )


def test_merge_labels_each_suite_and_keeps_both_ids() -> None:
    esco = AgentResult(
        capability="locate",
        suite="esco",
        nodes=[_node(suite="esco", node_id="esco:occupation:dev", label="software developer")],
        confidence=0.95,
    )
    onet = AgentResult(
        capability="locate",
        suite="onet",
        nodes=[
            _node(
                suite="onet",
                node_id="onet:occupation:15-1252.00",
                label="Software Developers",
            )
        ],
        confidence=0.95,
    )

    answer = merge_answers((esco, onet))

    assert answer.startswith("ESCO · ")
    assert "O*NET · " in answer
    assert "esco:occupation:dev" in answer
    assert "onet:occupation:15-1252.00" in answer
    assert answer.count("\n\n") == 1


def test_unreachable_suites_surface_as_warnings() -> None:
    answer = merge_answers((), extra_warnings=("suite_unavailable:onet",))
    assert "No attached taxonomy was reachable" in answer
    assert "suite_unavailable:onet" in answer


def test_suite_heading_for_known_and_future_suites() -> None:
    assert suite_heading("esco") == "ESCO"
    assert suite_heading("onet") == "O*NET"
    assert suite_heading("sfia") == "SFIA"
