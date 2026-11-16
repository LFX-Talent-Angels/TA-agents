"""'What can a X move to?': the job-to-job moves a taxonomy publishes (BX-10)."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import pytest

from talent_angels.assistant.transitions import (
    NO_DATA,
    WARNING,
    is_transition_question,
    transitions_for,
)
from talent_angels.contracts import AgentResult, NodeRef
from talent_angels.session.followup import render_connect_list
from talent_angels.suites import SuiteRegistry
from talent_angels.suites.registry import SuiteRuntime
from tests.fakes.taxonomy import FakeEdge, FakeNode, FakeToolResult

ELEV = NodeRef(
    id="jobtech:occupation:E",
    suite="jobtech",
    source="jobtech",
    source_id="E",
    kind="Occupation",
    pref_label="Elevassistent",
)


@pytest.mark.parametrize(
    "question",
    [
        "what can an Elevassistent move to?",
        "what other jobs could they switch to?",
        "similar jobs to nurse",
        "vilka yrken kan en elevassistent byta till?",
    ],
)
def test_move_questions(question: str) -> None:
    assert is_transition_question(question)


@pytest.mark.parametrize("question", ["what skills does a baker need?", "baker in jobtech"])
def test_other_questions(question: str) -> None:
    assert not is_transition_question(question)


class _Jobtech:
    def get_neighbors(self, node_id: str, rel_types: list[str] | None = None) -> FakeToolResult:
        def occ(cid: str, label: str) -> FakeNode:
            return FakeNode(
                f"jobtech:occupation:{cid}", "Occupation", label, source="jobtech", source_id=cid
            )

        nodes = [
            occ("E", "Elevassistent"),
            occ("S", "Skolvärd"),
            occ("B", "Barnskötare"),
            occ("F", "Fritidsledare"),
        ]
        edges = [
            FakeEdge(
                type="SUBSTITUTES",
                from_id="jobtech:occupation:E",
                to_id="jobtech:occupation:B",
                properties={"substitutability_pct": 25},
            ),
            FakeEdge(
                type="SUBSTITUTES",
                from_id="jobtech:occupation:E",
                to_id="jobtech:occupation:S",
                properties={"substitutability_pct": 75},
            ),
            # Incoming: Fritidsledare can move into Elevassistent, not the other way.
            FakeEdge(
                type="SUBSTITUTES",
                from_id="jobtech:occupation:F",
                to_id="jobtech:occupation:E",
                properties={"substitutability_pct": 25},
            ),
        ]
        result = FakeToolResult()
        result.nodes = nodes  # type: ignore[attr-defined]
        result.edges = edges  # type: ignore[attr-defined]
        return result


def _registry(suite: object, name: str = "jobtech") -> SuiteRegistry:
    @contextmanager
    def _open() -> Iterator[SuiteRuntime]:
        yield SuiteRuntime(name=name, suite=suite, health_check=lambda: True)  # type: ignore[arg-type]

    return SuiteRegistry({name: _open}, default=name)


def test_outgoing_moves_strongest_first() -> None:
    located = AgentResult(capability="locate", suite="jobtech", nodes=[ELEV], confidence=0.95)
    moved = transitions_for(_registry(_Jobtech()), located)
    assert moved is not None and WARNING in moved.warnings
    assert [n.pref_label for n in moved.nodes] == ["Elevassistent", "Skolvärd", "Barnskötare"]
    text = render_connect_list(moved)
    assert "3 jobs" not in text and "2 jobs to move to (JobTech)" in text
    assert "1. Skolvärd (75%)" in text and "2. Barnskötare (25%)" in text


def test_a_suite_without_moves_says_so() -> None:
    bakers = NodeRef(
        id="bls:occupation:51-3011",
        suite="bls",
        source="bls",
        source_id="51-3011",
        kind="Occupation",
        pref_label="Bakers",
    )
    located = AgentResult(capability="locate", suite="bls", nodes=[bakers])
    moved = transitions_for(_registry(object(), "bls"), located)
    assert moved is not None and NO_DATA in moved.warnings
    assert render_connect_list(moved) == "BLS publishes no job-to-job moves for **Bakers**."


def test_a_pick_list_is_left_alone() -> None:
    listed = AgentResult(
        capability="locate", suite="jobtech", nodes=[ELEV, ELEV], warnings=["ambiguous"]
    )
    assert transitions_for(_registry(_Jobtech()), listed) is None


def test_a_filtered_skill_list_says_how_many_in_all() -> None:
    # BX-19: "1 skill on the map" for Bakers, who have 8.
    skill = NodeRef(
        id="bls:skill:detail-oriented",
        suite="bls",
        source="bls",
        source_id="6",
        kind="Skill",
        pref_label="Detail oriented",
    )
    bakers = NodeRef(
        id="bls:occupation:51-3011",
        suite="bls",
        source="bls",
        source_id="51-3011",
        kind="Occupation",
        pref_label="Bakers",
    )
    result = AgentResult(
        capability="connect", suite="bls", nodes=[bakers, skill], warnings=["filtered:essential:8"]
    )
    assert "**Bakers** — 1 essential (of 8 in all) on the map:" in render_connect_list(result)
