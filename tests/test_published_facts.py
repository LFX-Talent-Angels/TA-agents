"""Published facts (BLS wage, outlook, education) travel from the suite node to the card."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from talent_angels.contracts import AgentResult, NodeRef
from talent_angels.skills.locate.rank import code_facts, published_facts
from talent_angels.skills.locate.resolve import node_ref


@dataclass
class _Node:
    id: str
    kind: str
    label: str
    source: str
    source_id: str
    properties: dict[str, Any] = field(default_factory=dict)


BLS_PROPS = {
    "median_wage_annual": 135980.0,
    "wage_period": "May 2025",
    "employment_change_pct": 10.2,
    "openings_annual": 95300,
    "projection_cycle": "2025-2035",
    "projection_code": "15-1252",
    "projection_aggregate": False,
    "entry_education": "Bachelor's degree",
    "work_experience": "None",
    "job_training": "None",
    "onet_soc_codes": ["15-1252.00"],  # not a published fact for the card
    "alt_labels": ["Software Engineer"],
}


def _ref(props: dict[str, Any], node_id: str = "bls:occupation:15-1252") -> NodeRef:
    return node_ref(
        _Node(node_id, "Occupation", "Software Developers", "bls", node_id[-7:], props), "bls"
    )


def test_node_ref_keeps_only_published_facts() -> None:
    ref = _ref(BLS_PROPS)
    assert ref.facts["median_wage_annual"] == 135980.0
    assert ref.facts["entry_education"] == "Bachelor's degree"
    assert "onet_soc_codes" not in ref.facts
    assert "alt_labels" not in ref.facts
    assert ref.alt_labels == ["Software Engineer"]


def test_nodes_without_facts_are_unchanged() -> None:
    assert _ref({"description": "x"}, "esco:occupation:abc").facts == {}


def test_card_lines_name_source_and_period() -> None:
    ref = _ref(BLS_PROPS)
    result = AgentResult(capability="locate", suite="bls", nodes=[ref])
    assert code_facts(result, ref) == [
        "SOC 2018 code: 15-1252",
        "Median annual wage: $135,980 (BLS, May 2025)",
        "Projected employment change 2025-2035: +10.2% (BLS)",
        "Openings per year, 2025-2035 average: 95,300 (BLS)",
        "Typical entry education: Bachelor's degree (BLS)",
        "Work experience in a related occupation: None (BLS)",
        "Typical on-the-job training: None (BLS)",
    ]


def test_aggregate_figures_say_so() -> None:
    lines = published_facts(
        _ref(
            {
                "median_wage_annual": 35800.0,
                "employment_change_pct": -3.0,
                "projection_code": "31-1120",
                "projection_aggregate": True,
            },
            "bls:occupation:31-1121",
        )
    )
    assert lines[1] == "Projected employment change: -3.0% (BLS)"
    assert lines[-1].startswith("Note: BLS publishes these figures for the combined group 31-1120")


def test_old_saved_node_refs_still_load() -> None:
    # Sessions and profiles saved before NodeRef had facts.
    old = {
        "id": "esco:occupation:x",
        "suite": "esco",
        "source": "esco",
        "source_id": "x",
        "kind": "Occupation",
        "pref_label": "nurse",
    }
    assert NodeRef.model_validate(old).facts == {}


def test_connect_card_carries_the_published_facts() -> None:
    from talent_angels.contracts import EdgeRef
    from talent_angels.session.phrase import connect_card

    center = _ref(BLS_PROPS)
    skill = NodeRef(
        id="bls:skill:mathematics",
        suite="bls",
        source="bls",
        source_id="10",
        kind="Skill",
        pref_label="Mathematics",
    )
    edge = EdgeRef(
        type="HAS_SKILL",
        suite="bls",
        source_node_id=center.id,
        target_node_id=skill.id,
        properties={"relation_type": "essential"},
    )
    card = connect_card(
        AgentResult(capability="connect", suite="bls", nodes=[center, skill], edges=[edge])
    )
    assert "Typical entry education: Bachelor's degree (BLS)" in card
    assert "Median annual wage: $135,980 (BLS, May 2025)" in card
