"""JobTech is wired like every other taxonomy: aliases, heading, SSYK fact, threshold."""

from __future__ import annotations

from talent_angels.assistant.merge import suite_heading
from talent_angels.assistant.suite_select import named_suites
from talent_angels.contracts import AgentResult, NodeRef
from talent_angels.skills.locate.rank import code_facts
from talent_angels.skills.pathfind.gap import _MAX_REALISTIC_GAP_SKILLS_BY_SUITE

ATTACHED = ("bls", "esco", "jobtech", "onet", "sfia")


def test_users_can_name_jobtech() -> None:
    for question in (
        "barnmorska in JobTech",
        "what does the Swedish taxonomy say about bakers",
        "bagare enligt Arbetsförmedlingen",
        "show me job tech",
    ):
        assert named_suites(question, ATTACHED) == ("jobtech",), question


def test_sweden_alone_does_not_pick_jobtech() -> None:
    # "I moved from Sweden to the US. How much do bakers earn here?" is about BLS.
    assert named_suites("I moved from Sweden to the US, bakers pay?", ATTACHED) == ()


def test_lightcast_is_named_as_not_attached() -> None:
    from talent_angels.assistant.suite_select import named_unattached

    assert named_unattached("baker in lightcast", ATTACHED) == ("lightcast",)


def test_ssyk_code_is_not_a_published_fact() -> None:
    from talent_angels.skills.locate.rank import published_facts

    node = NodeRef(
        id="jobtech:occupation:X",
        suite="jobtech",
        source="jobtech",
        source_id="X",
        kind="Occupation",
        pref_label="Bagare/Konditor",
        facts={"ssyk_code": "7612"},
    )
    assert published_facts(node) == []


def test_heading() -> None:
    assert suite_heading("jobtech") == "JobTech"


def test_ssyk_fact_on_an_occupation() -> None:
    node = NodeRef(
        id="jobtech:occupation:XugJ_6hw_wVV",
        suite="jobtech",
        source="jobtech",
        source_id="XugJ_6hw_wVV",
        kind="Occupation",
        pref_label="Bagare/Konditor",
        facts={"ssyk_code": "7612"},
    )
    result = AgentResult(capability="locate", suite="jobtech", nodes=[node])
    assert code_facts(result, node) == ["SSYK 2012 unit group: 7612 (JobTech)"]


def test_gap_threshold_is_set() -> None:
    assert _MAX_REALISTIC_GAP_SKILLS_BY_SUITE["jobtech"] == 15
