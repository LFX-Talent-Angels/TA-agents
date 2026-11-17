"""BLS is wired like every other taxonomy: aliases, heading, code fact, gap threshold."""

from __future__ import annotations

from talent_angels.assistant.merge import suite_heading
from talent_angels.assistant.suite_select import named_suites, select_suites
from talent_angels.contracts import AgentResult, NodeRef
from talent_angels.skills.locate.rank import code_facts
from talent_angels.skills.pathfind.gap import _MAX_REALISTIC_GAP_SKILLS_BY_SUITE

ATTACHED = ("bls", "esco", "onet", "sfia")


def test_users_can_name_bls() -> None:
    for question in (
        "web developer in BLS",
        "what does the Bureau of Labor Statistics say about nurses",
        "show me the US BLS view",
    ):
        assert named_suites(question, ATTACHED) == ("bls",), question
    assert select_suites(available=ATTACHED, question="nurse in bls") == ("bls",)
    # "bls" inside another word is not a mention.
    assert named_suites("blsomething", ATTACHED) == ()


def test_heading() -> None:
    assert suite_heading("bls") == "BLS"


def test_soc_code_fact_comes_from_the_id() -> None:
    node = NodeRef(
        id="bls:occupation:15-1252",
        suite="bls",
        source="bls",
        source_id="15-1252",
        kind="Occupation",
        pref_label="Software Developers",
    )
    result = AgentResult(capability="locate", suite="bls", nodes=[node])
    assert code_facts(result, node) == ["SOC 2018 code: 15-1252"]


def test_gap_threshold_is_set_for_bls() -> None:
    assert _MAX_REALISTIC_GAP_SKILLS_BY_SUITE["bls"] == 8
