"""How a compare reads in chat: shared and own skills per suite, nothing bound."""

from __future__ import annotations

from talent_angels.assistant.planning import build_plan_for_capability
from talent_angels.assistant.turn import TurnOutcome
from talent_angels.contracts import AgentResult, EdgeRef, NodeRef
from talent_angels.runlog import RunLogRecord
from talent_angels.session.compare_view import COMPARE_PREVIEW, render_compare
from talent_angels.session.kernel import handle_line
from talent_angels.session.store import new_session
from talent_angels.skills.connect.compare import compare_result


def _node(suite: str, label: str, kind: str = "Skill") -> NodeRef:
    return NodeRef(
        id=f"{suite}:{kind.casefold()}:{label}",
        suite=suite,
        source=suite,
        source_id=label,
        kind=kind,
        pref_label=label,
    )


def _connect(suite: str, occupation: str, skills: dict[str, str]) -> AgentResult:
    centre = _node(suite, occupation, "Occupation")
    nodes = [_node(suite, label) for label in skills]
    edges = [
        EdgeRef(
            type="HAS_SKILL",
            suite=suite,
            source_node_id=centre.id,
            target_node_id=node.id,
            properties={"relation_type": tag},
        )
        for node, tag in zip(nodes, skills.values(), strict=True)
    ]
    return AgentResult(capability="connect", suite=suite, nodes=[centre, *nodes], edges=edges)


def _chef_vs_baker(suite: str = "esco") -> AgentResult:
    chef = _connect(suite, "chef", {"plan menus": "optional", "cook food": "essential"})
    baker = _connect(suite, "baker", {"cook food": "essential", "knead dough": "essential"})
    return compare_result(chef, baker)


def _runner(*results: AgentResult):
    def runner(question: str, **_kwargs: object) -> TurnOutcome:
        return TurnOutcome(
            capability="connect",
            plan=build_plan_for_capability("connect", suites=tuple(r.suite for r in results)),
            result=results[0],
            results=results,
            answer="ignored",
            record=RunLogRecord(suite="esco,onet", plan=["connect"], question=question),
        )

    return runner


def test_render_names_both_titles_and_each_group() -> None:
    text = render_compare(_chef_vs_baker())
    assert "**chef** vs **baker** — 1 shared skill." in text
    assert "**Shared** (1): cook food" in text
    assert "**Only chef** (1): plan menus" in text
    assert "**Only baker** (1): knead dough" in text


def test_render_lists_essential_skills_first_and_caps_each_group() -> None:
    skills = {f"optional {i}": "optional" for i in range(COMPARE_PREVIEW)}
    skills["must have"] = "essential"
    result = compare_result(_connect("esco", "a", skills), _connect("esco", "b", {}))
    line = next(row for row in render_compare(result).splitlines() if "Only a" in row)
    assert line.startswith(f"**Only a** ({COMPARE_PREVIEW + 1}): must have, optional 0")
    assert line.endswith("(+1 more)")


def test_chat_shows_a_compare_per_suite_and_binds_nothing() -> None:
    state = new_session()
    runner = _runner(_chef_vs_baker("esco"), _chef_vs_baker("onet"))

    reply = handle_line(state, "chef vs baker", runner=runner)

    assert "## ESCO" in reply.text and "## O*NET" in reply.text
    assert reply.text.count("1 shared skill") == 2
    assert state.bindings == {}


def test_single_suite_compare_is_rendered() -> None:
    reply = handle_line(new_session(), "chef vs baker", runner=_runner(_chef_vs_baker()))
    assert "**Only baker** (1): knead dough" in reply.text


def test_a_suite_still_picking_shows_its_list_next_to_the_compare() -> None:
    teachers = AgentResult(
        capability="locate",
        suite="onet",
        nodes=[
            _node("onet", "Music Teachers", "Occupation"),
            _node("onet", "Art Teachers", "Occupation"),
        ],
        warnings=["ambiguous"],
        confidence=0.7,
    )
    state = new_session()

    reply = handle_line(state, "chef vs teacher", runner=_runner(_chef_vs_baker(), teachers))

    assert "1 shared skill" in reply.text
    assert "Music Teachers" in reply.text
    assert len(state.pending) == 2


def test_it_after_a_compare_asks_which_title() -> None:
    state = new_session()
    state.bindings["esco"] = _node("esco", "nurse", "Occupation")
    handle_line(state, "chef vs baker", runner=_runner(_chef_vs_baker()))

    def no_search(question: str, **_kwargs: object) -> TurnOutcome:
        raise AssertionError("must ask, not search")

    reply = handle_line(state, "what skills does it need?", runner=no_search)

    assert state.bindings == {}
    assert reply.text.startswith("Which one do you mean: **chef** or **baker**?")


def test_tools_are_counted_apart_from_skills() -> None:
    from talent_angels.session.followup import render_connect_list

    centre = _node("onet", "Software Developers", "Occupation")
    skill = _node("onet", "Critical Thinking")
    tools = [_node("onet", f"Tool {i}", "Software") for i in range(3)]
    edges = [
        EdgeRef(
            type="HAS_SKILL",
            suite="onet",
            source_node_id=centre.id,
            target_node_id=skill.id,
            properties={"relation_type": "essential"},
        ),
        *(
            EdgeRef(
                type="USES_SOFTWARE", suite="onet", source_node_id=centre.id, target_node_id=t.id
            )
            for t in tools
        ),
    ]
    result = AgentResult(
        capability="connect", suite="onet", nodes=[centre, skill, *tools], edges=edges
    )
    assert "**Software Developers** — 1 skill and 3 tools on the map:" in render_connect_list(
        result
    )


def test_a_compare_waiting_on_a_pick_runs_after_it() -> None:
    from talent_angels.assistant.llm_plan import PlanDraft

    engineers = AgentResult(
        capability="locate",
        suite="esco",
        nodes=[
            _node("esco", "civil engineer", "Occupation"),
            _node("esco", "test engineer", "Occupation"),
        ],
        warnings=["ambiguous", "compare_side:1"],
    )
    asked: list[str] = []

    def runner(question: str, **_kwargs: object) -> TurnOutcome:
        asked.append(question)
        result = engineers if len(asked) == 1 else _chef_vs_baker()
        return TurnOutcome(
            capability="connect",
            plan=build_plan_for_capability("connect", suites=("esco",)),
            result=result,
            results=(result,),
            answer="ignored",
            record=RunLogRecord(suite="esco", plan=["connect"], question=question),
            plan_draft=PlanDraft(target="connect", subject="engineer", secondary_subject="nurse"),
        )

    state = new_session()
    handle_line(state, "compare engineer and nurse", runner=runner)
    assert state.pending_compare == {"esco": ["engineer", "nurse", "1"]}
    reply = handle_line(state, "1", runner=runner)
    assert asked[-1] == "compare civil engineer and nurse"
    assert "vs" in reply.text
    assert "vs" in reply.text
    assert state.pending_compare == {}


def _two_suite_lists(asked: list[tuple[str, dict[str, object]]]):
    from talent_angels.assistant.llm_plan import PlanDraft

    esco = AgentResult(
        capability="locate",
        suite="esco",
        nodes=[
            _node("esco", "nurse assistant", "Occupation"),
            _node("esco", "specialist nurse", "Occupation"),
        ],
        warnings=["ambiguous", "compare_side:1"],
    )
    onet = AgentResult(
        capability="locate",
        suite="onet",
        nodes=[
            _node("onet", "Registered Nurses", "Occupation"),
            _node("onet", "Nurse Practitioners", "Occupation"),
        ],
        warnings=["ambiguous", "compare_side:1"],
    )

    def runner(question: str, **kwargs: object) -> TurnOutcome:
        asked.append((question, kwargs))
        results = (esco, onet) if len(asked) == 1 else (_chef_vs_baker(str(kwargs.get("suite"))),)
        return TurnOutcome(
            capability="connect",
            plan=build_plan_for_capability("connect", suites=tuple(r.suite for r in results)),
            result=results[0],
            results=results,
            answer="ignored",
            record=RunLogRecord(suite="esco,onet", plan=["connect"], question=question),
            plan_draft=PlanDraft(target="connect", subject="nurse", secondary_subject="doctor"),
        )

    return runner


def test_a_compare_list_says_it_is_a_compare_and_how_to_pick() -> None:
    asked: list[tuple[str, dict[str, object]]] = []
    state = new_session()
    reply = handle_line(state, "compare nurse and doctor", runner=_two_suite_lists(asked))
    assert reply.text.startswith("To compare nurse and doctor, I first need to know which nurse")
    assert "e.g. 1 3" in reply.text
    assert set(state.pending_compare) == {"esco", "onet"}


def test_a_pick_in_one_suite_compares_there_and_keeps_the_other_list() -> None:
    asked: list[tuple[str, dict[str, object]]] = []
    state = new_session()
    runner = _two_suite_lists(asked)
    handle_line(state, "compare nurse and doctor", runner=runner)
    reply = handle_line(state, "2", runner=runner)
    assert asked[-1][0] == "compare specialist nurse and doctor"
    assert asked[-1][1]["suite"] == "esco"
    assert "vs" in reply.text
    assert "O*NET still needs a pick" in reply.text
    assert [c.node.suite for c in state.pending] == ["onet", "onet"]
    assert set(state.pending_compare) == {"onet"}


def test_one_pick_per_list_in_one_reply() -> None:
    asked: list[tuple[str, dict[str, object]]] = []
    state = new_session()
    runner = _two_suite_lists(asked)
    handle_line(state, "compare nurse and doctor", runner=runner)
    handle_line(state, "2 3", runner=runner)
    suites = [kwargs.get("suite") for _q, kwargs in asked[1:]]
    assert suites == ["esco", "onet"]


def test_two_picks_from_the_same_list_are_refused() -> None:
    from tests.test_chat_guided import _broad_list

    (state, _), runner = _broad_list()
    reply = handle_line(state, "1 2", runner=runner)
    assert reply.text.startswith("Pick at most one title per taxonomy")
    assert not state.bindings


def test_compare_groups_are_separate_paragraphs() -> None:
    text = render_compare(_chef_vs_baker())
    assert "\n\n**Only chef**" in text and "\n\n**Only baker**" in text
