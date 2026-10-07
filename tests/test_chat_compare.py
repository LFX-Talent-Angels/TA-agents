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
