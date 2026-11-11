"""The TUI shows every suite's answer, not just the first one's."""

from __future__ import annotations

from talent_angels.assistant.planning import build_plan_for_capability
from talent_angels.assistant.turn import TurnOutcome
from talent_angels.contracts import AgentResult, EdgeRef, NodeRef
from talent_angels.runlog.models import RunLogRecord
from talent_angels.session.kernel import handle_line
from talent_angels.session.picker import render_picker
from talent_angels.session.store import new_session


def _node(suite: str, n: int, label: str, kind: str = "Occupation") -> NodeRef:
    return NodeRef(
        id=f"{suite}:{kind.lower()}:{n}",
        suite=suite,
        source=suite,
        source_id=str(n),
        kind=kind,
        pref_label=label,
    )


def _connect(suite: str, skills: int) -> AgentResult:
    center = _node(suite, 0, f"{suite} developer")
    nodes = [center, *(_node(suite, i, f"{suite} skill {i}", "Skill") for i in range(1, skills))]
    edges = [
        EdgeRef(type="HAS_SKILL", suite=suite, source_node_id=center.id, target_node_id=n.id)
        for n in nodes[1:]
    ]
    return AgentResult(capability="connect", suite=suite, nodes=nodes, edges=edges)


def _ambiguous(suite: str, count: int) -> AgentResult:
    return AgentResult(
        capability="locate",
        suite=suite,
        nodes=[_node(suite, i, f"{suite} match {i}") for i in range(count)],
        warnings=["ambiguous"],
        confidence=0.7,
    )


def _runner(*results: AgentResult):
    def run(question, **_kwargs):  # noqa: ANN001, ANN003, ANN202
        capability = results[0].capability
        return TurnOutcome(
            capability=capability,  # type: ignore[arg-type]
            plan=build_plan_for_capability(capability, suites=tuple(r.suite for r in results)),  # type: ignore[arg-type]
            result=results[0],
            answer="",
            record=RunLogRecord(suite="x", plan=[capability], question=question),
            results=tuple(results),
        )

    return run


def test_show_more_lists_every_suites_skills() -> None:
    state = new_session()
    runner = _runner(_connect("esco", 20), _connect("onet", 30))
    handle_line(state, "what skills does a developer need", runner=runner)

    more = handle_line(state, "show more skills", runner=runner)

    assert "esco skill 19" in more.text
    assert "onet skill 29" in more.text


def test_widening_pickers_keeps_numbers_continuous_across_suites() -> None:
    state = new_session()
    runner = _runner(_ambiguous("esco", 12), _ambiguous("onet", 12))
    handle_line(state, "developer", runner=runner)

    handle_line(state, "show more", runner=runner)

    numbers = [choice.number for choice in state.pending]
    assert numbers == list(range(1, 25))
    assert {choice.node.suite for choice in state.pending} == {"esco", "onet"}


def test_a_suites_miss_is_shown_next_to_another_suites_hit() -> None:
    state = new_session()
    miss = AgentResult(capability="connect", suite="onet", warnings=["not_found"])
    reply = handle_line(
        state, "what skills does a developer need", runner=_runner(_connect("esco", 3), miss)
    )
    assert "O*NET" in reply.text


def test_picker_markdown_keeps_group_headings_and_footer_lines_apart() -> None:
    from talent_angels.session.models import PendingChoice

    pending = [
        PendingChoice(number=1, node=_node("esco", 1, "web developer"), group_label="Web"),
        PendingChoice(number=2, node=_node("esco", 2, "app developer"), group_label="Apps"),
    ]
    text = render_picker("developer", pending, omitted=5)

    assert "1. web developer\n\n**Apps**" in text
    assert "\n\nShowing 2 of 7" in text
    assert "\n\nsource: esco" in text
