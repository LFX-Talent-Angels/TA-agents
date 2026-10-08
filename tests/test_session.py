import pytest

from talent_angels.contracts import AgentResult, EdgeRef, NodeRef
from talent_angels.session.budget import model_view
from talent_angels.session.commands import UnknownCommand, parse_command
from talent_angels.session.copy import ADVICE_REFUSE, GREETING, WELCOME
from talent_angels.session.followup import is_expand_list, parse_skill_mention, skill_from_connect
from talent_angels.session.models import LastBinding, TranscriptLine
from talent_angels.session.picker import bind_pick, choices_from_result, render_picker
from talent_angels.session.router import route_line
from talent_angels.session.store import (
    clear_conversation,
    load_last,
    load_session,
    new_session,
    save_session,
)


def _occ(label: str, n: int = 1) -> NodeRef:
    return NodeRef(
        id=f"esco:occupation:{n}",
        suite="esco",
        source="esco",
        source_id=f"src-{n}",
        kind="Occupation",
        pref_label=label,
    )


def test_save_and_resume_restores_binding_without_transcript_loss(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("TA_SESSIONS_DIR", str(tmp_path / "sessions"))
    state = new_session()
    state.name = "demo-nurse"

    state.transcript.extend(
        [
            TranscriptLine(role="user", text="hi", ts="t0"),
            TranscriptLine(role="assistant", text="Hey. I'm here.", ts="t1"),
        ]
    )
    state.binding = LastBinding(node=_occ("nurse responsible for general care"))
    path = save_session(state, name="demo-nurse")
    assert path.is_dir()
    loaded = load_session("demo-nurse")
    assert loaded.binding is not None
    assert loaded.binding.node.id == "esco:occupation:1"
    assert loaded.transcript[0].text == "hi"
    assert load_last().name == "demo-nurse"


def test_clear_drops_binding_and_transcript(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("TA_SESSIONS_DIR", str(tmp_path / "sessions"))
    state = new_session()

    state.transcript.append(TranscriptLine(role="user", text="developer", ts="t0"))
    state.binding = LastBinding(node=_occ("software developer"))
    cleared = clear_conversation(state)
    assert cleared.transcript == []
    assert cleared.binding is None
    assert cleared.pending == []


def test_parse_command_table() -> None:
    assert parse_command("/help").name == "help"
    assert parse_command("/quit").name == "quit"
    assert parse_command("/exit").name == "quit"
    assert parse_command("/save demo-nurse").name == "save"
    assert parse_command("/save demo-nurse").argument == "demo-nurse"
    assert parse_command("/save").argument is None
    assert parse_command("/resume last").argument == "last"
    assert parse_command("/resume").argument == "last"
    assert parse_command("/clear").name == "clear"
    assert parse_command(" /help ").name == "help"
    assert parse_command("developer") is None
    with pytest.raises(UnknownCommand):
        parse_command("/unknown")


def test_router_classifies_non_map_lines() -> None:
    assert route_line("hi").kind == "greet"
    assert route_line("Hello!").kind == "greet"
    assert route_line("thanks").kind == "greet"
    assert route_line("what can you do").kind == "help_plain"
    assert route_line("How can you help me ?").kind == "help_plain"
    assert route_line("how can you help me").kind == "help_plain"
    assert route_line("what do you do").kind == "help_plain"
    assert route_line("should I learn Python first?").kind == "advice"
    assert route_line("1").kind == "pick" and route_line("1").pick == 1
    assert route_line("3.").kind == "pick" and route_line("3.").pick == 3
    assert route_line("the first one").kind == "pick" and route_line("the first one").pick == 1
    assert route_line("show O*NET").kind == "show_suite"
    assert route_line("show all").kind == "show_suite"
    assert route_line("developer").kind == "map"
    assert route_line("what are the essential skills?").kind == "map"
    assert route_line("list all jobs").kind == "catalogue"
    assert route_line("list all occupations").kind == "catalogue"
    assert route_line("show all occupations").kind == "catalogue"
    assert route_line("/help").kind == "command"


def test_expand_list_does_not_steal_catalogue_asks() -> None:
    assert is_expand_list("list them all")
    assert is_expand_list("yes list them all")
    assert is_expand_list("list all skills")
    assert is_expand_list("show me the rest")
    assert not is_expand_list("list all jobs")
    assert not is_expand_list("list all occupations")
    assert not is_expand_list("show all occupations")
    assert not is_expand_list("list all the jobs")
    assert is_expand_list("show full list")
    assert is_expand_list("can you list all matches for developer?")


def test_welcome_is_not_a_taxonomy_mascot() -> None:
    blob = (WELCOME + GREETING + ADVICE_REFUSE).lower()
    assert "esco desk" not in blob
    assert "i'm the esco" not in blob
    assert "i am the esco" not in blob


def _node(i: int, label: str) -> NodeRef:
    return NodeRef(
        id=f"esco:occupation:{i}",
        suite="esco",
        source="esco",
        source_id=f"s{i}",
        kind="Occupation",
        pref_label=label,
    )


def test_choices_are_stable_and_bind_returns_that_id() -> None:
    result = AgentResult(
        capability="locate",
        suite="esco",
        nodes=[_node(1, "software developer"), _node(2, "web developer")],
        warnings=["ambiguous"],
    )
    pending = choices_from_result(result)
    assert [c.number for c in pending] == [1, 2]
    assert bind_pick(pending, 1).id == "esco:occupation:1"
    assert bind_pick(pending, 2).pref_label == "web developer"


def test_picker_markdown_does_not_search_and_cites_quietly() -> None:
    pending = choices_from_result(
        AgentResult(
            capability="locate",
            suite="esco",
            nodes=[_node(1, "software developer"), _node(2, "web developer")],
            warnings=["ambiguous"],
        )
    )
    text = render_picker("developer", pending, omitted=19)
    assert "1." in text and "software developer" in text
    assert "won't pick" in text.casefold()
    assert text.lower().count("esco") <= 1  # quiet source line only
    assert "19 more" in text


def test_picker_renders_classified_under_headings() -> None:
    pending = choices_from_result(
        AgentResult(
            capability="locate",
            suite="esco",
            nodes=[_node(1, "software developer"), _node(2, "web developer")],
            edges=[
                EdgeRef(
                    type="CLASSIFIED_UNDER",
                    suite="esco",
                    source_node_id="esco:occupation:1",
                    target_node_id="esco:isco:1",
                    properties={"group_label": "Software developers"},
                ),
                EdgeRef(
                    type="CLASSIFIED_UNDER",
                    suite="esco",
                    source_node_id="esco:occupation:2",
                    target_node_id="esco:isco:1",
                    properties={"group_label": "Software developers"},
                ),
            ],
            warnings=["ambiguous"],
        )
    )
    text = render_picker("developer", pending, omitted=0)
    assert "**Software developers**" in text
    assert text.index("Software developers") < text.index("1. software developer")


def test_model_view_is_current_plus_binding_not_the_skill_dump() -> None:
    state = new_session()
    skill_blob = (
        "Essential skills include computer programming, databases, "
        "and software development methodologies."
    )
    for i in range(20):
        state.transcript.append(
            TranscriptLine(role="assistant", text=f"{skill_blob} #{i}", ts=f"t{i}")
        )
    view = model_view(state, "what are the essential skills?")
    assert "what are the essential skills?" in view
    assert view.count("computer programming") == 0
    assert "Previous user:" not in view
    assert view.strip() == "what are the essential skills?"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("tell me about skill 11", 11),
        ("skill 11", 11),
        ("what is skill 11", 11),
        ("skill #11", 11),
        ("tell me about skill #11", 11),
        ("tell me about skill 11 please", 11),
        ("Skill 11.", 11),
        ("WHAT IS SKILL 3?", 3),
        ("11", None),
        ("11.", None),
        ("essential skills", None),
        ("developer", None),
        ("skill", None),
        ("skills 11", None),
    ],
)
def test_parse_skill_mention(text: str, expected: int | None) -> None:
    assert parse_skill_mention(text) == expected


def test_skill_from_connect_is_1_based_among_neighbors() -> None:
    occupation = NodeRef(
        id="test:occupation:1",
        suite="test",
        source="test",
        source_id="src-1",
        kind="Occupation",
        pref_label="web developer",
    )
    skills = [
        NodeRef(
            id=f"test:skill:{i}",
            suite="test",
            source="test",
            source_id=f"src-skill-{i}",
            kind="Skill",
            pref_label=f"skill {i}",
        )
        for i in range(1, 13)
    ]
    result = AgentResult(capability="connect", suite="test", nodes=[occupation, *skills])
    focused = skill_from_connect(result, 11)
    assert focused is result.nodes[11]
    assert focused.pref_label == "skill 11"
    with pytest.raises(ValueError):
        skill_from_connect(result, 99)
    with pytest.raises(ValueError):
        skill_from_connect(result, 0)


def test_connect_list_tags_essential_optional() -> None:
    from talent_angels.session.followup import render_connect_list

    occupation = NodeRef(
        id="test:occupation:1",
        suite="test",
        source="test",
        source_id="src-1",
        kind="Occupation",
        pref_label="web developer",
    )
    essential = NodeRef(
        id="test:skill:e",
        suite="test",
        source="test",
        source_id="e",
        kind="Skill",
        pref_label="computer programming",
    )
    optional = NodeRef(
        id="test:skill:o",
        suite="test",
        source="test",
        source_id="o",
        kind="Skill",
        pref_label="Joomla",
    )
    listed = render_connect_list(
        AgentResult(
            capability="connect",
            suite="test",
            nodes=[occupation, essential, optional],
            edges=[
                EdgeRef(
                    type="HAS_SKILL",
                    suite="test",
                    source_node_id=occupation.id,
                    target_node_id=essential.id,
                    properties={"relation_type": "essential"},
                ),
                EdgeRef(
                    type="HAS_SKILL",
                    suite="test",
                    source_node_id=occupation.id,
                    target_node_id=optional.id,
                    properties={"relation_type": "optional"},
                ),
            ],
        )
    )
    assert "computer programming (essential)" in listed
    assert "Joomla (optional)" in listed


def test_model_view_includes_bound_occupation_id() -> None:
    state = new_session()
    state.binding = LastBinding(node=_occ("software developer", n=42))
    state.transcript.append(
        TranscriptLine(
            role="assistant",
            text="Essential skills: computer programming, SQL.",
            ts="t0",
        )
    )
    view = model_view(state, "what are the essential skills?")
    assert "what are the essential skills?" in view
    assert "Bound occupation: software developer (esco:occupation:42)" in view
    assert "computer programming" not in view
