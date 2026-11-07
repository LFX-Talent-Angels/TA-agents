"""Code-written sentences follow the user's language."""

from __future__ import annotations

from talent_angels.assistant.llm_plan import PlanDraft
from talent_angels.assistant.planning import build_plan_for_capability
from talent_angels.assistant.turn import TurnOutcome
from talent_angels.contracts import AgentResult, NodeRef
from talent_angels.runlog import RunLogRecord
from talent_angels.session.i18n import _EN, _ES, language, set_language, t
from talent_angels.session.kernel import handle_line
from talent_angels.session.store import load_session, new_session, save_session
from talent_angels.tui.render import parse_list_reply


def _node(label: str) -> NodeRef:
    return NodeRef(
        id=f"esco:{label}",
        suite="esco",
        source="esco",
        source_id=label,
        kind="Occupation",
        pref_label=label,
    )


def _runner(lang: str | None):
    nurses = AgentResult(
        capability="locate",
        suite="esco",
        nodes=[_node("nurse assistant"), _node("specialist nurse")],
        warnings=["ambiguous"],
    )

    def runner(question: str, **_kwargs: object) -> TurnOutcome:
        return TurnOutcome(
            capability="locate",
            plan=build_plan_for_capability("locate", suites=("esco",)),
            result=nurses,
            answer="ignored",
            record=RunLogRecord(suite="esco", plan=["locate"], question=question),
            plan_draft=PlanDraft(target="locate", subject="nurse", language=lang),
        )

    return runner


def test_every_sentence_has_a_spanish_version() -> None:
    assert set(_EN) == set(_ES)
    for key, text in _EN.items():
        assert text.count("{") == _ES[key].count("{"), key


def test_unknown_languages_fall_back_to_english() -> None:
    set_language("xx")
    assert language() == "en"
    set_language("es-MX")
    assert language() == "es"
    assert t("bound", title="enfermera") == "Seleccionado: enfermera."
    set_language("en")


def test_a_spanish_question_gets_a_spanish_list() -> None:
    state = new_session()
    reply = handle_line(state, "enfermero", runner=_runner("es"))
    assert reply.text.startswith(
        'Entendí tu solicitud como "nurse". ESCO tiene 2 títulos que coinciden.'
    )
    assert "Elige el que buscas" in reply.text
    assert "No elegiré por ti" in reply.text
    assert state.language == "es"
    picked = handle_line(state, "1", runner=_runner("es"))
    assert picked.text == "Seleccionado: nurse assistant."


def test_the_language_is_kept_in_a_saved_session() -> None:
    state = new_session()
    handle_line(state, "enfermero", runner=_runner("es"))
    state.name = "es-session"
    save_session(state)
    assert load_session("es-session").language == "es"


def test_an_english_question_after_spanish_switches_back() -> None:
    state = new_session()
    handle_line(state, "enfermero", runner=_runner("es"))
    reply = handle_line(state, "nurse", runner=_runner("en"))
    assert "Pick the one you mean" in reply.text
    assert state.language == "en"


def test_the_tui_reads_a_spanish_skill_list_as_a_table() -> None:
    text = (
        "**enfermera** — 2 habilidades y 1 herramienta en el mapa:\n\n"
        "1. cuidar (essential)\n2. medir (essential)\n3. Excel (tool)\n"
    )
    parsed = parse_list_reply(text)
    assert len(parsed.rows) == 3
    assert "habilidades" in parsed.intro
