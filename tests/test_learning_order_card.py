"""The learning-order section of the chat path card: compact, sourced, existing card intact."""

from __future__ import annotations

from talent_angels.session.i18n import set_language
from talent_angels.session.pathfind_view import render_pathfind
from talent_angels.skills.pathfind.dispatch import pathfind_one
from tests.test_roadmap import A, B, _esco


def _card() -> tuple[str, str]:
    result, _ = pathfind_one(_esco(), "esco", "analyst", "scientist")
    plain = result.model_copy(
        update={"warnings": [w for w in result.warnings if not w.startswith("learning_order:")]}
    )
    return render_pathfind(result), render_pathfind(plain)


def test_the_card_gets_a_compact_learning_order_after_the_existing_card() -> None:
    with_order, without = _card()

    assert with_order.startswith(without)  # the existing card is untouched
    section = with_order[len(without) :].strip()
    lines = section.splitlines()
    assert lines[0] == "**Learning order (learn-order-v1, ESCO):**"
    assert lines[2] == "- 1. Foundations: algebra, statistics (stated prerequisite)"
    assert lines[3] == "- 2. Core: modelling, same label, same label (+3 more)"
    assert lines[4] == "- 3. Field: field skill (stated prerequisite), domain theory"
    assert lines[5] == "- 4. Job-specific: job skill"
    assert "Only items marked as stated prerequisites" in lines[-1]


def test_a_taxonomy_with_no_gap_items_adds_no_section() -> None:
    from tests.fakes.order_suite import OrderSuite
    from tests.test_roadmap import BA, BB

    result, _ = pathfind_one(OrderSuite([BA, BB], []), "bls", "Analysts", "Scientists")

    assert not any(w.startswith("learning_order:") for w in result.warnings)
    assert "Learning order" not in render_pathfind(result)


def test_the_section_follows_the_reply_language() -> None:
    set_language("es")
    try:
        with_order, _ = _card()
    finally:
        set_language("en")

    assert "**Orden de aprendizaje (learn-order-v1, ESCO):**" in with_order
    assert "Fundamentos" in with_order


def test_the_card_never_breaks_pathfind_when_the_roadmap_fails(monkeypatch) -> None:
    from talent_angels.skills.roadmap import card

    def boom(*args: object, **kwargs: object) -> None:
        raise RuntimeError("down")

    monkeypatch.setattr(card, "roadmap_suite", boom)
    result, _ = pathfind_one(_esco(), "esco", "analyst", "scientist")
    assert not any(w.startswith("learning_order:") for w in result.warnings)
    assert result.nodes[:2] != [] and A.id and B.id
