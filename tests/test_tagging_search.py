"""`search.py`: query building, kind routing, merge/re-rank, failure handling."""

from __future__ import annotations

from talent_angels.tagging.models import TagUnit
from talent_angels.tagging.search import Router, queries_for, search_unit
from tests.fakes.tagging_suite import Answer, TableSuite


def _unit(**overrides: object) -> TagUnit:
    base: dict[str, object] = {
        "unit_key": "k1",
        "doc_key": "doc",
        "row_id": "1",
        "kind": "performance",
        "role": "leaf",
        "text": "raw text",
        "clean_text": "define acceptance criteria",
        "context_path": [],
        "group_key": "g",
        "seed_phrases": [],
    }
    base.update(overrides)
    return TagUnit(**base)  # type: ignore[arg-type]


# --- queries_for ------------------------------------------------------------


def test_queries_for_falls_back_to_clean_text() -> None:
    unit = _unit()
    assert queries_for(unit) == ["define acceptance criteria"]


def test_queries_for_uses_seed_phrases_when_present() -> None:
    unit = _unit(seed_phrases=["Prompt Architect", "Prompt Engineer"])
    assert queries_for(unit) == ["Prompt Architect", "Prompt Engineer"]


def test_queries_for_strips_leading_code() -> None:
    unit = _unit(seed_phrases=["2143 - Define acceptance criteria"])
    assert queries_for(unit) == ["Define acceptance criteria"]


def test_queries_for_drops_pure_code_queries() -> None:
    unit = _unit(seed_phrases=["2143", "12.3-4", "1234 Software Developer"])
    assert queries_for(unit) == ["Software Developer"]


def test_queries_for_dedupes_casefold_keeping_order() -> None:
    unit = _unit(seed_phrases=["Prompt Architect", "prompt architect", "Prompt Engineer"])
    assert queries_for(unit) == ["Prompt Architect", "Prompt Engineer"]


def test_queries_for_trims_to_200_chars() -> None:
    long_phrase = "a" * 250
    unit = _unit(seed_phrases=[long_phrase])
    [query] = queries_for(unit)
    assert len(query) == 200


def test_queries_for_with_labels_and_description() -> None:
    unit = _unit()
    result = queries_for(unit, labels=["skill a", "skill b"], description="a longer description")
    assert result == ["skill a", "skill b", "a longer description"]


# --- Router -------------------------------------------------------------


def test_router_falls_through_unknown_kind() -> None:
    suite = TableSuite(
        "fake",
        {("query", "task"): Answer(hits=[("fake:task:1", "Task", "Task One", "exact_pref")])},
        unsupported_kinds=frozenset({"skill"}),
    )
    router = Router()
    routed = router.route("fake", suite.search_nodes, "performance", ["skill", "task"], "query")
    assert routed == ["task"]


def test_router_returns_none_when_every_kind_unsupported() -> None:
    suite = TableSuite("fake", {}, unsupported_kinds=frozenset({"skill", "task"}))
    router = Router()
    routed = router.route("fake", suite.search_nodes, "performance", ["skill", "task"], "query")
    assert routed == []


def test_router_caches_per_suite_and_unit_kind() -> None:
    suite = TableSuite("fake", {}, unsupported_kinds=frozenset({"skill"}))
    router = Router()
    router.route("fake", suite.search_nodes, "performance", ["task"], "query")
    calls_after_first = len(suite.calls)
    router.route("fake", suite.search_nodes, "performance", ["task"], "query")
    assert len(suite.calls) == calls_after_first


def test_router_not_found_still_routes_the_kind() -> None:
    """`not_found` means the kind is supported — distinct from `unknown_kind`."""
    suite = TableSuite("fake", {})
    router = Router()
    routed = router.route("fake", suite.search_nodes, "performance", ["skill"], "query")
    assert routed == ["skill"]


def test_router_keeps_every_supported_kind() -> None:
    """O*NET knowledge rows must reach skills/tasks too, not only its 33 knowledge areas."""
    suite = TableSuite("fake", {})
    router = Router()
    routed = router.route("fake", suite.search_nodes, "knowledge", ["knowledge", "skill"], "q")
    assert routed == ["knowledge", "skill"]


# --- search_unit ----------------------------------------------------------


def test_search_unit_reranks_lexical_results() -> None:
    table = {
        ("acceptance criteria", "skill"): Answer(
            hits=[
                (
                    "esco:skill:far",
                    "Skill",
                    "something about acceptance of criteria broadly",
                    "contains",
                ),
                ("esco:skill:near", "Skill", "acceptance criteria", "contains"),
            ]
        ),
    }
    suite = TableSuite("esco", table)
    outcome = search_unit(suite.search_nodes, "esco", _unit(), ["acceptance criteria"], "skill")
    assert outcome.status == "ok"
    assert outcome.candidates[0].id == "esco:skill:near"


def test_search_unit_keeps_hybrid_order() -> None:
    table = {
        ("acceptance criteria", "skill"): Answer(
            hits=[
                (
                    "esco:skill:far",
                    "Skill",
                    "something about acceptance of criteria broadly",
                    "hybrid_rrf",
                ),
                ("esco:skill:near", "Skill", "acceptance criteria", "hybrid_rrf"),
            ]
        ),
    }
    suite = TableSuite("esco", table)
    outcome = search_unit(suite.search_nodes, "esco", _unit(), ["acceptance criteria"], "skill")
    assert [c.id for c in outcome.candidates] == ["esco:skill:far", "esco:skill:near"]


def test_search_unit_merges_hits_across_queries() -> None:
    table = {
        ("q1", "skill"): Answer(hits=[("esco:skill:a", "Skill", "a", "exact_pref")]),
        ("q2", "skill"): Answer(hits=[("esco:skill:a", "Skill", "a", "exact_pref")]),
    }
    suite = TableSuite("esco", table)
    outcome = search_unit(suite.search_nodes, "esco", _unit(), ["q1", "q2"], "skill")
    assert len(outcome.candidates) == 1
    candidate = outcome.candidates[0]
    assert candidate.hits == 2
    assert set(candidate.queries) == {"q1", "q2"}
    assert candidate.best_rank == 1


def test_search_unit_filters_by_occupation_kind() -> None:
    table = {
        ("q", "occupation"): Answer(
            hits=[
                ("esco:occupation:dev", "Occupation", "developer", "exact_pref"),
                ("esco:skill:python", "Skill", "python", "exact_pref"),
            ]
        ),
    }
    suite = TableSuite("esco", table)
    outcome = search_unit(suite.search_nodes, "esco", _unit(), ["q"], "occupation")
    assert [c.kind for c in outcome.candidates] == ["Occupation"]


def test_search_unit_filters_out_occupation_kind_for_non_occupation_routes() -> None:
    table = {
        ("q", "skill"): Answer(
            hits=[
                ("esco:occupation:dev", "Occupation", "developer", "exact_pref"),
                ("esco:skill:python", "Skill", "python", "exact_pref"),
            ]
        ),
    }
    suite = TableSuite("esco", table)
    outcome = search_unit(suite.search_nodes, "esco", _unit(), ["q"], "skill")
    assert [c.kind for c in outcome.candidates] == ["Skill"]


def test_search_unit_flags_alias_unconfirmed() -> None:
    table = {
        ("q", "skill"): Answer(
            hits=[("esco:skill:a", "Skill", "a", "exact_alt")],
            warnings=["alias_unconfirmed"],
        ),
    }
    suite = TableSuite("esco", table)
    outcome = search_unit(suite.search_nodes, "esco", _unit(), ["q"], "skill")
    assert outcome.candidates[0].alias_unconfirmed is True


def test_search_unit_unavailable_on_exception_hides_driver_message() -> None:
    suite = TableSuite("esco", {}, raise_on=frozenset({"q"}), raise_as=ConnectionError)
    outcome = search_unit(suite.search_nodes, "esco", _unit(), ["q"], "skill")
    assert outcome.status == "unavailable"
    assert outcome.error == "ConnectionError"
    assert outcome.candidates == []


def test_search_unit_merges_hits_across_routed_kinds() -> None:
    table = {
        ("q", "knowledge"): Answer(hits=[("onet:k:1", "Knowledge", "Computers", "exact_pref")]),
        ("q", "skill"): Answer(hits=[("onet:s:1", "Skill", "Programming", "exact_pref")]),
    }
    suite = TableSuite("onet", table)
    outcome = search_unit(suite.search_nodes, "onet", _unit(), ["q"], ["knowledge", "skill"])
    assert {c.id for c in outcome.candidates} == {"onet:k:1", "onet:s:1"}
    assert outcome.routed_kind == "knowledge+skill"
