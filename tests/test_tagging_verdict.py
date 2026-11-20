"""`verdict.decide`: the full tag-v1 policy, no-LLM and llm_on rules alike."""

from __future__ import annotations

from talent_angels.tagging.choose import ValidatedChoice
from talent_angels.tagging.models import CandidateRef, SearchOutcome, Verdict
from talent_angels.tagging.verdict import decide


def _candidate(
    id_: str, *, best_rank: int = 1, hits: int = 1, alias_unconfirmed: bool = False
) -> CandidateRef:
    return CandidateRef(
        suite="esco",
        id=id_,
        source_id=f"urn:{id_}",
        kind="Skill",
        label=id_,
        method="exact_pref",
        best_rank=best_rank,
        hits=hits,
        queries=["q"],
        alias_unconfirmed=alias_unconfirmed,
    )


def _choice(fit, picks, reason="r", missing_skill=None) -> ValidatedChoice:
    return ValidatedChoice(fit=fit, picks=picks, reason=reason, missing_skill=missing_skill)


# --- rule 1: suite status ---------------------------------------------------


def test_unavailable_suite_is_not_cached_as_a_tag() -> None:
    outcome = SearchOutcome(suite="esco", status="unavailable", routed_kind=None, error="boom")
    result = decide(outcome)
    assert result.verdict == Verdict.UNAVAILABLE
    assert result.reason_code == "suite_unavailable"
    assert result.tags == []


def test_not_applicable_kind() -> None:
    outcome = SearchOutcome(suite="esco", status="n/a", routed_kind=None)
    result = decide(outcome)
    assert result.verdict == Verdict.NOT_APPLICABLE
    assert result.reason_code == "kind_not_supported"


def test_unavailable_wins_even_with_llm_on() -> None:
    outcome = SearchOutcome(suite="esco", status="unavailable", routed_kind=None)
    result = decide(outcome, llm_on=True)
    assert result.verdict == Verdict.UNAVAILABLE


# --- rule 2: no-LLM ----------------------------------------------------------


def test_no_candidates_is_a_gap() -> None:
    outcome = SearchOutcome(suite="esco", status="ok", routed_kind="skill", candidates=[])
    result = decide(outcome)
    assert result.verdict == Verdict.GAP
    assert result.reason_code == "no_candidates"
    assert result.tags == []


def test_candidates_become_review_with_top_three_alternatives() -> None:
    candidates = [_candidate(f"esco:skill:{i}") for i in range(5)]
    outcome = SearchOutcome(suite="esco", status="ok", routed_kind="skill", candidates=candidates)
    result = decide(outcome)
    assert result.verdict == Verdict.REVIEW
    assert result.reason_code == "search_only"
    assert result.tags == [candidates[0]]
    assert result.alternatives == candidates[1:4]


def test_no_candidates_is_a_gap_even_with_llm_on_and_no_model_call_needed() -> None:
    outcome = SearchOutcome(suite="esco", status="ok", routed_kind="skill", candidates=[])
    result = decide(outcome, llm_on=True)
    assert result.verdict == Verdict.GAP
    assert result.reason_code == "no_candidates"


# --- rule 3: model/JSON failure ----------------------------------------------


def test_model_failure_is_error_not_cached() -> None:
    candidates = [_candidate("esco:skill:1")]
    outcome = SearchOutcome(suite="esco", status="ok", routed_kind="skill", candidates=candidates)
    result = decide(outcome, None, llm_on=True)
    assert result.verdict == Verdict.ERROR
    assert result.reason_code == "model_failed"


# --- rule 4: fit=exact --------------------------------------------------------


def test_exact_corroborated_by_lexical_rank_is_accepted() -> None:
    candidate = _candidate("esco:skill:1")
    outcome = SearchOutcome(suite="esco", status="ok", routed_kind="skill", candidates=[candidate])
    choice = _choice("exact", [(candidate, "")])
    result = decide(outcome, choice, llm_on=True, labels=[candidate.label])
    assert result.verdict == Verdict.ACCEPTED
    assert result.reason_code == "model_exact_corroborated"
    assert result.tags == [candidate]


def test_exact_corroborated_by_hits_and_rank_without_a_label_match() -> None:
    candidate = _candidate("esco:skill:1", best_rank=3, hits=2)
    outcome = SearchOutcome(suite="esco", status="ok", routed_kind="skill", candidates=[candidate])
    choice = _choice("exact", [(candidate, "")])
    result = decide(outcome, choice, llm_on=True, labels=["something unrelated"])
    assert result.verdict == Verdict.ACCEPTED


def test_exact_uncorroborated_falls_back_to_review() -> None:
    candidate = _candidate("esco:skill:1", best_rank=9, hits=1)
    outcome = SearchOutcome(suite="esco", status="ok", routed_kind="skill", candidates=[candidate])
    choice = _choice("exact", [(candidate, "")])
    result = decide(outcome, choice, llm_on=True, labels=["totally different phrase"])
    assert result.verdict == Verdict.REVIEW
    assert result.reason_code == "model_exact_uncorroborated"


def test_alias_unconfirmed_blocks_corroboration_even_with_a_label_match() -> None:
    candidate = _candidate("esco:skill:1", alias_unconfirmed=True)
    outcome = SearchOutcome(suite="esco", status="ok", routed_kind="skill", candidates=[candidate])
    choice = _choice("exact", [(candidate, "")])
    result = decide(outcome, choice, llm_on=True, labels=[candidate.label])
    assert result.verdict == Verdict.REVIEW
    assert result.reason_code == "model_exact_uncorroborated"


def test_one_uncorroborated_pick_among_many_still_falls_back_to_review() -> None:
    good = _candidate("esco:skill:1")
    bad = _candidate("esco:skill:2", best_rank=9, hits=1)
    outcome = SearchOutcome(suite="esco", status="ok", routed_kind="skill", candidates=[good, bad])
    choice = _choice("exact", [(good, ""), (bad, "")])
    result = decide(outcome, choice, llm_on=True, labels=[good.label])
    assert result.verdict == Verdict.REVIEW


# --- rule 5: fit=partial -------------------------------------------------------


def test_partial_with_covers_is_adjacent() -> None:
    candidate = _candidate("esco:skill:1")
    outcome = SearchOutcome(suite="esco", status="ok", routed_kind="skill", candidates=[candidate])
    choice = _choice(
        "partial", [(candidate, "handles part of the task")], missing_skill="new skill"
    )
    result = decide(outcome, choice, llm_on=True)
    assert result.verdict == Verdict.ADJACENT
    assert result.reason_code == "model_partial"
    assert result.covers == {candidate.id: "handles part of the task"}
    assert result.missing_skill == "new skill"


def test_partial_without_any_covers_text_needs_widening() -> None:
    candidate = _candidate("esco:skill:1")
    outcome = SearchOutcome(suite="esco", status="ok", routed_kind="skill", candidates=[candidate])
    choice = _choice("partial", [(candidate, "")])
    result = decide(outcome, choice, llm_on=True)
    assert result.needs_widening is True


# --- rule 6: fit=none / no valid picks -----------------------------------------


def test_fit_none_requests_widening_when_not_yet_widened() -> None:
    outcome = SearchOutcome(
        suite="esco", status="ok", routed_kind="skill", candidates=[_candidate("esco:skill:1")]
    )
    choice = _choice("none", [], missing_skill="candidate skill")
    result = decide(outcome, choice, llm_on=True, widened=False)
    assert result.needs_widening is True
    assert result.verdict != Verdict.GAP


def test_fit_none_still_none_after_widening_is_a_gap() -> None:
    outcome = SearchOutcome(
        suite="esco", status="ok", routed_kind="skill", candidates=[_candidate("esco:skill:1")]
    )
    choice = _choice("none", [], missing_skill="candidate skill")
    result = decide(outcome, choice, llm_on=True, widened=True)
    assert result.verdict == Verdict.GAP
    assert result.reason_code == "model_none"
    assert result.missing_skill == "candidate skill"


def test_fit_exact_with_no_valid_picks_also_needs_widening() -> None:
    # e.g. every pick dropped by validate() (out of range / kind mismatch).
    outcome = SearchOutcome(
        suite="esco", status="ok", routed_kind="skill", candidates=[_candidate("esco:skill:1")]
    )
    choice = _choice("exact", [])
    result = decide(outcome, choice, llm_on=True, widened=False)
    assert result.needs_widening is True


def test_alternatives_exclude_picked_candidates_and_cap_at_three() -> None:
    picked = _candidate("esco:skill:1")
    rest = [_candidate(f"esco:skill:{i}") for i in range(2, 7)]
    outcome = SearchOutcome(
        suite="esco", status="ok", routed_kind="skill", candidates=[picked, *rest]
    )
    choice = _choice("exact", [(picked, "")])
    result = decide(outcome, choice, llm_on=True, labels=[picked.label])
    assert picked not in result.alternatives
    assert len(result.alternatives) == 3


def test_an_exact_pick_ranked_top_3_by_search_is_corroborated() -> None:
    """tag-v2: O*NET tasks never match a label word for word; search's top-3 agrees."""
    from talent_angels.tagging.verdict import _corroborated

    pick = CandidateRef(
        suite="onet",
        id="onet:task:1",
        source_id="1",
        kind="Task",
        label="Document test results.",
        method="hybrid_rrf",
        best_rank=1,
        hits=1,
        queries=["q"],
    )
    assert _corroborated(pick, ["record test runs"])
    assert not _corroborated(pick.model_copy(update={"best_rank": 4}), ["record test runs"])
    assert not _corroborated(pick.model_copy(update={"alias_unconfirmed": True}), [])
