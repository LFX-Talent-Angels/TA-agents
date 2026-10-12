"""`verdict.decide`: the no-LLM rules of policy tag-v1 (rules 1-2)."""

from __future__ import annotations

import pytest

from talent_angels.tagging.models import CandidateRef, SearchOutcome, Verdict
from talent_angels.tagging.verdict import decide


def _candidate(id_: str) -> CandidateRef:
    return CandidateRef(
        suite="esco",
        id=id_,
        source_id=f"urn:{id_}",
        kind="Skill",
        label=id_,
        method="exact_pref",
        best_rank=1,
        hits=1,
        queries=["q"],
    )


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


def test_llm_on_raises_not_implemented_in_pr2() -> None:
    outcome = SearchOutcome(suite="esco", status="ok", routed_kind="skill", candidates=[])
    with pytest.raises(NotImplementedError):
        decide(outcome, llm_on=True)
