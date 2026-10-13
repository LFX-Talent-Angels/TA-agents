"""Review loop (PR4): reading a workbook's Decision/Hint back, and applying it once.

`read_decisions` is exercised against real `openpyxl` worksheets (built by
hand, not by `write_workbook`, so the header-by-name scan is tested against
layouts it didn't produce itself — a banner row, out-of-order rows). `apply_
decisions` is exercised against a plain `Ledger` + `ReviewDecision`, which is
the layer the runner actually calls.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import openpyxl

from talent_angels.tagging.ledger import Ledger
from talent_angels.tagging.models import CandidateRef, TagRecord, Verdict, Versions
from talent_angels.tagging.review import (
    RetryRequest,
    ReviewAction,
    ReviewDecision,
    apply_decisions,
    read_decisions,
    review_key,
)

_VERSIONS = Versions(
    pipeline="tag-1", policy="tag-v2", prompts="rw-1/ch-3", ta_taxonomies="unknown", model="none"
)


def _candidate(id_: str, *, label: str | None = None) -> CandidateRef:
    return CandidateRef(
        suite="esco",
        id=id_,
        source_id=f"urn:{id_}",
        kind="Skill",
        label=label or id_,
        method="exact_pref",
        best_rank=1,
        hits=1,
        queries=["q"],
    )


def _record(
    unit_key: str = "u1",
    suite: str = "esco",
    *,
    fingerprint: str = "fp1",
    verdict: Verdict = Verdict.ACCEPTED,
    attempt: int = 0,
    tags: list[CandidateRef] | None = None,
    alternatives: list[CandidateRef] | None = None,
    missing_skill: str | None = None,
    kind: str = "performance",
) -> TagRecord:
    return TagRecord(
        run_id="run-0",
        ts="2026-10-11T00:00:00+00:00",
        doc_key="doc",
        unit_key=unit_key,
        row_id="3",
        suite=suite,
        fingerprint=fingerprint,
        attempt=attempt,
        versions=_VERSIONS,
        kind=kind,
        clean_text="do the thing",
        context_path=["Section", "Heading"],
        verdict=verdict,
        reason_code="model_exact_corroborated",
        tags=tags or [_candidate("esco:skill:a")],
        alternatives=alternatives or [],
        missing_skill=missing_skill,
    )


def _decision(
    *,
    unit_key: str = "u1",
    suite: str = "esco",
    action: ReviewAction = ReviewAction.APPROVE,
    alt_index: int | None = None,
    hint: str = "",
    base_fp: str = "fp1",
    alt_ids: list[str] | None = None,
) -> ReviewDecision:
    return ReviewDecision(
        unit_key=unit_key,
        suite=suite,
        action=action,
        alt_index=alt_index,
        hint=hint,
        base_fp=base_fp,
        source_sha="sha-1",
        row_id="3",
        alt_ids=alt_ids or [],
    )


def _no_search(_suite: str) -> Callable[[str, str | None], object] | None:
    return None


# --- read_decisions: header-by-name tolerance -------------------------------


def _review_sheet(path: Path, rows: list[list[object]], *, banner: bool = False) -> None:
    workbook = openpyxl.Workbook()
    ws = workbook.active
    ws.title = "Review"
    if banner:
        ws.append(["Review — banner text spans this row"])
    headers = [
        "Row ID",
        "Where",
        "Line",
        "Kind",
        "Taxonomy",
        "Suggested tag(s)",
        "Verdict",
        "Why",
        "Alternatives",
        "Previous decision",
        "Decision ▼",
        "Hint",
        "key",
        "suite",
        "alt_ids",
        "fp",
    ]
    ws.append(headers)
    for row in rows:
        ws.append(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    workbook.save(path)


def _row(row_id: str, unit_key: str, suite: str, decision: str, hint: str = "") -> list[object]:
    return [
        row_id, "where", "line", "performance", suite.upper(), "tags", "verdict", "why", "alts",
        "", decision, hint, unit_key, suite, "[]", "fp1",
    ]  # fmt: skip


def test_missing_workbook_returns_empty(tmp_path: Path) -> None:
    decisions, warnings = read_decisions(tmp_path / "nope.xlsx")
    assert decisions == []
    assert warnings == []


def test_finds_header_row_despite_a_banner_above_it(tmp_path: Path) -> None:
    path = tmp_path / "out.xlsx"
    _review_sheet(path, [_row("3", "u1", "esco", "Approve")], banner=True)

    decisions, warnings = read_decisions(path)
    assert warnings == []
    [decision] = decisions
    assert decision.action == ReviewAction.APPROVE
    assert decision.unit_key == "u1"


def test_blank_decision_is_ignored(tmp_path: Path) -> None:
    path = tmp_path / "out.xlsx"
    _review_sheet(path, [_row("3", "u1", "esco", "")])

    decisions, warnings = read_decisions(path)
    assert decisions == []
    assert warnings == []


def test_tolerates_rows_out_of_original_order(tmp_path: Path) -> None:
    path = tmp_path / "out.xlsx"
    _review_sheet(
        path,
        [
            _row("9", "u9", "esco", "Approve"),
            _row("3", "u3", "esco", "Mark as gap"),
        ],
    )
    decisions, _warnings = read_decisions(path)
    by_unit = {d.unit_key: d.action for d in decisions}
    assert by_unit == {"u9": ReviewAction.APPROVE, "u3": ReviewAction.MARK_GAP}


def test_spanish_decision_words_are_understood(tmp_path: Path) -> None:
    path = tmp_path / "out.xlsx"
    _review_sheet(
        path,
        [
            _row("1", "u1", "esco", "Aprobar"),
            _row("2", "u2", "esco", "Marcar como brecha"),
            _row("3", "u3", "esco", "Usar alternativa 3"),
            _row("4", "u4", "esco", "Reintentar con pista", "una pista"),
        ],
    )
    decisions, warnings = read_decisions(path)
    assert warnings == []
    by_unit = {d.unit_key: d for d in decisions}
    assert by_unit["u1"].action == ReviewAction.APPROVE
    assert by_unit["u2"].action == ReviewAction.MARK_GAP
    assert by_unit["u3"].action == ReviewAction.USE_ALT
    assert by_unit["u3"].alt_index == 3
    assert by_unit["u4"].action == ReviewAction.RETRY
    assert by_unit["u4"].hint == "una pista"


def test_unknown_decision_word_warns_and_is_skipped(tmp_path: Path) -> None:
    path = tmp_path / "out.xlsx"
    _review_sheet(path, [_row("3", "u1", "esco", "do a backflip")])

    decisions, warnings = read_decisions(path)
    assert decisions == []
    assert len(warnings) == 1
    assert "3" in warnings[0]


def test_gaps_sheet_yes_and_no(tmp_path: Path) -> None:
    path = tmp_path / "out.xlsx"
    workbook = openpyxl.Workbook()
    workbook.active.title = "Review"
    gaps = workbook.create_sheet("Gaps")
    gaps.append(
        [
            "Row ID", "Where", "Line", "Kind", "Taxonomy", "Verdict",
            "Proposed missing skill", "Nearest existing (adjacent picks)",
            "Reviewer confirms gap? ▼", "key", "suite", "fp",
        ]
    )  # fmt: skip
    gaps.append(["5", "w", "l", "performance", "ESCO", "gap", "ms", "", "Yes", "u5", "esco", "fp5"])
    gaps.append(["6", "w", "l", "performance", "ESCO", "gap", "ms", "", "No", "u6", "esco", "fp6"])
    path.parent.mkdir(parents=True, exist_ok=True)
    workbook.save(path)

    decisions, warnings = read_decisions(path)
    assert warnings == []
    by_unit = {d.unit_key: d for d in decisions}
    assert by_unit["u5"].action == ReviewAction.MARK_GAP
    assert by_unit["u6"].action == ReviewAction.RETRY
    assert by_unit["u6"].hint == "reviewer says an existing skill fits"


# --- apply_decisions: effects ------------------------------------------------


def test_approve_makes_a_human_accepted_record(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / ".ta-tag")
    ledger.append(_record())

    outcome = apply_decisions(
        [_decision(action=ReviewAction.APPROVE)], ledger, search_by_suite=_no_search, run_id="r1"
    )

    assert outcome.applied == 1
    [record] = outcome.human_records
    assert record.verdict == Verdict.HUMAN_ACCEPTED
    assert record.decided_by == "human"
    assert record.reason_code == "human_approved"
    assert ledger.latest_for("u1", "esco") is record


def test_use_alternative_promotes_the_chosen_alternative(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / ".ta-tag")
    alt = _candidate("esco:skill:alt", label="alt")
    ledger.append(_record(alternatives=[alt]))

    outcome = apply_decisions(
        [_decision(action=ReviewAction.USE_ALT, alt_index=2, alt_ids=["esco:skill:alt"])],
        ledger,
        search_by_suite=_no_search,
        run_id="r1",
    )

    [record] = outcome.human_records
    assert record.verdict == Verdict.HUMAN_ACCEPTED
    assert [t.id for t in record.tags] == ["esco:skill:alt"]
    assert record.alternatives == []
    assert record.reason_code == "human_alternative"


def test_use_alternative_out_of_range_warns(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / ".ta-tag")
    ledger.append(_record(alternatives=[_candidate("esco:skill:alt")]))

    outcome = apply_decisions(
        [_decision(action=ReviewAction.USE_ALT, alt_index=3, alt_ids=["esco:skill:alt"])],
        ledger,
        search_by_suite=_no_search,
        run_id="r1",
    )

    assert outcome.human_records == []
    assert outcome.applied == 0
    assert len(outcome.warnings) == 1


def test_mark_gap_clears_tags_keeps_missing_skill(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / ".ta-tag")
    ledger.append(_record(verdict=Verdict.GAP, tags=[], missing_skill="some skill"))

    outcome = apply_decisions(
        [_decision(action=ReviewAction.MARK_GAP)], ledger, search_by_suite=_no_search, run_id="r1"
    )

    [record] = outcome.human_records
    assert record.verdict == Verdict.HUMAN_GAP
    assert record.tags == []
    assert record.missing_skill == "some skill"
    assert record.reason_code == "human_gap"


def test_retry_bumps_attempt_and_carries_the_hint(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / ".ta-tag")
    ledger.append(_record(attempt=0))

    outcome = apply_decisions(
        [_decision(action=ReviewAction.RETRY, hint="try harder")],
        ledger,
        search_by_suite=_no_search,
        run_id="r1",
    )

    assert outcome.retries == {("u1", "esco"): RetryRequest(hint="try harder", attempt=1)}
    assert outcome.human_records == []


def test_reject_retries_with_an_empty_hint(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / ".ta-tag")
    ledger.append(_record(attempt=0))

    outcome = apply_decisions(
        [_decision(action=ReviewAction.REJECT, hint="ignored")],
        ledger,
        search_by_suite=_no_search,
        run_id="r1",
    )

    assert outcome.retries == {("u1", "esco"): RetryRequest(hint="", attempt=1)}


def _log_reviewer_retry(ledger: Ledger, n: int) -> None:
    ledger.review_append(
        {"review_key": f"earlier-{n}", "unit_key": "u1", "suite": "esco", "result": "retry"}
    )


def test_reject_after_two_retries_becomes_human_gap_exhausted(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / ".ta-tag")
    ledger.append(_record(attempt=2))
    _log_reviewer_retry(ledger, 1)
    _log_reviewer_retry(ledger, 2)

    outcome = apply_decisions(
        [_decision(action=ReviewAction.REJECT)], ledger, search_by_suite=_no_search, run_id="r1"
    )

    assert outcome.retries == {}
    [record] = outcome.human_records
    assert record.verdict == Verdict.HUMAN_GAP
    assert record.reason_code == "retries_exhausted"


def test_automatic_second_look_does_not_use_up_a_reviewer_retry(tmp_path: Path) -> None:
    """Second look (attempt 1) + one reviewer retry (attempt 2): one reviewer retry is left."""
    ledger = Ledger(tmp_path / ".ta-tag")
    ledger.append(_record(attempt=2))
    _log_reviewer_retry(ledger, 1)

    outcome = apply_decisions(
        [_decision(action=ReviewAction.RETRY, hint="second try")],
        ledger,
        search_by_suite=_no_search,
        run_id="r1",
    )

    assert outcome.retries == {("u1", "esco"): RetryRequest(hint="second try", attempt=3)}
    assert outcome.human_records == []


def test_stale_fingerprint_is_ignored_with_a_warning(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / ".ta-tag")
    ledger.append(_record(fingerprint="fp-current"))

    outcome = apply_decisions(
        [_decision(base_fp="fp-old")], ledger, search_by_suite=_no_search, run_id="r1"
    )

    assert outcome.human_records == []
    assert outcome.applied == 0
    assert any("changed since" in w for w in outcome.warnings)


@dataclass
class _FakeNode:
    id: str
    label: str
    kind: str = "Skill"
    source: str = "esco"
    source_id: str = "urn:x"
    properties: dict[str, object] | None = None

    def __post_init__(self) -> None:
        if self.properties is None:
            self.properties = {}


@dataclass
class _FakeCandidate:
    node: _FakeNode
    method: str


@dataclass
class _FakeResult:
    candidates: list[_FakeCandidate]


def test_use_my_tag_accepts_a_single_exact_match(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / ".ta-tag")
    ledger.append(_record())

    def search_by_suite(suite: str) -> Callable[[str, str | None], object] | None:
        assert suite == "esco"

        def _search(text: str, kind: str | None) -> object:
            return _FakeResult([_FakeCandidate(_FakeNode("esco:skill:x", text), "exact_pref")])

        return _search

    outcome = apply_decisions(
        [_decision(action=ReviewAction.USE_MY_TAG, hint="testing software")],
        ledger,
        search_by_suite=search_by_suite,
        run_id="r1",
    )

    [record] = outcome.human_records
    assert record.verdict == Verdict.HUMAN_ACCEPTED
    assert record.tags[0].id == "esco:skill:x"
    assert record.reason_code == "human_tag"


def test_use_my_tag_not_found_warns_with_candidates(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / ".ta-tag")
    ledger.append(_record())

    def search_by_suite(_suite: str) -> Callable[[str, str | None], object] | None:
        def _search(text: str, kind: str | None) -> object:
            return _FakeResult(
                [
                    _FakeCandidate(_FakeNode("esco:skill:x", "contains hit 1"), "contains"),
                    _FakeCandidate(_FakeNode("esco:skill:y", "contains hit 2"), "contains"),
                ]
            )

        return _search

    outcome = apply_decisions(
        [_decision(action=ReviewAction.USE_MY_TAG, hint="not a real tag")],
        ledger,
        search_by_suite=search_by_suite,
        run_id="r1",
    )

    assert outcome.human_records == []
    assert any("not found exactly" in w for w in outcome.warnings)


def test_applying_the_same_decision_twice_is_a_no_op(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / ".ta-tag")
    ledger.append(_record())
    decision = _decision(action=ReviewAction.APPROVE)

    first = apply_decisions([decision], ledger, search_by_suite=_no_search, run_id="r1")
    assert first.applied == 1

    second = apply_decisions([decision], ledger, search_by_suite=_no_search, run_id="r2")
    assert second.applied == 0
    assert second.human_records == []
    assert second.warnings == []


def test_review_key_is_stable_for_the_same_decision() -> None:
    a = _decision()
    b = _decision()
    assert review_key(a) == review_key(b)
    assert review_key(a) != review_key(_decision(hint="different"))


def test_approving_a_policy_accepted_answer_counts_toward_precision(tmp_path: Path) -> None:
    """The human copy shares the reviewed answer's fingerprint; precision must still
    find what the *policy* said (this read 0 approved after a real approval)."""
    from talent_angels.tagging.models import TagUnit
    from talent_angels.tagging.review import review_stats

    ledger = Ledger(tmp_path / ".ta-tag")
    ledger.append(_record("u1", "esco", fingerprint="fp1", verdict=Verdict.ACCEPTED))
    ledger.append(_record("u2", "esco", fingerprint="fp2", verdict=Verdict.ACCEPTED))
    ledger.append(_record("u2", "onet", fingerprint="fp3", verdict=Verdict.ADJACENT))
    apply_decisions(
        [
            _decision(unit_key="u1", base_fp="fp1", action=ReviewAction.APPROVE),
            _decision(unit_key="u2", base_fp="fp2", action=ReviewAction.REJECT),
            _decision(unit_key="u2", suite="onet", base_fp="fp3", action=ReviewAction.APPROVE),
        ],
        Ledger(tmp_path / ".ta-tag"),
        search_by_suite=lambda _suite: None,
        run_id="run-1",
    )
    units = [
        TagUnit(
            unit_key=key,
            doc_key="doc",
            row_id=key,
            kind="performance",
            role="leaf",
            text="t",
            clean_text="t",
            context_path=[],
            group_key="0",
        )
        for key in ("u1", "u2")
    ]
    stats = review_stats(Ledger(tmp_path / ".ta-tag"), units, ["esco", "onet"])
    assert (stats.approved_total, stats.rejected_total) == (1, 1)  # adjacent approve excluded
    assert (stats.reviewed, stats.total) == (3, 4)  # per answer: 2 lines x 2 taxonomies
