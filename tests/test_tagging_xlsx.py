"""`writers/xlsx.write_workbook`: reopen with openpyxl and check what landed."""

from __future__ import annotations

import json
from pathlib import Path

import openpyxl
import pytest

from talent_angels.tagging.models import (
    CandidateRef,
    Document,
    SkippedRow,
    SourceRow,
    TagRecord,
    TagUnit,
    Verdict,
    Versions,
)
from talent_angels.tagging.writers.xlsx import write_workbook

_VERSIONS = Versions(
    pipeline="tag-1", policy="tag-v1", prompts="rw-1/ch-1", ta_taxonomies="unknown", model="none"
)


def _unit(unit_key: str, row_id: str, clean_text: str, kind: str = "performance") -> TagUnit:
    return TagUnit(
        unit_key=unit_key,
        doc_key="doc",
        row_id=row_id,
        kind=kind,
        role="leaf",
        text=clean_text,
        clean_text=clean_text,
        context_path=["Element 1", "Performances"],
        group_key="parent",
    )


def _candidate(id_: str, suite: str, label: str = "label") -> CandidateRef:
    return CandidateRef(
        suite=suite,
        id=id_,
        source_id=f"urn:{id_}",
        kind="Skill",
        label=label,
        method="exact_pref",
        best_rank=1,
        hits=1,
        queries=["q"],
    )


def _record(
    unit: TagUnit, suite: str, verdict: Verdict, reason: str = "search_only", **extra: object
) -> TagRecord:
    defaults: dict[str, object] = dict(
        run_id="run-1",
        ts="2026-10-11T00:00:00+00:00",
        doc_key=unit.doc_key,
        unit_key=unit.unit_key,
        row_id=unit.row_id,
        suite=suite,
        fingerprint=f"fp-{unit.unit_key}-{suite}",
        versions=_VERSIONS,
        kind=unit.kind,
        clean_text=unit.clean_text,
        context_path=unit.context_path,
        verdict=verdict,
        reason_code=reason,
    )
    defaults.update(extra)
    return TagRecord(**defaults)  # type: ignore[arg-type]


@pytest.fixture
def _doc() -> Document:
    return Document(
        key="sample-doc",
        path="sample.json",
        source_format="neo4j-tree-json",
        rows=[
            SourceRow(
                row_id="0", parent_id=None, label="Document", text="Sample Standard", order=0
            ),
            SourceRow(row_id="1", parent_id="0", label="Section", text="Section I", order=1),
            SourceRow(
                row_id="2", parent_id="1", label="Item", text="=SUM(A1:A2) weird line", order=2
            ),
            SourceRow(row_id="3", parent_id="1", label="Item", text="a clean line", order=3),
        ],
    )


def test_write_workbook_creates_all_four_sheets(tmp_path: Path, _doc: Document) -> None:
    unit = _unit("u2", "2", "=SUM(A1:A2) weird line")
    skipped = [
        SkippedRow(row_id="0", reason="root"),
        SkippedRow(row_id="1", reason="heading", kind=None),
        SkippedRow(row_id="3", reason="excluded_kind", kind="performance"),
    ]
    results = {(unit.unit_key, "esco"): _record(unit, "esco", Verdict.REVIEW)}

    out_path = write_workbook(
        tmp_path / "out.xlsx", _doc, [unit], skipped, results, ["esco"], {"title": "Sample"}
    )
    workbook = openpyxl.load_workbook(out_path)
    assert workbook.sheetnames == ["Summary", "Review", "Standard", "Gaps"]


def test_formula_like_text_stays_literal_text(tmp_path: Path, _doc: Document) -> None:
    unit = _unit("u2", "2", "=SUM(A1:A2) weird line")
    results = {(unit.unit_key, "esco"): _record(unit, "esco", Verdict.REVIEW)}
    out_path = write_workbook(tmp_path / "out.xlsx", _doc, [unit], [], results, ["esco"], {})

    workbook = openpyxl.load_workbook(out_path)
    standard = workbook["Standard"]
    texts = [row[2].value for row in standard.iter_rows(min_row=3) if row[0].value == "2"]
    assert texts == ["=SUM(A1:A2) weird line"]
    # The cell must not have been written as an actual formula.
    cell = next(row[2] for row in standard.iter_rows(min_row=3) if row[0].value == "2")
    assert cell.data_type == "s"


def test_illegal_control_characters_are_stripped(tmp_path: Path, _doc: Document) -> None:
    unit = _unit("u3", "3", "a clean line\x0bwith control char")
    results = {(unit.unit_key, "esco"): _record(unit, "esco", Verdict.ACCEPTED)}
    out_path = write_workbook(tmp_path / "out.xlsx", _doc, [unit], [], results, ["esco"], {})

    workbook = openpyxl.load_workbook(out_path)
    review = workbook["Review"]
    lines = [row[2].value for row in review.iter_rows(min_row=3)]
    assert all(line is None or "\x0b" not in line for line in lines)


def test_standard_sheet_has_per_suite_columns_with_unknown_name_uppercased(
    tmp_path: Path, _doc: Document
) -> None:
    unit = _unit("u3", "3", "a clean line")
    results = {
        (unit.unit_key, "esco"): _record(
            unit, "esco", Verdict.ACCEPTED, tags=[_candidate("esco:skill:a", "esco")]
        ),
        (unit.unit_key, "onet"): _record(unit, "onet", Verdict.REVIEW),
        (unit.unit_key, "sfia"): _record(unit, "sfia", Verdict.GAP),
    }
    out_path = write_workbook(
        tmp_path / "out.xlsx", _doc, [unit], [], results, ["esco", "onet", "sfia"], {}
    )

    workbook = openpyxl.load_workbook(out_path)
    standard = workbook["Standard"]
    headers = [cell.value for cell in standard[2]]
    assert "ESCO tags" in headers
    assert "O*NET tags" in headers
    assert "SFIA tags" in headers  # unknown suite falls back to name.upper()


def test_review_sheet_has_hidden_key_suite_alt_ids_fp_columns(
    tmp_path: Path, _doc: Document
) -> None:
    unit = _unit("u3", "3", "a clean line")
    alt = _candidate("esco:skill:alt", "esco", label="alt label")
    results = {
        (unit.unit_key, "esco"): _record(
            unit,
            "esco",
            Verdict.REVIEW,
            tags=[_candidate("esco:skill:a", "esco")],
            alternatives=[alt],
        )
    }
    out_path = write_workbook(tmp_path / "out.xlsx", _doc, [unit], [], results, ["esco"], {})

    workbook = openpyxl.load_workbook(out_path)
    review = workbook["Review"]
    headers = [cell.value for cell in review[2]]
    assert headers[-4:] == ["key", "suite", "alt_ids", "fp"]
    for idx in (13, 14, 15, 16):  # 1-based columns for the 4 hidden fields
        letter = review.cell(row=2, column=idx).column_letter
        assert review.column_dimensions[letter].hidden is True

    data_row = next(row for row in review.iter_rows(min_row=3) if row[0].value == "3")
    assert data_row[12].value == unit.unit_key
    assert data_row[13].value == "esco"
    assert json.loads(data_row[14].value) == ["esco:skill:alt"]
    assert data_row[15].value == results[(unit.unit_key, "esco")].fingerprint


def test_review_sheet_has_decision_dropdown(tmp_path: Path, _doc: Document) -> None:
    unit = _unit("u3", "3", "a clean line")
    results = {(unit.unit_key, "esco"): _record(unit, "esco", Verdict.REVIEW)}
    out_path = write_workbook(tmp_path / "out.xlsx", _doc, [unit], [], results, ["esco"], {})

    workbook = openpyxl.load_workbook(out_path)
    review = workbook["Review"]
    assert len(review.data_validations.dataValidation) == 1
    validation = review.data_validations.dataValidation[0]
    assert "Approve" in validation.formula1
    assert "Mark as gap" in validation.formula1


def test_not_applicable_verdict_excluded_from_review(tmp_path: Path, _doc: Document) -> None:
    unit = _unit("u3", "3", "a clean line")
    results = {
        (unit.unit_key, "esco"): _record(
            unit, "esco", Verdict.NOT_APPLICABLE, reason="kind_not_supported"
        )
    }
    out_path = write_workbook(tmp_path / "out.xlsx", _doc, [unit], [], results, ["esco"], {})

    workbook = openpyxl.load_workbook(out_path)
    review = workbook["Review"]
    assert review.max_row == 2  # title + header only, no data rows


# --- Gaps sheet/csv: report_kinds + verdict filter --------------------------


def test_gaps_sheet_only_lists_report_kinds_and_gap_like_verdicts(
    tmp_path: Path, _doc: Document
) -> None:
    gap_unit = _unit("u-gap", "3", "a gap line", kind="performance")
    adjacent_unit = _unit("u-adj", "3", "an adjacent line", kind="performance")
    definition_unit = _unit("u-def", "3", "a definition line", kind="definition")
    accepted_unit = _unit("u-acc", "3", "an accepted line", kind="performance")
    results = {
        (gap_unit.unit_key, "esco"): _record(
            gap_unit, "esco", Verdict.GAP, missing_skill="missing one"
        ),
        (adjacent_unit.unit_key, "esco"): _record(
            adjacent_unit,
            "esco",
            Verdict.ADJACENT,
            tags=[_candidate("esco:skill:a", "esco")],
            covers={"esco:skill:a": "part of it"},
            missing_skill="missing two",
        ),
        (definition_unit.unit_key, "esco"): _record(
            definition_unit, "esco", Verdict.GAP, missing_skill="out of scope"
        ),
        (accepted_unit.unit_key, "esco"): _record(
            accepted_unit, "esco", Verdict.ACCEPTED, tags=[_candidate("esco:skill:a", "esco")]
        ),
    }
    out_path = write_workbook(
        tmp_path / "out.xlsx",
        _doc,
        [gap_unit, adjacent_unit, definition_unit, accepted_unit],
        [],
        results,
        ["esco"],
        {},
        report_kinds=frozenset({"performance"}),
    )

    workbook = openpyxl.load_workbook(out_path)
    gaps = workbook["Gaps"]
    # Every fixture unit shares row_id "3"; the hidden "key" column (index 9,
    # 0-based) is what actually tells them apart.
    keys = {row[9].value for row in gaps.iter_rows(min_row=3)}
    assert keys == {gap_unit.unit_key, adjacent_unit.unit_key}


def test_true_gaps_come_before_adjacent_rows_on_the_gaps_sheet() -> None:
    from talent_angels.tagging.writers.xlsx import build_gap_rows

    units = [_unit("k1", "1", "first"), _unit("k2", "2", "second"), _unit("k3", "3", "third")]
    results = {
        ("k1", "esco"): _record(units[0], "esco", Verdict.ADJACENT, missing_skill="m1"),
        ("k2", "esco"): _record(units[1], "esco", Verdict.GAP, missing_skill="m2"),
        ("k3", "esco"): _record(units[2], "esco", Verdict.ADJACENT, missing_skill="m3"),
    }
    rows = build_gap_rows(units, results, ["esco"], frozenset({"performance"}))
    assert [row.row_id for row in rows] == ["2", "1", "3"]


def test_review_sheet_shows_best_matches_first_and_failures_on_top(
    tmp_path: Path, _doc: Document
) -> None:
    order = [
        Verdict.GAP,
        Verdict.ADJACENT,
        Verdict.HUMAN_GAP,
        Verdict.ACCEPTED,
        Verdict.REVIEW,
        Verdict.HUMAN_ACCEPTED,
        Verdict.ERROR,
    ]
    units = [_unit(f"u{i}", str(10 + i), f"line {i}") for i in range(len(order))]
    results = {
        (unit.unit_key, "esco"): _record(unit, "esco", verdict)
        for unit, verdict in zip(units, order, strict=True)
    }

    out_path = write_workbook(tmp_path / "out.xlsx", _doc, units, [], results, ["esco"], {})

    review = openpyxl.load_workbook(out_path)["Review"]
    verdict_col = [c.value for c in review[2]].index("Verdict") + 1
    shown = [review.cell(row=r, column=verdict_col).value for r in range(3, review.max_row + 1)]
    assert shown == [
        "❌ error",
        "✔ approved",
        "✅ accepted",
        "🔍 review",
        "🟡 adjacent",
        "🔴 gap",
        "✔ gap (confirmed)",
    ]
