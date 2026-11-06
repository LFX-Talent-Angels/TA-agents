"""Excel deliverable: Summary / Review / Standard sheets.

Reuses the mock's (`/tmp/mock_xlsx.py`) colours, fills, and dropdown
approach — not its banner text or sample data. The Gaps sheet and the
review-decision read-back come in PR4; this module only *writes*.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib import resources
from pathlib import Path

from openpyxl import Workbook
from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE, Cell
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.datavalidation import DataValidation
from openpyxl.worksheet.worksheet import Worksheet

from talent_angels.tagging.models import (
    Document,
    SkippedRow,
    TagRecord,
    TagUnit,
    Verdict,
)
from talent_angels.tagging.review import ReviewStats, tag_reference

_MAX_CELL_LEN = 32_767
_FORMULA_PREFIXES = ("=", "+", "-", "@")

_HEADER_FILL = PatternFill("solid", fgColor="1F3864")
_HEADER_FONT = Font(bold=True, color="FFFFFF")
_EDIT_FILL = PatternFill("solid", fgColor="DDEBF7")
_THIN = Side(style="thin", color="BBBBBB")
_BOX = Border(left=_THIN, right=_THIN, top=_THIN, bottom=_THIN)

#: Fill colour per verdict *value* (``Verdict.ACCEPTED.value == "accepted"``).
_VERDICT_FILL = {
    "accepted": "D9EAD3",
    "human_accepted": "D9EAD3",
    "review": "FFF2CC",
    "adjacent": "FCE5CD",
    "gap": "F4CCCC",
    "human_gap": "F4CCCC",
    "skipped": "EEEEEE",
    "n/a": "EEEEEE",
    "unavailable": "EAD1DC",
    "error": "EAD1DC",
}

_VERDICT_ICON = {
    Verdict.ACCEPTED: "✅ accepted",
    Verdict.REVIEW: "🔍 review",
    Verdict.ADJACENT: "🟡 adjacent",
    Verdict.GAP: "🔴 gap",
    Verdict.SKIPPED: "— skipped",
    Verdict.NOT_APPLICABLE: "— n/a",
    Verdict.UNAVAILABLE: "⚠️ unavailable",
    Verdict.ERROR: "❌ error",
    Verdict.HUMAN_ACCEPTED: "✔ approved",
    Verdict.HUMAN_GAP: "✔ gap (confirmed)",
}

#: Review sheet sort order: best match first, so the sheet opens on what
#: worked, down to the gaps. Rows whose run failed (`error`, `unavailable`)
#: stay above everything: rare, and easy to miss at the bottom. Ties keep
#: document order. Rows are matched by key, so re-sorting in Excel/Numbers is safe.
_SORT_TIER = {
    Verdict.ERROR: 0,
    Verdict.UNAVAILABLE: 0,
    Verdict.HUMAN_ACCEPTED: 1,
    Verdict.ACCEPTED: 2,
    Verdict.REVIEW: 3,
    Verdict.ADJACENT: 4,
    Verdict.GAP: 5,
    Verdict.HUMAN_GAP: 6,
}

_PREVIOUS_DECISION_LABELS = {
    "approve": "Approved",
    "reject": "Rejected",
    "use_alt": "Used alternative {alt_index}",
    "mark_gap": "Marked as gap",
    "retry": "Retry with hint '{hint}'",
    "use_my_tag": "Used tag '{hint}'",
}

_REASON_SENTENCES = {
    "no_candidates": "No candidates were found in this taxonomy for this line.",
    "search_only": "No-LLM mode: showing the top search result for manual review.",
    "suite_unavailable": "The taxonomy suite was unavailable during this run.",
    "kind_not_supported": "This suite does not support this unit's kind.",
    "model_failed": "The model call failed after a retry; this line needs a human look.",
    "model_exact_corroborated": "The model found an exact match, corroborated by the search.",
    "model_exact_uncorroborated": (
        "The model claims an exact match; not yet corroborated by search — please check."
    ),
    "model_partial": "No exact match; the model found skills that partially cover this line.",
    "needs_widening": "No match nearby; retried with a wider context, still inconclusive.",
    "model_none": "No taxonomy entry covers this line, even after widening the search.",
    "human_approved": "Approved by the reviewer.",
    "human_alternative": "The reviewer chose this alternative.",
    "human_tag": "The reviewer chose this tag.",
    "human_gap": "The reviewer confirmed that no taxonomy entry fits.",
    # Written before a retry past the limit was refused instead (older ledgers).
    "retries_exhausted": "Still no fit after two reviewer retries; recorded as a confirmed gap.",
}

#: The model's reasoning was about tags the reviewer has since replaced.
_DROP_MODEL_REASON = frozenset({"human_alternative", "human_tag"})

#: Rows re-run with a model (PR3) still land in one of these two buckets.
_NEEDS_REVIEW = {Verdict.REVIEW, Verdict.ADJACENT, Verdict.GAP, Verdict.ERROR}

_DECISION_OPTIONS = (
    "Approve,Reject,Use alternative 2,Use alternative 3,Use alternative 4,"
    "Mark as gap,Retry with hint,Use my tag"
)

_DISPLAY_NAMES = {"esco": "ESCO", "onet": "O*NET"}

_DEFAULT_ATTRIBUTION = {
    "esco": "ESCO © European Union, CC BY 4.0",
    "onet": "O*NET 31.0 Database, U.S. Department of Labor, CC BY 4.0",
    "sfia": (
        "SFIA — skill codes, names, levels and structure only (ADR-0006); "
        "SFIA descriptive text is not reproduced."
    ),
    "bls": "BLS/SOC — U.S. Bureau of Labor Statistics (public domain).",
}


def display_name(suite: str) -> str:
    """The suite's human name on the sheets (and in the `choose` prompt)."""
    return _DISPLAY_NAMES.get(suite, suite.upper())


_display_name = display_name


def _clean(value: object) -> object:
    """Strip illegal XML chars and cap length; non-strings pass through untouched."""
    if not isinstance(value, str):
        return value
    text = ILLEGAL_CHARACTERS_RE.sub("", value)
    if len(text) > _MAX_CELL_LEN:
        text = text[:_MAX_CELL_LEN]
    return text


def _write_cell(cell: Cell, value: object) -> None:
    """Every string written to the workbook goes through here (formula-safety)."""
    text = _clean(value)
    cell.value = text
    if isinstance(text, str) and text.startswith(_FORMULA_PREFIXES):
        # openpyxl marks a leading "=" as a formula (data_type "f"); forcing
        # "s" keeps a client's own text ("=5 kg flour") literal on reopen.
        cell.data_type = "s"


def _set(ws: Worksheet, row: int, col: int, value: object) -> Cell:
    cell = ws.cell(row=row, column=col)
    _write_cell(cell, value)
    return cell


def _package_attribution(suite: str) -> str | None:
    """The suite's own NOTES.md attribution blockquote, if the package is installed."""
    try:
        text = (
            resources.files("ta_taxonomies.suites")
            .joinpath(f"{suite}/NOTES.md")
            .read_text(encoding="utf-8")
        )
    except (ModuleNotFoundError, FileNotFoundError, OSError):
        return None
    collecting = False
    quoted: list[str] = []
    for line in text.splitlines():
        if not collecting:
            if "attribution" in line.lower():
                collecting = True
            continue
        stripped = line.strip()
        if stripped.startswith(">"):
            quoted.append(stripped.lstrip(">").strip())
        elif quoted:
            break
    return " ".join(quoted) if quoted else None


def _attribution_lines(suites: list[str]) -> list[str]:
    lines = []
    for suite in suites:
        # Try the installed package's own NOTES.md for *any* suite: a suite's
        # attribution should not need a hardcoded default to be picked up.
        text = _package_attribution(suite)
        lines.append(
            text or _DEFAULT_ATTRIBUTION.get(suite, f"{suite.upper()}: see the suite's license.")
        )
    return lines


def _write_table(
    ws: Worksheet,
    *,
    title: str,
    headers: list[str],
    rows: list[list[object]],
    widths: list[int],
    row_fills: list[str | None] | None = None,
    edit_cols: tuple[int, ...] = (),
    hidden_cols: tuple[int, ...] = (),
) -> None:
    """One title row + a header-by-name row + data; frozen panes, autofilter."""
    _set(ws, 1, 1, title).font = Font(bold=True, size=14)
    header_row = 2
    for col, header in enumerate(headers, start=1):
        cell = _set(ws, header_row, col, header)
        cell.fill = _HEADER_FILL
        cell.font = _HEADER_FONT
        cell.alignment = Alignment(wrap_text=True, vertical="center")
        cell.border = _BOX

    for offset, row in enumerate(rows):
        row_idx = header_row + 1 + offset
        fill_color = row_fills[offset] if row_fills else None
        for col_idx, value in enumerate(row, start=1):
            cell = _set(ws, row_idx, col_idx, value)
            cell.alignment = Alignment(wrap_text=True, vertical="top")
            cell.border = _BOX
            if (col_idx - 1) in edit_cols:
                cell.fill = _EDIT_FILL
            elif fill_color:
                cell.fill = PatternFill("solid", fgColor=fill_color)

    for col_idx, width in enumerate(widths, start=1):
        ws.column_dimensions[get_column_letter(col_idx)].width = width
    for col_idx in hidden_cols:
        ws.column_dimensions[get_column_letter(col_idx + 1)].hidden = True

    last_row = header_row + len(rows)
    last_col_letter = get_column_letter(len(headers))
    ws.freeze_panes = f"A{header_row + 1}"
    ws.auto_filter.ref = f"A{header_row}:{last_col_letter}{last_row}"


def _format_previous_decision(payload: dict[str, object] | None) -> str:
    """ "Retry with hint 'x' (applied 2026-10-12)" from one `reviews.jsonl` line."""
    if payload is None:
        return ""
    if "note" in payload:  # this run's message about a decision it could not apply
        return str(payload["note"])
    action = str(payload.get("action", ""))
    hint = str(payload.get("hint") or "")
    if payload.get("result") == "retries_exhausted":
        text = "Retries used up; marked as gap"
    elif action in ("retry", "reject") and payload.get("result") == "retry" and not hint:
        text = "Rejected; retried" if action == "reject" else "Retried (no hint)"
    else:
        template = _PREVIOUS_DECISION_LABELS.get(action)
        if template is None:
            return ""
        if action == "use_my_tag":
            hint = tag_reference(hint)
        text = template.format(hint=hint, alt_index=payload.get("alt_index"))
    date = str(payload.get("ts") or "")[:10]
    return f"{text} (applied {date})" if date else text


def _review_rows(
    units: list[TagUnit],
    results: dict[tuple[str, str], TagRecord],
    suites: list[str],
    previous_decisions: dict[tuple[str, str], dict[str, object]],
    kept_decisions: dict[tuple[str, str], tuple[str, str]] | None = None,
) -> tuple[list[list[object]], list[str | None]]:
    collected: list[tuple[int, list[object], str | None]] = []
    for unit in units:
        for suite in suites:
            record = results.get((unit.unit_key, suite))
            if record is None or record.verdict == Verdict.NOT_APPLICABLE:
                continue
            where = " › ".join(unit.context_path[-2:])
            if record.verdict == Verdict.GAP and record.missing_skill:
                tags_text = (
                    f"Proposed missing skill (by TA, not taxonomy fact): {record.missing_skill}"
                )
            else:
                tags_text = "\n".join(
                    f"{tag.label} → {record.covers[tag.id]}\n{tag.id}"
                    if tag.id in record.covers
                    else f"{tag.label}\n{tag.id}"
                    for tag in record.tags
                )
            alt_text = "\n".join(
                f"{i}. {alt.label} ({alt.id})" for i, alt in enumerate(record.alternatives, start=2)
            )
            icon = _VERDICT_ICON.get(record.verdict, str(record.verdict.value))
            why = _REASON_SENTENCES.get(record.reason_code, record.reason_code)
            if record.model_reason and record.reason_code not in _DROP_MODEL_REASON:
                why = f"{why}\nModel: {record.model_reason}"
            alt_ids = json.dumps([alt.id for alt in record.alternatives])
            previous = _format_previous_decision(previous_decisions.get((unit.unit_key, suite)))
            kept_decision, kept_hint = (kept_decisions or {}).get((unit.unit_key, suite), ("", ""))
            row: list[object] = [
                unit.row_id,
                where,
                unit.clean_text,
                _kind_label(unit),
                _display_name(suite),
                tags_text,
                icon,
                why,
                alt_text,
                previous,
                kept_decision,
                kept_hint,
                unit.unit_key,
                suite,
                alt_ids,
                record.fingerprint,
            ]
            priority = _SORT_TIER.get(record.verdict, len(_SORT_TIER))
            collected.append((priority, row, _VERDICT_FILL.get(record.verdict.value)))
    # Stable sort: ties (same priority) keep the document order they were
    # appended in (unit, then suite within unit).
    collected.sort(key=lambda item: item[0])
    rows = [item[1] for item in collected]
    fills = [item[2] for item in collected]
    return rows, fills


def _write_review(
    ws: Worksheet,
    units: list[TagUnit],
    results: dict[tuple[str, str], TagRecord],
    suites: list[str],
    previous_decisions: dict[tuple[str, str], dict[str, object]],
    kept_decisions: dict[tuple[str, str], tuple[str, str]] | None = None,
) -> None:
    headers = [
        "Row ID",
        "Where",
        "Line",
        "Kind",
        "Taxonomy",
        "Suggested tag(s)",
        "Verdict",
        "Why (model reasoning is inference)",
        "Alternatives",
        "Previous decision",
        "Decision ▼",
        "Hint",
        "key",
        "suite",
        "alt_ids",
        "fp",
    ]
    rows, fills = _review_rows(units, results, suites, previous_decisions, kept_decisions)
    widths = [8, 24, 44, 12, 10, 36, 14, 38, 32, 26, 20, 26, 10, 8, 12, 14]
    _write_table(
        ws,
        title="Review — set a Decision on any row, save, and re-run the same command",
        headers=headers,
        rows=rows,
        widths=widths,
        row_fills=fills,
        edit_cols=(10, 11),
        hidden_cols=(12, 13, 14, 15),
    )
    header_row = 2
    decision_col = get_column_letter(11)
    last_row = max(ws.max_row, header_row + 1)
    validation = DataValidation(type="list", formula1=f'"{_DECISION_OPTIONS}"', allow_blank=True)
    ws.add_data_validation(validation)
    validation.add(f"{decision_col}{header_row + 1}:{decision_col}{last_row}")


def _standard_rows(
    doc: Document,
    units: list[TagUnit],
    skipped: list[SkippedRow],
    results: dict[tuple[str, str], TagRecord],
    suites: list[str],
) -> list[list[object]]:
    units_by_row = {unit.row_id: unit for unit in units}
    skipped_by_row = {item.row_id: item for item in skipped}
    rows: list[list[object]] = []
    for row in doc.rows:
        unit = units_by_row.get(row.row_id)
        skip = skipped_by_row.get(row.row_id)
        if unit is not None:
            kind, status = _kind_label(unit), "tagged"
        elif skip is not None:
            kind, status = skip.kind or "", skip.reason
        else:  # pragma: no cover - select_units guarantees every row lands in one
            kind, status = "", ""
        out: list[object] = [row.row_id, row.label, row.text, row.parent_id or "", kind, status]
        for suite in suites:
            record = results.get((unit.unit_key, suite)) if unit is not None else None
            if record is not None:
                tags_text = "; ".join(f"{tag.label} [{tag.id}]" for tag in record.tags)
                verdict_text = _VERDICT_ICON.get(record.verdict, str(record.verdict.value))
            else:
                tags_text, verdict_text = "", ""
            out.extend([tags_text, verdict_text])
        rows.append(out)
    return rows


def _write_standard(
    ws: Worksheet,
    doc: Document,
    units: list[TagUnit],
    skipped: list[SkippedRow],
    results: dict[tuple[str, str], TagRecord],
    suites: list[str],
    document_label: str,
) -> None:
    headers = ["Row ID", "Type", "Text", "Parent ID", "Kind", "Status"]
    for suite in suites:
        name = _display_name(suite)
        headers.extend([f"{name} tags", f"{name} verdict"])
    rows = _standard_rows(doc, units, skipped, results, suites)
    widths = [8, 12, 50, 12, 14, 14] + [36, 14] * len(suites)
    _write_table(
        ws,
        title=(
            f"The client's {document_label.lower()}, row for row, with tag columns added "
            "(deliverable — regenerated each run, not edited)"
        ),
        headers=headers,
        rows=rows,
        widths=widths,
    )


@dataclass
class GapRow:
    """One (unit, suite) worth a human look: a gap, or a partial cover with one."""

    row_id: str
    where: str
    line: str
    kind: str
    taxonomy: str
    verdict: str
    missing_skill: str
    nearest_existing: str
    unit_key: str
    suite: str
    fp: str


def build_gap_rows(
    units: list[TagUnit],
    results: dict[tuple[str, str], TagRecord],
    suites: list[str],
    report_kinds: frozenset[str],
) -> list[GapRow]:
    """Rows for the Gaps sheet/csv: `report_kinds` only (definitions/summary stay out)."""
    rows: list[GapRow] = []
    for unit in units:
        if unit.kind not in report_kinds:
            continue
        for suite in suites:
            record = results.get((unit.unit_key, suite))
            if record is None:
                continue
            is_gap = record.verdict in (Verdict.GAP, Verdict.HUMAN_GAP)
            is_adjacent_gap = record.verdict == Verdict.ADJACENT and bool(record.missing_skill)
            if not (is_gap or is_adjacent_gap):
                continue
            if record.verdict == Verdict.ADJACENT:
                nearest = "; ".join(f"{tag.label} ({tag.id})" for tag in record.tags)
            else:
                nearest = "; ".join(f"{alt.label} ({alt.id})" for alt in record.alternatives)
            rows.append(
                GapRow(
                    row_id=unit.row_id,
                    where=" › ".join(unit.context_path[-2:]),
                    line=unit.clean_text,
                    kind=_kind_label(unit),
                    taxonomy=_display_name(suite),
                    verdict=_VERDICT_ICON.get(record.verdict, str(record.verdict.value)),
                    missing_skill=record.missing_skill or "",
                    nearest_existing=nearest,
                    unit_key=unit.unit_key,
                    suite=suite,
                    fp=record.fingerprint,
                )
            )
    # True gaps first (the "missing from the taxonomies" list), then the
    # partial covers; stable, so document order holds within each group.
    adjacent_icon = _VERDICT_ICON.get(Verdict.ADJACENT)
    return sorted(rows, key=lambda row: row.verdict == adjacent_icon)


def _write_gaps(ws: Worksheet, rows: list[GapRow]) -> None:
    headers = [
        "Row ID",
        "Where",
        "Line",
        "Kind",
        "Taxonomy",
        "Verdict",
        "Proposed missing skill (by TA — not taxonomy fact)",
        "Nearest existing (adjacent picks)",
        "Reviewer confirms gap? ▼",
        "key",
        "suite",
        "fp",
    ]
    data_rows: list[list[object]] = [
        [
            row.row_id,
            row.where,
            row.line,
            row.kind,
            row.taxonomy,
            row.verdict,
            row.missing_skill,
            row.nearest_existing,
            "",
            row.unit_key,
            row.suite,
            row.fp,
        ]
        for row in rows
    ]
    widths = [8, 24, 44, 14, 10, 16, 32, 32, 22, 10, 8, 14]
    _write_table(
        ws,
        title="Gaps — confirm whether an existing taxonomy entry really fits",
        headers=headers,
        rows=data_rows,
        widths=widths,
        edit_cols=(8,),
        hidden_cols=(9, 10, 11),
    )
    header_row = 2
    decision_col = get_column_letter(9)
    last_row = max(ws.max_row, header_row + 1)
    validation = DataValidation(type="list", formula1='"Yes,No"', allow_blank=True)
    ws.add_data_validation(validation)
    validation.add(f"{decision_col}{header_row + 1}:{decision_col}{last_row}")


def _write_summary(
    ws: Worksheet,
    doc: Document,
    units: list[TagUnit],
    results: dict[tuple[str, str], TagRecord],
    suites: list[str],
    summary: dict[str, object],
    stats: ReviewStats,
) -> None:
    title = str(summary.get("title") or doc.key)
    _set(ws, 1, 1, title).font = Font(bold=True, size=14)
    row = 3

    verdict_order = [member.value for member in Verdict]
    present = sorted(
        {record.verdict.value for record in results.values()},
        key=lambda value: (
            verdict_order.index(value) if value in verdict_order else len(verdict_order)
        ),
    )

    def _counts_table(
        heading: str, row_labels: list[str], match: Callable[[str, str, str], bool]
    ) -> None:
        nonlocal row
        _set(ws, row, 1, heading).font = Font(bold=True)
        row += 1
        header_row = row
        for col, header in enumerate(["", *present], start=1):
            cell = _set(ws, header_row, col, header)
            cell.fill = _HEADER_FILL
            cell.font = _HEADER_FONT
        row += 1
        for label in row_labels:
            counts = dict.fromkeys(present, 0)
            for (unit_key, suite), record in results.items():
                if match(label, unit_key, suite):
                    counts[record.verdict.value] = counts.get(record.verdict.value, 0) + 1
            _set(ws, row, 1, label)
            for col, verdict in enumerate(present, start=2):
                _set(ws, row, col, counts[verdict])
            row += 1
        row += 1

    _counts_table(
        "Per taxonomy",
        [_display_name(suite) for suite in suites],
        lambda label, unit_key, suite: _display_name(suite) == label,
    )

    units_by_key = {unit.unit_key: unit for unit in units}
    kinds_seen: list[str] = []
    for unit in units:
        if unit.kind not in kinds_seen:
            kinds_seen.append(unit.kind)

    def _kind_match(label: str, unit_key: str, suite: str) -> bool:
        del suite
        unit = units_by_key.get(unit_key)
        return unit is not None and unit.kind == label

    _counts_table("Per kind", kinds_seen, _kind_match)

    cost_usd = summary.get("cost_usd")
    cost_text = f"${cost_usd:.4f}" if isinstance(cost_usd, (int, float)) else "n/a"
    run_info = [
        ("Run date", summary.get("date") or datetime.now(UTC).isoformat()),
        ("Profile", summary.get("profile", "")),
        ("Units", summary.get("units", len(units))),
        ("Seconds", summary.get("seconds", 0.0)),
        ("Model", summary.get("model", "none")),
        ("Versions", summary.get("versions", "")),
        ("Reused", summary.get("reused", 0)),
        ("Searched", summary.get("searched", 0)),
        ("Model calls", summary.get("model_calls", 0)),
        ("Estimated cost", cost_text),
        ("Reviews applied", summary.get("reviews_applied", 0)),
        ("Retried", summary.get("retried", 0)),
    ]
    for label, value in run_info:
        _set(ws, row, 1, label).font = Font(bold=True)
        _set(ws, row, 2, value)
        row += 1
    row += 1

    _set(ws, row, 1, f"Reviewed: {stats.reviewed} of {stats.total} answers").font = Font(bold=True)
    row += 2
    _set(ws, row, 1, "Accepted precision (human-checked)").font = Font(bold=True)
    row += 1
    header_row = row
    for col, header in enumerate(["Kind", "Approved", "Rejected", "Precision"], start=1):
        cell = _set(ws, header_row, col, header)
        cell.fill = _HEADER_FILL
        cell.font = _HEADER_FONT
    row += 1
    precision_kinds = sorted(set(stats.approved_by_kind) | set(stats.rejected_by_kind))
    for kind in precision_kinds:
        approved = stats.approved_by_kind.get(kind, 0)
        rejected = stats.rejected_by_kind.get(kind, 0)
        denom = approved + rejected
        precision = f"{approved}/{denom}" if denom else "—"
        _set(ws, row, 1, kind)
        _set(ws, row, 2, approved)
        _set(ws, row, 3, rejected)
        _set(ws, row, 4, precision)
        row += 1
    total_denom = stats.approved_total + stats.rejected_total
    total_precision = f"{stats.approved_total}/{total_denom}" if total_denom else "—"
    _set(ws, row, 1, "Total").font = Font(bold=True)
    _set(ws, row, 2, stats.approved_total)
    _set(ws, row, 3, stats.rejected_total)
    _set(ws, row, 4, total_precision)
    row += 2

    _set(
        ws,
        row,
        1,
        "How to review: open the Review sheet, set Decision (blue column) for "
        "any row, save, and re-run the same command.",
    )
    row += 2

    for line in _attribution_lines(suites):
        _set(ws, row, 1, line)
        row += 1

    for col, width in enumerate((30, 14, 14, 14, 14, 14, 14, 14), start=1):
        ws.column_dimensions[get_column_letter(col)].width = width


def _kind_label(unit: TagUnit) -> str:
    """ "performance (group)" for a heading tagged together with its steps."""
    return f"{unit.kind} (group)" if unit.role == "group" else unit.kind


def write_workbook(
    path: Path,
    doc: Document,
    units: list[TagUnit],
    skipped: list[SkippedRow],
    results: dict[tuple[str, str], TagRecord],
    suites: list[str],
    summary: dict[str, object],
    *,
    report_kinds: frozenset[str] = frozenset(),
    previous_decisions: dict[tuple[str, str], dict[str, object]] | None = None,
    kept_decisions: dict[tuple[str, str], tuple[str, str]] | None = None,
    stats: ReviewStats | None = None,
    document_label: str = "Standard",
) -> Path:
    """Write the deliverable to `path`: temp file in the same dir, then atomic replace.

    `document_label` (profile.py) names the document sheet — "Standard" for CONOCER's
    competency_standard profile (the only tab name this ever changes).
    """
    previous_decisions = previous_decisions or {}
    stats = stats or ReviewStats(reviewed=0, total=len(units))

    workbook = Workbook()
    summary_ws = workbook.active
    assert summary_ws is not None
    summary_ws.title = "Summary"
    _write_summary(summary_ws, doc, units, results, suites, summary, stats)
    _write_review(
        workbook.create_sheet("Review"),
        units,
        results,
        suites,
        previous_decisions,
        kept_decisions,
    )
    _write_standard(
        workbook.create_sheet(document_label), doc, units, skipped, results, suites, document_label
    )
    gap_rows = build_gap_rows(units, results, suites, report_kinds)
    _write_gaps(workbook.create_sheet("Gaps"), gap_rows)

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.stem}-", suffix=".xlsx.tmp")
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        workbook.save(tmp_path)
        os.replace(tmp_path, path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise
    return path
