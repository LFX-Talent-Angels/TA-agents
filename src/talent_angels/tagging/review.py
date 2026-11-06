"""Review loop: read the expert's Decision/Hint from a workbook, apply once.

The expert never edits the ledger directly — they edit `out/<doc>-tagged.xlsx`
(Review sheet, and PR4's Gaps sheet) and re-run the same command. This module
is the read-back half: find the Decision/Hint columns **by name** (so a
reordered or partly-deleted sheet still parses), turn each non-blank Decision
into a typed `ReviewDecision`, and apply it against the ledger's current
`latest_for(unit, suite)` record — never blindly, since the workbook on disk
may be older than the ledger (rule: a stale `fp` is a warning, not a crash).

Every decision applies **once**: `review_key` hashes the exact decision
(unit, suite, action, alt_index, hint, base fingerprint), and a replayed
workbook — the expert just re-saves without changing anything — is a no-op
against `reviews.jsonl` (no repeat model spend, no repeat ledger write).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from openpyxl import load_workbook
from pydantic import BaseModel

from talent_angels.skills.locate.resolve import node_ref
from talent_angels.tagging.ledger import Ledger
from talent_angels.tagging.models import CandidateRef, TagRecord, TagUnit, Verdict

#: A hidden-column search over the first N rows of a sheet finds the header
#: row even if the expert inserted a banner or two above it.
_HEADER_SCAN_ROWS = 10
_MAX_RETRY_ATTEMPTS = 2
_EXACT_METHODS = frozenset({"exact_code", "exact_pref", "exact_alt", "casefold_pref"})
_REVIEWER_GAP_HINT = "reviewer says an existing skill fits"


class ReviewAction(StrEnum):
    APPROVE = "approve"
    REJECT = "reject"
    USE_ALT = "use_alt"
    MARK_GAP = "mark_gap"
    RETRY = "retry"
    USE_MY_TAG = "use_my_tag"


#: English + Spanish decision words (PILOT-PLAN.md "Review loop"), case- and
#: whitespace-insensitive. `{n}` stands for the alternative number (2..4).
_WORDS: dict[str, ReviewAction] = {
    "approve": ReviewAction.APPROVE,
    "aprobar": ReviewAction.APPROVE,
    "reject": ReviewAction.REJECT,
    "rechazar": ReviewAction.REJECT,
    "mark as gap": ReviewAction.MARK_GAP,
    "marcar como brecha": ReviewAction.MARK_GAP,
    "retry with hint": ReviewAction.RETRY,
    "reintentar con pista": ReviewAction.RETRY,
    "use my tag": ReviewAction.USE_MY_TAG,
    "usar mi etiqueta": ReviewAction.USE_MY_TAG,
}
_ALT_WORDS = ("use alternative", "usar alternativa")


def _parse_action(text: str) -> tuple[ReviewAction, int | None] | None:
    """One Decision cell's text -> `(action, alt_index)`, or `None` if unrecognised."""
    normalized = " ".join(text.strip().casefold().split())
    if not normalized:
        return None
    if normalized in _WORDS:
        return _WORDS[normalized], None
    for prefix in _ALT_WORDS:
        if normalized.startswith(prefix):
            tail = normalized[len(prefix) :].strip()
            if tail.isdigit() and 2 <= int(tail) <= 4:
                return ReviewAction.USE_ALT, int(tail)
    return None


class ReviewDecision(BaseModel):
    """One applied-or-not decision read off a workbook row."""

    unit_key: str
    suite: str
    action: ReviewAction
    alt_index: int | None = None
    hint: str = ""
    base_fp: str
    source_sha: str
    #: Extra context for warnings/logging beyond the spec's minimal field set;
    #: not part of `review_key` or the applied effect.
    row_id: str = ""
    alt_ids: list[str] = []


@dataclass
class RetryRequest:
    """One (unit, suite) the runner must re-search and re-choose this run."""

    hint: str
    attempt: int


@dataclass
class ReviewOutcome:
    """Everything `apply_decisions` learned, for the runner to fold in."""

    human_records: list[TagRecord] = field(default_factory=list)
    retries: dict[tuple[str, str], RetryRequest] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    applied: int = 0


def review_key(decision: ReviewDecision) -> str:
    parts = [
        decision.unit_key,
        decision.suite,
        decision.action.value,
        str(decision.alt_index),
        decision.hint,
        decision.base_fp,
    ]
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()[:24]


def _source_sha(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()[:24]
    except OSError:
        return "unknown"


def _find_header(
    ws: Any, *, required: str, decision_prefix: str
) -> tuple[int, dict[str, int]] | None:
    """The header row + `{column name (casefold, stripped) -> 1-based index}`.

    Scans the first `_HEADER_SCAN_ROWS` rows for one containing a cell equal
    to `required` and a cell starting with `decision_prefix` — tolerant of a
    title row above it, and of the sheet being otherwise sorted/filtered.
    """
    max_row = min(ws.max_row, _HEADER_SCAN_ROWS)
    for row_idx in range(1, max_row + 1):
        values: dict[str, int] = {}
        has_required = False
        has_decision = False
        for col_idx, cell in enumerate(ws[row_idx], start=1):
            text = str(cell.value).strip() if cell.value is not None else ""
            key = " ".join(text.casefold().split())
            if not key:
                continue
            values[key] = col_idx
            if key == required:
                has_required = True
            if key.startswith(decision_prefix):
                has_decision = True
        if has_required and has_decision:
            return row_idx, values
    return None


def _col(columns: dict[str, int], *names: str) -> int | None:
    for name in names:
        if name in columns:
            return columns[name]
    for key, idx in columns.items():
        if any(key.startswith(name) for name in names):
            return idx
    return None


def _cell_text(ws: Any, row: int, col: int | None) -> str:
    if col is None:
        return ""
    value = ws.cell(row=row, column=col).value
    return str(value).strip() if value is not None else ""


def _read_review_sheet(ws: Any, source_sha: str, warnings: list[str]) -> list[ReviewDecision]:
    header = _find_header(ws, required="key", decision_prefix="decision")
    if header is None:
        return []
    header_row, columns = header
    col_key = _col(columns, "key")
    col_suite = _col(columns, "suite")
    col_decision = _col(columns, "decision")
    col_hint = _col(columns, "hint")
    col_fp = _col(columns, "fp")
    col_alt_ids = _col(columns, "alt_ids")
    col_row_id = _col(columns, "row id")

    decisions: list[ReviewDecision] = []
    for row_idx in range(header_row + 1, ws.max_row + 1):
        decision_text = _cell_text(ws, row_idx, col_decision)
        if not decision_text:
            continue
        unit_key = _cell_text(ws, row_idx, col_key)
        suite = _cell_text(ws, row_idx, col_suite)
        if not unit_key or not suite:
            continue
        row_id = _cell_text(ws, row_idx, col_row_id)
        parsed = _parse_action(decision_text)
        if parsed is None:
            warnings.append(f"row {row_id or unit_key}: unknown decision {decision_text!r}")
            continue
        action, alt_index = parsed
        alt_ids_raw = _cell_text(ws, row_idx, col_alt_ids)
        try:
            alt_ids = json.loads(alt_ids_raw) if alt_ids_raw else []
        except ValueError:
            alt_ids = []
        decisions.append(
            ReviewDecision(
                unit_key=unit_key,
                suite=suite,
                action=action,
                alt_index=alt_index,
                hint=_cell_text(ws, row_idx, col_hint),
                base_fp=_cell_text(ws, row_idx, col_fp),
                source_sha=source_sha,
                row_id=row_id,
                alt_ids=alt_ids if isinstance(alt_ids, list) else [],
            )
        )
    return decisions


def _read_gaps_sheet(ws: Any, source_sha: str, warnings: list[str]) -> list[ReviewDecision]:
    header = _find_header(ws, required="key", decision_prefix="reviewer confirms gap")
    if header is None:
        return []
    header_row, columns = header
    col_key = _col(columns, "key")
    col_suite = _col(columns, "suite")
    col_decision = _col(columns, "reviewer confirms gap")
    col_fp = _col(columns, "fp")
    col_row_id = _col(columns, "row id")

    decisions: list[ReviewDecision] = []
    for row_idx in range(header_row + 1, ws.max_row + 1):
        answer = _cell_text(ws, row_idx, col_decision)
        if not answer:
            continue
        unit_key = _cell_text(ws, row_idx, col_key)
        suite = _cell_text(ws, row_idx, col_suite)
        if not unit_key or not suite:
            continue
        row_id = _cell_text(ws, row_idx, col_row_id)
        base_fp = _cell_text(ws, row_idx, col_fp)
        normalized = answer.strip().casefold()
        if normalized == "yes":
            decisions.append(
                ReviewDecision(
                    unit_key=unit_key,
                    suite=suite,
                    action=ReviewAction.MARK_GAP,
                    base_fp=base_fp,
                    source_sha=source_sha,
                    row_id=row_id,
                )
            )
        elif normalized == "no":
            decisions.append(
                ReviewDecision(
                    unit_key=unit_key,
                    suite=suite,
                    action=ReviewAction.RETRY,
                    hint=_REVIEWER_GAP_HINT,
                    base_fp=base_fp,
                    source_sha=source_sha,
                    row_id=row_id,
                )
            )
        else:
            warnings.append(f"row {row_id or unit_key} (Gaps): unknown answer {answer!r}")
    return decisions


def unrecognised_decisions(path: Path) -> dict[tuple[str, str], tuple[str, str]]:
    """(unit_key, suite) -> (decision text, hint) for Review rows whose word wasn't understood.

    The rewritten workbook puts these back in their cells, so a typo stays
    visible to fix instead of silently disappearing with the consumed ones.
    """
    if not path.is_file():
        return {}
    workbook = load_workbook(path, data_only=True)
    if "Review" not in workbook.sheetnames:
        return {}
    ws = workbook["Review"]
    header = _find_header(ws, required="key", decision_prefix="decision")
    if header is None:
        return {}
    header_row, columns = header
    col_key, col_suite = _col(columns, "key"), _col(columns, "suite")
    col_decision, col_hint = _col(columns, "decision"), _col(columns, "hint")
    kept: dict[tuple[str, str], tuple[str, str]] = {}
    for row_idx in range(header_row + 1, ws.max_row + 1):
        text = _cell_text(ws, row_idx, col_decision)
        unit_key, suite = _cell_text(ws, row_idx, col_key), _cell_text(ws, row_idx, col_suite)
        if text and unit_key and suite and _parse_action(text) is None:
            kept[(unit_key, suite)] = (text, _cell_text(ws, row_idx, col_hint))
    return kept


def read_decisions(path: Path) -> tuple[list[ReviewDecision], list[str]]:
    """Every non-blank Decision on the Review + Gaps sheets of `path`, and warnings.

    `([], [])` when `path` doesn't exist yet (first run, nothing to read back).
    """
    if not path.is_file():
        return [], []
    warnings: list[str] = []
    source_sha = _source_sha(path)
    workbook = load_workbook(path, data_only=True)
    decisions: list[ReviewDecision] = []
    if "Review" in workbook.sheetnames:
        decisions.extend(_read_review_sheet(workbook["Review"], source_sha, warnings))
    if "Gaps" in workbook.sheetnames:
        decisions.extend(_read_gaps_sheet(workbook["Gaps"], source_sha, warnings))
    return decisions, warnings


def _human_copy(
    latest: TagRecord,
    *,
    run_id: str,
    verdict: Verdict,
    reason_code: str,
    tags: list[CandidateRef] | None = None,
    alternatives: list[CandidateRef] | None = None,
    missing_skill: str | None = None,
) -> TagRecord:
    update: dict[str, object] = {
        "run_id": run_id,
        "ts": datetime.now(UTC).isoformat(),
        "verdict": verdict,
        "reason_code": reason_code,
        "decided_by": "human",
    }
    if tags is not None:
        update["tags"] = tags
        update["covers"] = {}
    if alternatives is not None:
        update["alternatives"] = alternatives
    if missing_skill is not None:
        update["missing_skill"] = missing_skill
    return latest.model_copy(update=update)


def _apply_approve(latest: TagRecord, *, run_id: str) -> TagRecord:
    return _human_copy(
        latest, run_id=run_id, verdict=Verdict.HUMAN_ACCEPTED, reason_code="human_approved"
    )


def _apply_use_alt(
    decision: ReviewDecision, latest: TagRecord, *, run_id: str, warnings: list[str]
) -> TagRecord | None:
    index = (decision.alt_index - 2) if decision.alt_index is not None else -1
    if index < 0 or index >= len(decision.alt_ids):
        warnings.append(
            f"row {decision.row_id or decision.unit_key}/{decision.suite}: "
            f"alternative {decision.alt_index} not found"
        )
        return None
    target_id = decision.alt_ids[index]
    chosen = next((alt for alt in latest.alternatives if alt.id == target_id), None)
    if chosen is None:
        warnings.append(
            f"row {decision.row_id or decision.unit_key}/{decision.suite}: "
            f"alternative {decision.alt_index} not found among recorded alternatives"
        )
        return None
    others = [alt for alt in latest.alternatives if alt.id != target_id]
    return _human_copy(
        latest,
        run_id=run_id,
        verdict=Verdict.HUMAN_ACCEPTED,
        reason_code="human_alternative",
        tags=[chosen],
        alternatives=others,
    )


def _apply_mark_gap(latest: TagRecord, *, run_id: str, reason_code: str = "human_gap") -> TagRecord:
    return _human_copy(
        latest,
        run_id=run_id,
        verdict=Verdict.HUMAN_GAP,
        reason_code=reason_code,
        tags=[],
    )


def _apply_retry_or_reject(
    decision: ReviewDecision, latest: TagRecord, *, run_id: str
) -> TagRecord | RetryRequest:
    next_attempt = latest.attempt + 1
    if next_attempt > _MAX_RETRY_ATTEMPTS:
        return _apply_mark_gap(latest, run_id=run_id, reason_code="retries_exhausted")
    hint = decision.hint if decision.action == ReviewAction.RETRY else ""
    return RetryRequest(hint=hint, attempt=next_attempt)


def _candidate_from_raw(raw: object, suite: str, query: str) -> CandidateRef:
    ref = node_ref(raw.node, suite)  # type: ignore[attr-defined]
    return CandidateRef(
        suite=suite,
        id=ref.id,
        source_id=ref.source_id,
        kind=ref.kind,
        label=ref.pref_label,
        method=getattr(raw, "method", ""),
        best_rank=1,
        hits=1,
        queries=[query],
    )


def _apply_use_my_tag(
    decision: ReviewDecision,
    latest: TagRecord,
    *,
    run_id: str,
    search_by_suite: Callable[[str], Callable[[str, str | None], object] | None],
    warnings: list[str],
) -> TagRecord | None:
    search = search_by_suite(decision.suite)
    if search is None:
        warnings.append(
            f"'{decision.hint}' not found exactly in {decision.suite.upper()}; suite unavailable"
        )
        return None
    result = search(decision.hint, None)
    candidates = list(getattr(result, "candidates", None) or [])
    exact = [
        c
        for c in candidates
        if getattr(c, "method", "") in _EXACT_METHODS
        or getattr(c.node, "id", None) == decision.hint
    ]
    unique_ids = {c.node.id for c in exact}
    if len(unique_ids) == 1:
        tag = _candidate_from_raw(exact[0], decision.suite, decision.hint)
        return _human_copy(
            latest,
            run_id=run_id,
            verdict=Verdict.HUMAN_ACCEPTED,
            reason_code="human_tag",
            tags=[tag],
            alternatives=[],
        )
    names = ", ".join(c.node.label for c in candidates[:5]) or "none"
    warnings.append(
        f"'{decision.hint}' not found exactly in {decision.suite.upper()}; candidates: {names}"
    )
    return None


@dataclass
class ReviewStats:
    """Summary-sheet numbers: how much of this doc has a human look, and at what precision."""

    reviewed: int = 0
    total: int = 0
    approved_by_kind: dict[str, int] = field(default_factory=dict)
    rejected_by_kind: dict[str, int] = field(default_factory=dict)
    approved_total: int = 0
    rejected_total: int = 0


def review_stats(ledger: Ledger, units: list[TagUnit]) -> ReviewStats:
    """ "Reviewed: X of Y" plus "approved / (approved + rejected) of rows the policy had accepted".

    Precision only counts an approve/reject whose decision's own `base_fp`
    still resolves (via `ledger.lookup`) to a record the *policy* had marked
    `accepted` — a reviewer approving/rejecting a `review`/`adjacent` row
    doesn't move this number, by design (that is not what it measures).
    """
    unit_keys = {unit.unit_key for unit in units}
    reviewed_units: set[str] = set()
    approved: dict[str, int] = {}
    rejected: dict[str, int] = {}
    approved_total = 0
    rejected_total = 0

    for payload in ledger.reviews():
        unit_key = payload.get("unit_key")
        if not isinstance(unit_key, str) or unit_key not in unit_keys:
            continue
        reviewed_units.add(unit_key)
        action = payload.get("action")
        if action not in (ReviewAction.APPROVE.value, ReviewAction.REJECT.value):
            continue
        base_fp = payload.get("base_fp")
        original = ledger.lookup(str(base_fp)) if isinstance(base_fp, str) else None
        if original is None or original.verdict != Verdict.ACCEPTED:
            continue
        if action == ReviewAction.APPROVE.value:
            approved_total += 1
            approved[original.kind] = approved.get(original.kind, 0) + 1
        else:
            rejected_total += 1
            rejected[original.kind] = rejected.get(original.kind, 0) + 1

    return ReviewStats(
        reviewed=len(reviewed_units),
        total=len(units),
        approved_by_kind=approved,
        rejected_by_kind=rejected,
        approved_total=approved_total,
        rejected_total=rejected_total,
    )


def apply_decisions(
    decisions: Sequence[ReviewDecision],
    ledger: Ledger,
    *,
    search_by_suite: Callable[[str], Callable[[str, str | None], object] | None],
    run_id: str,
) -> ReviewOutcome:
    """Apply every unseen decision; skip (silently) any already in `reviews.jsonl`."""
    outcome = ReviewOutcome()

    for decision in decisions:
        key = review_key(decision)
        if ledger.review_applied(key):
            continue

        latest = ledger.latest_for(decision.unit_key, decision.suite)
        if latest is None or latest.fingerprint != decision.base_fp:
            outcome.warnings.append(
                f"row {decision.row_id or decision.unit_key}/{decision.suite} changed since this "
                "workbook was written; decision ignored"
            )
            continue

        result: TagRecord | RetryRequest | None
        if decision.action == ReviewAction.APPROVE:
            result = _apply_approve(latest, run_id=run_id)
        elif decision.action == ReviewAction.USE_ALT:
            result = _apply_use_alt(decision, latest, run_id=run_id, warnings=outcome.warnings)
        elif decision.action == ReviewAction.MARK_GAP:
            result = _apply_mark_gap(latest, run_id=run_id)
        elif decision.action in (ReviewAction.RETRY, ReviewAction.REJECT):
            result = _apply_retry_or_reject(decision, latest, run_id=run_id)
        elif decision.action == ReviewAction.USE_MY_TAG:
            result = _apply_use_my_tag(
                decision,
                latest,
                run_id=run_id,
                search_by_suite=search_by_suite,
                warnings=outcome.warnings,
            )
        else:  # pragma: no cover - _parse_action never returns an unknown action
            result = None

        if result is None:
            continue

        if isinstance(result, RetryRequest):
            outcome.retries[(decision.unit_key, decision.suite)] = result
            effect = "retry"
        else:
            ledger.append(result)
            outcome.human_records.append(result)
            effect = result.verdict.value

        outcome.applied += 1
        ledger.review_append(
            {
                "ts": datetime.now(UTC).isoformat(),
                "review_key": key,
                "unit_key": decision.unit_key,
                "suite": decision.suite,
                "action": decision.action.value,
                "alt_index": decision.alt_index,
                "hint": decision.hint,
                "base_fp": decision.base_fp,
                "source_sha": decision.source_sha,
                "run_id": run_id,
                "result": effect,
            }
        )

    return outcome
