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
import re
import unicodedata
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

_MAX_RETRY_ATTEMPTS = 2
_EXACT_METHODS = frozenset({"exact_code", "exact_pref", "exact_alt", "casefold_pref"})
_REVIEWER_GAP_HINT = "reviewer says an existing skill fits"
#: Gaps-sheet answers, after casefolding and stripping accents ("Sí" -> "si").
_GAP_YES = frozenset({"yes", "y", "si", "s"})
_GAP_NO = frozenset({"no", "n"})
#: A suite-scoped node id in a "Use my tag" hint ("esco:skill:<uuid>", "onet:task:3967"),
#: alone or inside text pasted from a tag cell ("Software Developers → …\nonet:…").
_NODE_ID_RE = re.compile(r"\b([a-z][a-z0-9_]*):([A-Za-z_]+):([^\s,;()]+)")
#: Verdicts with no tag to approve; Approve on these is refused (gap ones are confirmed).
_NOTHING_TO_APPROVE = frozenset(
    {Verdict.ERROR, Verdict.UNAVAILABLE, Verdict.NOT_APPLICABLE, Verdict.SKIPPED}
)
#: A reviewer action on a policy-`accepted` answer that says the answer was wrong.
_CORRECTIONS = frozenset({"reject", "retry", "use_alt", "use_my_tag", "mark_gap"})


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
    #: Came from the Gaps sheet's Yes/No column rather than the Review sheet.
    from_gaps: bool = False


@dataclass
class RetryRequest:
    """One (unit, suite) the runner must re-search and re-choose this run.

    `attempt` keeps the ledger fingerprint unique (the pilot's own second look
    also counts there); `level` is the reviewer's own retry number (1 or 2),
    which picks how much of the standard's context the retry adds.
    """

    hint: str
    attempt: int
    level: int = 1


@dataclass
class ReviewOutcome:
    """Everything `apply_decisions` learned, for the runner to fold in."""

    human_records: list[TagRecord] = field(default_factory=list)
    retries: dict[tuple[str, str], RetryRequest] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    applied: int = 0
    #: Decisions read but not applied (warned about); their cells are kept.
    unapplied: set[tuple[str, str]] = field(default_factory=set)
    #: A line for the row's "Previous decision" cell, e.g. tags to choose from.
    notes: dict[tuple[str, str], str] = field(default_factory=dict)


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

    Scans every row for one containing a cell equal to `required` and a cell
    starting with `decision_prefix`. Not just the top: Apple Numbers imports
    the sheet without header rows, so sorting it there moves the header (and
    the title row) in among the data.
    """
    for row_idx in range(1, ws.max_row + 1):
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


def _data_rows(ws: Any, header_row: int) -> list[int]:
    """Every row but the header: rows are matched by their own key cell, not by position."""
    return [row_idx for row_idx in range(1, ws.max_row + 1) if row_idx != header_row]


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
    for row_idx in _data_rows(ws, header_row):
        decision_text = _cell_text(ws, row_idx, col_decision)
        unit_key = _cell_text(ws, row_idx, col_key)
        suite = _cell_text(ws, row_idx, col_suite)
        if not unit_key or not suite:
            continue
        row_id = _cell_text(ws, row_idx, col_row_id)
        if not decision_text:
            if _cell_text(ws, row_idx, col_hint):
                warnings.append(
                    f"row {row_id or unit_key}/{suite}: a Hint without a Decision does nothing; "
                    "choose Retry with hint or Use my tag"
                )
            continue
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
    for row_idx in _data_rows(ws, header_row):
        answer = _cell_text(ws, row_idx, col_decision)
        if not answer:
            continue
        unit_key = _cell_text(ws, row_idx, col_key)
        suite = _cell_text(ws, row_idx, col_suite)
        if not unit_key or not suite:
            continue
        row_id = _cell_text(ws, row_idx, col_row_id)
        base_fp = _cell_text(ws, row_idx, col_fp)
        normalized = _strip_accents(answer.strip().casefold())
        if normalized in _GAP_YES:
            decisions.append(
                ReviewDecision(
                    unit_key=unit_key,
                    suite=suite,
                    action=ReviewAction.MARK_GAP,
                    base_fp=base_fp,
                    source_sha=source_sha,
                    row_id=row_id,
                    from_gaps=True,
                )
            )
        elif normalized in _GAP_NO:
            decisions.append(
                ReviewDecision(
                    unit_key=unit_key,
                    suite=suite,
                    action=ReviewAction.RETRY,
                    hint=_REVIEWER_GAP_HINT,
                    base_fp=base_fp,
                    source_sha=source_sha,
                    row_id=row_id,
                    from_gaps=True,
                )
            )
        else:
            warnings.append(f"row {row_id or unit_key} (Gaps): unknown answer {answer!r}")
    return decisions


def _strip_accents(text: str) -> str:
    return "".join(
        ch for ch in unicodedata.normalize("NFKD", text) if not unicodedata.combining(ch)
    )


def decision_cells(path: Path) -> dict[tuple[str, str], tuple[str, str]]:
    """(unit_key, suite) -> (decision text, hint) for every Review row with either filled in."""
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
    cells: dict[tuple[str, str], tuple[str, str]] = {}
    for row_idx in _data_rows(ws, header_row):
        text, hint = _cell_text(ws, row_idx, col_decision), _cell_text(ws, row_idx, col_hint)
        unit_key, suite = _cell_text(ws, row_idx, col_key), _cell_text(ws, row_idx, col_suite)
        if (text or hint) and unit_key and suite:
            cells[(unit_key, suite)] = (text, hint)
    return cells


def kept_cells(
    cells: dict[tuple[str, str], tuple[str, str]], unapplied: set[tuple[str, str]]
) -> dict[tuple[str, str], tuple[str, str]]:
    """The Review cells to put back in the next workbook: whatever wasn't applied.

    A typo, a hint with no decision, or a decision that couldn't be carried
    out (warned about) stays visible to fix, instead of silently vanishing
    along with the decisions that were consumed.
    """
    return {
        key: (text, hint)
        for key, (text, hint) in cells.items()
        if not text or _parse_action(text) is None or key in unapplied
    }


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
    for name, reader, prefix in (
        ("Review", _read_review_sheet, "decision"),
        ("Gaps", _read_gaps_sheet, "reviewer confirms gap"),
    ):
        if name not in workbook.sheetnames:
            continue
        ws = workbook[name]
        if _find_header(ws, required="key", decision_prefix=prefix) is None:
            if not any(cell.value is not None for row in ws.iter_rows() for cell in row):
                continue
            warnings.append(
                f"{name} sheet: its header row (with the 'key' and decision columns) is gone, "
                "so no decisions could be read from it; re-run without changes to get a fresh "
                "workbook"
            )
            continue
        decisions.extend(reader(ws, source_sha, warnings))
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


def _apply_approve(
    decision: ReviewDecision, latest: TagRecord, *, run_id: str, warnings: list[str]
) -> TagRecord | None:
    """Approve what the row shows: its tags, or — on a gap row — the gap itself."""
    if not latest.tags:
        if latest.verdict in (Verdict.GAP, Verdict.HUMAN_GAP):
            return _apply_mark_gap(latest, run_id=run_id)
        if latest.verdict in _NOTHING_TO_APPROVE:
            warnings.append(
                f"row {decision.row_id or decision.unit_key}/{decision.suite}: nothing to approve "
                f"(the answer is '{latest.verdict.value}'); choose Retry with hint or Mark as gap"
            )
            return None
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


def _apply_mark_gap(
    latest: TagRecord, *, run_id: str, reason_code: str = "human_gap", keep_tags: bool = False
) -> TagRecord:
    """A confirmed gap. `keep_tags`: a Gaps-sheet "Yes" on an adjacent row confirms only
    the missing part, so its partial tags stay."""
    return _human_copy(
        latest,
        run_id=run_id,
        verdict=Verdict.HUMAN_GAP,
        reason_code=reason_code,
        tags=None if keep_tags else [],
    )


def _reviewer_retries(ledger: Ledger, unit_key: str, suite: str) -> int:
    """Retries/rejects already applied for this unit x suite, from `reviews.jsonl`.

    Counted from the reviewer's own decisions, not from `TagRecord.attempt`:
    the pilot's automatic second look also bumps `attempt`, and must not use
    up one of the reviewer's retries.
    """
    return sum(
        1
        for review in ledger.reviews()
        if review.get("unit_key") == unit_key
        and review.get("suite") == suite
        and review.get("result") == "retry"
    )


def _apply_retry_or_reject(
    decision: ReviewDecision,
    latest: TagRecord,
    *,
    run_id: str,
    retries_so_far: int,
    warnings: list[str],
) -> TagRecord | RetryRequest:
    if retries_so_far >= _MAX_RETRY_ATTEMPTS:
        warnings.append(
            f"row {decision.row_id or decision.unit_key}/{decision.suite}: already retried "
            f"{_MAX_RETRY_ATTEMPTS} times; marked as a confirmed gap"
        )
        return _apply_mark_gap(latest, run_id=run_id, reason_code="retries_exhausted")
    hint = decision.hint if decision.action == ReviewAction.RETRY else ""
    return RetryRequest(hint=hint, attempt=latest.attempt + 1, level=retries_so_far + 1)


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


def tag_reference(hint: str) -> str:
    """What a "Use my tag" hint refers to, short: its id(s), else the name before any "→"."""
    ids = list(
        dict.fromkeys(
            f"{m.group(1)}:{m.group(2)}:{m.group(3).rstrip('.')}"
            for m in _NODE_ID_RE.finditer(hint)
        )
    )
    if ids:
        return ", ".join(ids)
    lines = hint.strip().splitlines()
    return lines[0].split("→")[0].strip() if lines else ""


def _apply_use_my_tag(
    decision: ReviewDecision,
    latest: TagRecord,
    *,
    run_id: str,
    search_by_suite: Callable[[str], Callable[[str, str | None], object] | None],
    lookup_by_suite: Callable[[str], Callable[[str], object | None] | None] | None,
    warnings: list[str],
    notes: dict[tuple[str, str], str] | None = None,
) -> TagRecord | None:
    where = f"row {decision.row_id or decision.unit_key}/{decision.suite}"
    if not decision.hint:
        warnings.append(f"{where}: Use my tag needs the label or id in the Hint cell")
        return None
    search = search_by_suite(decision.suite)
    if search is None:
        warnings.append(
            f"{where}: '{decision.hint}' not checked; {decision.suite.upper()} is unavailable"
        )
        return None
    lookup = lookup_by_suite(decision.suite) if lookup_by_suite is not None else None
    reference = tag_reference(decision.hint)
    ids = reference.split(", ") if _NODE_ID_RE.search(decision.hint) else []
    if ids:
        # One or more ids (typed, or pasted with a whole tag cell): keep exactly those.
        foreign = [node_id for node_id in ids if not node_id.startswith(f"{decision.suite}:")]
        if foreign:
            warnings.append(
                f"{where}: {', '.join(foreign)} is not a {decision.suite.upper()} id; "
                "use an id from this taxonomy"
            )
            return None
        tags: list[CandidateRef] = []
        for node_id in ids:
            node = lookup(node_id) if lookup is not None else None
            if node is None:
                warnings.append(f"{where}: {node_id} not found in {decision.suite.upper()}")
                return None
            ref = node_ref(node, decision.suite)  # type: ignore[arg-type]
            tags.append(
                CandidateRef(
                    suite=decision.suite,
                    id=ref.id,
                    source_id=ref.source_id,
                    kind=ref.kind,
                    label=ref.pref_label,
                    method="id",
                    best_rank=1,
                    hits=1,
                    queries=[node_id],
                )
            )
        return _human_copy(
            latest,
            run_id=run_id,
            verdict=Verdict.HUMAN_ACCEPTED,
            reason_code="human_tag",
            tags=tags,
            alternatives=[],
        )
    label = reference  # a name: the label part of a pasted "label → covers" line
    result = search(label, None)
    candidates = list(getattr(result, "candidates", None) or [])
    exact = [c for c in candidates if getattr(c, "method", "") in _EXACT_METHODS]
    unique_ids = {c.node.id for c in exact}
    if len(unique_ids) == 1:
        tag = _candidate_from_raw(exact[0], decision.suite, label)
        return _human_copy(
            latest,
            run_id=run_id,
            verdict=Verdict.HUMAN_ACCEPTED,
            reason_code="human_tag",
            tags=[tag],
            alternatives=[],
        )
    # Never guess: a partial or misspelt name is not applied. Offer the closest
    # tags with their ids, in the workbook row too, so one can be pasted in.
    options = [f"{c.node.label} ({c.node.id})" for c in candidates[:5]]
    if options:
        warnings.append(
            f"{where}: '{label}' is not exactly one {decision.suite.upper()} tag; did you mean: "
            + "; ".join(options)
            + ". Paste one id into Hint and re-run"
        )
        if notes is not None:
            notes[(decision.unit_key, decision.suite)] = (
                f"Use my tag: '{label}' is not exactly one tag, so nothing was applied. "
                "Paste one of these ids into Hint and re-run:\n" + "\n".join(options)
            )
    else:
        warnings.append(f"{where}: '{label}' not found in {decision.suite.upper()}")
        if notes is not None:
            notes[(decision.unit_key, decision.suite)] = (
                f"Use my tag: '{label}' was not found in {decision.suite.upper()}; "
                "nothing was applied"
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


def review_stats(ledger: Ledger, units: list[TagUnit], suites: Sequence[str] = ()) -> ReviewStats:
    """ "Reviewed: X of Y" plus "approved / (approved + rejected) of rows the policy had accepted".

    Precision only counts decisions on an answer the *policy* had marked
    `accepted` (its `base_fp`'s policy record) — a reviewer approving a
    `review`/`adjacent` row doesn't move this number, by design. Approve counts
    as right; any correction (reject, retry, another alternative, own tag, gap)
    as wrong; only the latest decision per answer counts.
    """
    unit_keys = {unit.unit_key for unit in units}
    # Decisions are made per answer (one line x one taxonomy), so count those.
    reviewed_answers: set[tuple[str, str]] = set()
    approved: dict[str, int] = {}
    rejected: dict[str, int] = {}
    approved_total = 0
    rejected_total = 0
    latest_per_answer: dict[tuple[str, str, str], str] = {}

    for payload in ledger.reviews():
        unit_key = payload.get("unit_key")
        if not isinstance(unit_key, str) or unit_key not in unit_keys:
            continue
        suite = str(payload.get("suite", ""))
        reviewed_answers.add((unit_key, suite))
        base_fp = payload.get("base_fp")
        if isinstance(base_fp, str):
            latest_per_answer[(unit_key, suite, base_fp)] = str(payload.get("action", ""))

    for (_unit_key, _suite, base_fp), action in latest_per_answer.items():
        if action != ReviewAction.APPROVE.value and action not in _CORRECTIONS:
            continue
        original = ledger.policy_record(base_fp)
        if original is None or original.verdict != Verdict.ACCEPTED:
            continue
        if action == ReviewAction.APPROVE.value:
            approved_total += 1
            approved[original.kind] = approved.get(original.kind, 0) + 1
        else:
            rejected_total += 1
            rejected[original.kind] = rejected.get(original.kind, 0) + 1

    return ReviewStats(
        reviewed=len(reviewed_answers),
        total=len(units) * max(len(suites), 1),
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
    lookup_by_suite: Callable[[str], Callable[[str], object | None] | None] | None = None,
) -> ReviewOutcome:
    """Apply every unseen decision; skip (silently) any already in `reviews.jsonl`.

    Two different decisions for the same answer in one workbook (say Approve on
    the Review sheet and "No" on the Gaps sheet) are both held back with a
    warning rather than applied one over the other.
    """
    outcome = ReviewOutcome()

    fresh: list[tuple[str, ReviewDecision]] = []
    seen_keys: set[str] = set()
    for decision in decisions:
        key = review_key(decision)
        if ledger.review_applied(key) or key in seen_keys:
            continue
        seen_keys.add(key)
        fresh.append((key, decision))

    by_answer: dict[tuple[str, str], list[ReviewDecision]] = {}
    for _key, decision in fresh:
        by_answer.setdefault((decision.unit_key, decision.suite), []).append(decision)
    conflicted = {answer for answer, items in by_answer.items() if len(items) > 1}
    for answer in sorted(conflicted):
        items = by_answer[answer]
        row = items[0].row_id or answer[0]
        words = " and ".join(
            f"{'Gaps' if item.from_gaps else 'Review'} sheet: {item.action.value}" for item in items
        )
        outcome.warnings.append(
            f"row {row}/{answer[1]}: two different decisions ({words}); neither applied. "
            "The Review sheet's decision stays in its cell: re-run to apply it, or change it"
        )
        outcome.unapplied.add(answer)

    for key, decision in fresh:
        answer = (decision.unit_key, decision.suite)
        if answer in conflicted:
            continue

        latest = ledger.latest_for(decision.unit_key, decision.suite)
        if latest is None or latest.fingerprint != decision.base_fp:
            outcome.warnings.append(
                f"row {decision.row_id or decision.unit_key}/{decision.suite} changed since this "
                "workbook was written; decision ignored (if you kept an older copy open in "
                "Numbers or Excel, reopen the file in out/ and decide again)"
            )
            continue

        retries_so_far = _reviewer_retries(ledger, decision.unit_key, decision.suite)
        result: TagRecord | RetryRequest | None
        if decision.action == ReviewAction.APPROVE:
            result = _apply_approve(decision, latest, run_id=run_id, warnings=outcome.warnings)
        elif decision.action == ReviewAction.USE_ALT:
            result = _apply_use_alt(decision, latest, run_id=run_id, warnings=outcome.warnings)
        elif decision.action == ReviewAction.MARK_GAP:
            keep_tags = decision.from_gaps and latest.verdict == Verdict.ADJACENT
            result = _apply_mark_gap(latest, run_id=run_id, keep_tags=keep_tags)
        elif decision.action in (ReviewAction.RETRY, ReviewAction.REJECT):
            result = _apply_retry_or_reject(
                decision,
                latest,
                run_id=run_id,
                retries_so_far=retries_so_far,
                warnings=outcome.warnings,
            )
        elif decision.action == ReviewAction.USE_MY_TAG:
            result = _apply_use_my_tag(
                decision,
                latest,
                run_id=run_id,
                search_by_suite=search_by_suite,
                lookup_by_suite=lookup_by_suite,
                warnings=outcome.warnings,
                notes=outcome.notes,
            )
        else:  # pragma: no cover - _parse_action never returns an unknown action
            result = None

        if result is None:
            outcome.unapplied.add(answer)
            continue

        payload: dict[str, object] = {
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
        }
        if isinstance(result, RetryRequest):
            outcome.retries[answer] = result
            payload.update(
                result="retry",
                retry_hint=result.hint,
                attempt=result.attempt,
                level=result.level,
            )
        else:
            ledger.append(result)
            outcome.human_records.append(result)
            payload["result"] = (
                "retries_exhausted"
                if result.reason_code == "retries_exhausted"
                else result.verdict.value
            )

        outcome.applied += 1
        ledger.review_append(payload)

    return outcome


def pending_retries(ledger: Ledger) -> dict[tuple[str, str], RetryRequest]:
    """Every reviewer retry that hasn't produced an answer yet, ready to run again.

    A retry is written to `reviews.jsonl` when it is read, but only *done* once
    a later answer for that row is in the ledger. Until then — the run was
    stopped, the model call failed (`error`), the suite was down, or the row
    wasn't part of this run's `--rows`/`--limit`/`--suites` — it stays pending
    and keeps its hint, instead of being silently lost.
    """
    latest_retry: dict[tuple[str, str], dict[str, object]] = {}
    for payload in ledger.reviews():
        if payload.get("result") != "retry":
            continue
        unit_key, suite = payload.get("unit_key"), payload.get("suite")
        if isinstance(unit_key, str) and isinstance(suite, str):
            latest_retry[(unit_key, suite)] = payload

    pending: dict[tuple[str, str], RetryRequest] = {}
    for answer, payload in latest_retry.items():
        if ledger.latest_review_for(*answer) is not payload:
            continue  # the reviewer decided something else since
        latest = ledger.latest_for(*answer)
        if latest is None:
            continue
        done = latest.ts > str(payload.get("ts", "")) and (
            latest.decided_by == "human"
            or latest.verdict not in (Verdict.ERROR, Verdict.UNAVAILABLE)
        )
        if done:
            continue
        if "retry_hint" in payload:
            hint = str(payload.get("retry_hint") or "")
        else:  # written before retries were resumable
            hint = str(payload.get("hint") or "") if payload.get("action") == "retry" else ""
        attempt = payload.get("attempt")
        level = payload.get("level")
        pending[answer] = RetryRequest(
            hint=hint,
            attempt=attempt if isinstance(attempt, int) else latest.attempt + 1,
            level=level if isinstance(level, int) else 1,
        )
    return pending
