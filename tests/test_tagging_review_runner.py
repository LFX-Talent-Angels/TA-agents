"""Offline e2e of the review loop (PR4): edit the workbook, re-run, check it stuck.

One doc, eight leaf "performance" units behind distinct keywords, each set up
for a different Decision: Approve, Use alternative 2, Retry with hint,
Reject, Mark as gap, an unrecognised word ("banana"), and the Gaps sheet's
Yes/No. A second `ta-agent tag` run (same command, same ledger) must apply
every decision exactly once, re-call the model only for the retried/rejected
units, and a third run must make zero model calls at all.
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import openpyxl
import pytest

from talent_angels.llm.protocol import LLMResult, LLMUsage, Message
from talent_angels.suites.schema import SuiteSchema
from talent_angels.tagging.models import RunOptions
from talent_angels.tagging.readers import read_document
from talent_angels.tagging.runner import WorkbookOpenError, run
from tests.fakes.tagging_suite import FakeCandidate, FakeNode, FakeToolResult, table_registry

_SCHEMA = SuiteSchema(
    skill_rel_types=("HAS_SKILL",),
    optional_rel_values=frozenset(),
    group_rel_type=None,
    group_node_kinds=frozenset(),
)

#: Row id -> keyword, in document order. Each keyword appears, lowercased, in
#: its row's own text, in the scripted rewrite's labels (same lowercasing),
#: and in the scripted choose's candidate matching.
_LEAVES = [
    ("11", "alphaapprove"),
    ("12", "alphaalt"),
    ("13", "betaretry"),
    ("14", "gammagap"),
    ("15", "deltabanana"),
    ("16", "epsilonyes"),
    ("17", "zetano"),
    ("18", "etareject"),
]
_GAP_KEYWORDS = frozenset({"betaretry", "gammagap", "epsilonyes", "zetano"})


def _node(identity: int, labels: list[str], text: str) -> dict[str, object]:
    return {
        "identity": identity,
        "labels": labels,
        "properties": {"text": text},
        "elementId": f"4:fixture:{identity}",
    }


def _edge(identity: int, start: int, end: int) -> dict[str, object]:
    return {
        "identity": identity,
        "start": start,
        "end": end,
        "type": "IS_PART_OF",
        "properties": {},
        "elementId": f"5:fixture:{identity}",
    }


def _build_fixture() -> list[dict[str, object]]:
    tree = [
        (1, ["Section"], "Section I", 0, ["Document"], "Fixture Review Loop Doc"),
        (2, ["Item"], "Title: Fixture Review Loop Doc", 1, ["Section"], "Section I"),
        (10, ["Item"], "Performances", 1, ["Section"], "Section I"),
    ]
    for row_id, keyword in _LEAVES:
        label = keyword[:1].upper() + keyword[1:]
        text = f"{label}: do the {keyword} procedure"
        tree.append((int(row_id), ["Item"], text, 10, ["Item"], "Performances"))

    rows = []
    edge_id = 1000
    for child_id, child_labels, child_text, parent_id, parent_labels, parent_text in tree:
        edge_id += 1
        rows.append(
            {
                "n": _node(child_id, child_labels, child_text),
                "r": _edge(edge_id, child_id, parent_id),
                "m": _node(parent_id, parent_labels, parent_text),
            }
        )
    return rows


@pytest.fixture
def project(tmp_path: Path) -> Path:
    project_dir = tmp_path / "project"
    (project_dir / "inputs").mkdir(parents=True)
    (project_dir / "inputs" / "standard.json").write_text(json.dumps(_build_fixture()))
    (project_dir / "tag.yaml").write_text("groups: false\n")
    return project_dir


@dataclass
class KeywordSuite:
    name: str
    hit_kinds: frozenset[str] = field(default_factory=lambda: frozenset({"skill", "task"}))
    calls: list[tuple[str, str | None]] = field(default_factory=list)

    @property
    def suite_schema(self) -> SuiteSchema:
        return _SCHEMA

    def search_nodes(self, text: str, kind: str | None = None) -> FakeToolResult:
        self.calls.append((text, kind))
        if kind is None or kind not in self.hit_kinds:
            return FakeToolResult(warnings=[f"unknown_kind:{kind}"])
        lowered = text.casefold()
        keyword = next((k for _row_id, k in _LEAVES if k in lowered), None)
        if keyword is None:
            return FakeToolResult(warnings=["not_found"])
        if keyword == "alphaalt":
            candidates = [
                FakeCandidate(
                    node=FakeNode(id=f"{self.name}:skill:alt1", kind="Skill", label="alt one"),
                    confidence=0.9,
                    method="exact_pref",
                ),
                FakeCandidate(
                    node=FakeNode(id=f"{self.name}:skill:alt2", kind="Skill", label="alt two"),
                    confidence=0.85,
                    method="exact_pref",
                ),
            ]
        else:
            candidates = [
                FakeCandidate(
                    node=FakeNode(id=f"{self.name}:skill:{keyword}", kind="Skill", label=keyword),
                    confidence=0.9,
                    method="exact_pref",
                )
            ]
        return FakeToolResult(candidates=candidates)

    def get_neighbors(self, node_id: str, rel_types: list[str] | None = None) -> FakeToolResult:
        return FakeToolResult(warnings=["node_not_found"])


def _registry() -> tuple[object, KeywordSuite]:
    esco = KeywordSuite("esco")
    return table_registry({"esco": esco}, default="esco"), esco


_REWRITE_ROW_RE = re.compile(r'<row id="(r\d+)"[^>]*>([^<]*)</row>')
_NONE_FIT = {"fit": "none", "picks": [], "reason": "no match"}


class ReviewScriptedClient:
    provider = "litellm"
    model = "test-model"

    def __init__(self) -> None:
        self.calls: list[list[Message]] = []

    def complete(
        self, messages: list[Message], *, tools: list[dict[str, object]] | None = None
    ) -> LLMResult:
        self.calls.append(messages)
        user = messages[1].content
        text = self._rewrite_reply(user) if "<row id=" in user else self._choose_reply(user)
        return LLMResult(
            text=text, provider=self.provider, model=self.model,
            usage=LLMUsage(input_tokens=5, output_tokens=5),
        )  # fmt: skip

    @staticmethod
    def _rewrite_reply(user: str) -> str:
        rows = {}
        for row_id, row_text in _REWRITE_ROW_RE.findall(user):
            keyword = row_text.strip().split(":")[0].strip().casefold()
            rows[row_id] = {"labels": [f"{keyword} label"], "description": f"{keyword} description"}
        return json.dumps({"rows": rows})

    @staticmethod
    def _choose_reply(user: str) -> str:
        lowered = user.casefold()
        if "alphaapprove" in lowered or "alphaalt" in lowered or "deltabanana" in lowered:
            return json.dumps({"fit": "exact", "picks": [{"n": 1}], "reason": "matches"})
        if "etareject" in lowered:
            return json.dumps({"fit": "exact", "picks": [{"n": 1}], "reason": "matches"})
        return json.dumps({**_NONE_FIT, "missing_skill": "a missing skill"})


def _ledger_lines(project_dir: Path, name: str) -> list[dict[str, object]]:
    path = project_dir / ".ta-tag" / name
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _open_review(out_path: Path) -> tuple[openpyxl.Workbook, object]:
    workbook = openpyxl.load_workbook(out_path)
    return workbook, workbook["Review"]


def _header_index(ws: object, name: str) -> int:
    for cell in ws[2]:  # type: ignore[attr-defined]
        if str(cell.value).strip().casefold() == name.casefold():
            return cell.column
    raise AssertionError(f"no {name!r} column")


def _row_by_id(ws: object, row_id: str, row_id_col: int) -> int:
    for row in ws.iter_rows(min_row=3):  # type: ignore[attr-defined]
        if str(row[row_id_col - 1].value) == row_id:
            return row[0].row
    raise AssertionError(f"no row with Row ID {row_id!r}")


def _set_decision(ws: object, row_id: str, decision: str, hint: str = "") -> None:
    row_id_col = _header_index(ws, "Row ID")
    row = _row_by_id(ws, row_id, row_id_col)
    ws.cell(row=row, column=_header_index(ws, "Decision ▼"), value=decision)  # type: ignore[attr-defined]
    if hint:
        ws.cell(row=row, column=_header_index(ws, "Hint"), value=hint)  # type: ignore[attr-defined]


def test_review_loop_applies_each_decision_once_and_retries_only_those_units(
    project: Path,
) -> None:
    registry1, _suite1 = _registry()
    client1 = ReviewScriptedClient()
    first = run(
        project, registry=registry1, opts=RunOptions(no_llm=False, fresh=True), llm_client=client1
    )
    [out_path_str] = first.outputs
    out_path = Path(out_path_str)

    # --- sanity on the first run's own verdicts -----------------------------
    ledger_records = {r["row_id"]: r["verdict"] for r in _ledger_lines(project, "ledger.jsonl")}
    assert ledger_records["11"] == "accepted"  # alphaapprove
    assert ledger_records["12"] == "accepted"  # alphaalt
    assert ledger_records["13"] == "gap"  # betaretry
    assert ledger_records["14"] == "gap"  # gammagap
    assert ledger_records["15"] == "accepted"  # deltabanana
    assert ledger_records["16"] == "gap"  # epsilonyes
    assert ledger_records["17"] == "gap"  # zetano
    assert ledger_records["18"] == "accepted"  # etareject

    # --- the Gaps sheet/csv: report_kinds + gap verdicts only ---------------
    workbook = openpyxl.load_workbook(out_path)
    gaps_ws = workbook["Gaps"]
    gap_row_ids = {
        str(row[0].value) for row in gaps_ws.iter_rows(min_row=3) if row[0].value is not None
    }
    assert gap_row_ids == {"13", "14", "16", "17"}

    gaps_csv_path = out_path.parent / f"{out_path.stem.removesuffix('-tagged')}-gaps.csv"
    with gaps_csv_path.open(encoding="utf-8-sig", newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    assert {row["Row ID"] for row in csv_rows} == {"13", "14", "16", "17"}

    # --- edit the workbook (one workbook object for both sheets!) -----------
    workbook, review_ws = _open_review(out_path)
    _set_decision(review_ws, "11", "Approve")
    _set_decision(review_ws, "12", "Use alternative 2")
    _set_decision(review_ws, "13", "Retry with hint", "this is about software testing")
    _set_decision(review_ws, "14", "Mark as gap")
    _set_decision(review_ws, "15", "banana")
    _set_decision(review_ws, "18", "Reject")

    gaps_ws = workbook["Gaps"]
    gaps_row_id_col = _header_index(gaps_ws, "Row ID")
    gaps_decision_col = _header_index(gaps_ws, "Reviewer confirms gap? ▼")
    gaps_ws.cell(
        row=_row_by_id(gaps_ws, "16", gaps_row_id_col), column=gaps_decision_col, value="Yes"
    )
    gaps_ws.cell(
        row=_row_by_id(gaps_ws, "17", gaps_row_id_col), column=gaps_decision_col, value="No"
    )
    workbook.save(out_path)

    # --- second run: applies decisions, retries only beta/zeta/eta ----------
    registry2, suite2 = _registry()
    client2 = ReviewScriptedClient()
    second = run(project, registry=registry2, opts=RunOptions(no_llm=False), llm_client=client2)

    # approve(11), use_alt(12), retry(13), mark_gap(14), mark_gap-via-yes(16),
    # retry-via-no(17), reject(18) = 7 applied; "banana" (15) is unrecognised
    # and never becomes an applied decision at all.
    assert second.reviews_applied == 7
    assert second.retried == 3  # betaretry (retry), zetano (no), etareject (reject)
    assert any("banana" in w for w in second.warnings)

    choose_calls = [c for c in client2.calls if "<row id=" not in c[1].content]
    assert len(choose_calls) == 3
    assert any("this is about software testing" in c[1].content for c in choose_calls)
    # The hint is searched too, not only shown to the model.
    assert any(text == "this is about software testing" for text, _kind in suite2.calls)

    reviews = _ledger_lines(project, "reviews.jsonl")
    assert len(reviews) == 7
    actions = sorted(r["action"] for r in reviews)
    assert actions == ["approve", "mark_gap", "mark_gap", "reject", "retry", "retry", "use_alt"]

    # --- the new workbook: blank decisions, previous-decision column, history
    second_workbook, second_review = _open_review(out_path)
    row_id_col = _header_index(second_review, "Row ID")
    decision_col = _header_index(second_review, "Decision ▼")
    previous_col = _header_index(second_review, "Previous decision")
    for row_id in ("11", "12", "13", "14", "18"):
        row = _row_by_id(second_review, row_id, row_id_col)
        assert second_review.cell(row=row, column=decision_col).value in (None, "")
    # The unrecognised "banana" stays visible so the reviewer can fix it.
    banana_row = _row_by_id(second_review, "15", row_id_col)
    assert second_review.cell(row=banana_row, column=decision_col).value == "banana"
    approve_row = _row_by_id(second_review, "11", row_id_col)
    assert "Approved" in str(second_review.cell(row=approve_row, column=previous_col).value)

    history_dir = out_path.parent / "history"
    assert history_dir.is_dir()
    assert any(history_dir.iterdir())

    # --- precision: alpha-approve (accepted->approved), eta (accepted->rejected)
    summary_ws = second_workbook["Summary"]
    summary_text = "\n".join(
        str(cell.value) for row in summary_ws.iter_rows() for cell in row if cell.value is not None
    )
    assert "Reviewed: " in summary_text
    assert "Accepted precision" in summary_text

    # --- third run: everything settled, zero model calls ---------------------
    registry3, suite3 = _registry()
    client3 = ReviewScriptedClient()
    third = run(project, registry=registry3, opts=RunOptions(no_llm=False), llm_client=client3)
    assert client3.calls == []
    assert third.model_calls == 0
    assert third.reviews_applied == 0
    assert third.retried == 0


def test_locked_workbook_raises_before_any_decision_is_read(project: Path) -> None:
    registry, _suite = _registry()
    client = ReviewScriptedClient()
    run(project, registry=registry, opts=RunOptions(no_llm=False, fresh=True), llm_client=client)

    doc_key = read_document(project / "inputs" / "standard.json").key
    out_dir = project / "out"
    (out_dir / f"~${doc_key}-tagged.xlsx").write_text("lock")

    client2 = ReviewScriptedClient()
    with pytest.raises(WorkbookOpenError):
        run(project, registry=_registry()[0], opts=RunOptions(no_llm=False), llm_client=client2)
    assert client2.calls == []


def test_an_unrecognised_decision_stays_in_its_cell(project: Path) -> None:
    """A typo must not vanish with the consumed decisions: it stays visible to fix."""
    registry, _ = _registry()
    first = run(
        project,
        registry=registry,
        opts=RunOptions(no_llm=False, fresh=True),
        llm_client=ReviewScriptedClient(),
    )
    out_path = Path(first.outputs[0])
    workbook, ws = _open_review(out_path)
    _set_decision(ws, "15", "maybe later", hint="ask Alejandro")
    _set_decision(ws, "11", "Approve")
    workbook.save(out_path)

    second = run(
        project,
        registry=_registry()[0],
        opts=RunOptions(no_llm=False),
        llm_client=ReviewScriptedClient(),
    )
    assert any("unknown decision 'maybe later'" in w for w in second.warnings)

    _, ws = _open_review(out_path)
    decision_col, hint_col = _header_index(ws, "Decision ▼"), _header_index(ws, "Hint")
    typo_row = _row_by_id(ws, "15", _header_index(ws, "Row ID"))
    approved_row = _row_by_id(ws, "11", _header_index(ws, "Row ID"))
    assert ws.cell(row=typo_row, column=decision_col).value == "maybe later"  # type: ignore[attr-defined]
    assert ws.cell(row=typo_row, column=hint_col).value == "ask Alejandro"  # type: ignore[attr-defined]
    assert not ws.cell(row=approved_row, column=decision_col).value  # type: ignore[attr-defined]


# --- a retry is never silently lost ---------------------------------------------------


class _InterruptOnHint(ReviewScriptedClient):
    """The run is stopped (Ctrl-C) as soon as the retried row reaches the model."""

    def complete(
        self, messages: list[Message], *, tools: list[dict[str, object]] | None = None
    ) -> LLMResult:
        if "this is about software testing" in messages[1].content:
            raise KeyboardInterrupt
        return super().complete(messages, tools=tools)


class _BrokenOnHint(ReviewScriptedClient):
    """The model returns unusable text for the retried row (verdict `error`)."""

    def complete(
        self, messages: list[Message], *, tools: list[dict[str, object]] | None = None
    ) -> LLMResult:
        if "this is about software testing" in messages[1].content:
            self.calls.append(messages)
            return LLMResult(
                text="not json", provider=self.provider, model=self.model,
                usage=LLMUsage(input_tokens=5, output_tokens=5),
            )  # fmt: skip
        return super().complete(messages, tools=tools)


def _first_run_with_retry_on_13(project: Path) -> Path:
    first = run(
        project,
        registry=_registry()[0],
        opts=RunOptions(no_llm=False, fresh=True),
        llm_client=ReviewScriptedClient(),
    )
    out_path = Path(first.outputs[0])
    workbook, ws = _open_review(out_path)
    _set_decision(ws, "13", "Retry with hint", "this is about software testing")
    workbook.save(out_path)
    return out_path


def _hint_searched(suite: KeywordSuite) -> bool:
    return any(text == "this is about software testing" for text, _kind in suite.calls)


def test_a_retry_interrupted_mid_run_runs_on_the_next_run(project: Path) -> None:
    _first_run_with_retry_on_13(project)
    with pytest.raises(KeyboardInterrupt):
        run(
            project,
            registry=_registry()[0],
            opts=RunOptions(no_llm=False),
            llm_client=_InterruptOnHint(),
        )

    registry, suite = _registry()
    client = ReviewScriptedClient()
    again = run(project, registry=registry, opts=RunOptions(no_llm=False), llm_client=client)

    assert again.retried == 1
    assert _hint_searched(suite)
    assert any("this is about software testing" in c[1].content for c in client.calls)


def test_a_retry_that_errors_is_retried_with_its_hint(project: Path) -> None:
    _first_run_with_retry_on_13(project)
    broken = run(
        project, registry=_registry()[0], opts=RunOptions(no_llm=False), llm_client=_BrokenOnHint()
    )
    assert broken.counts["esco"].get("error") == 1

    registry, suite = _registry()
    again = run(
        project, registry=registry, opts=RunOptions(no_llm=False), llm_client=ReviewScriptedClient()
    )

    assert again.retried == 1
    assert _hint_searched(suite)
    assert again.counts["esco"].get("error", 0) == 0


def test_a_retry_outside_the_selected_rows_waits_for_them(project: Path) -> None:
    _first_run_with_retry_on_13(project)
    partial = run(
        project,
        registry=_registry()[0],
        opts=RunOptions(no_llm=False, rows={"11"}),
        llm_client=ReviewScriptedClient(),
    )
    assert partial.retried == 0
    assert any("outside this run's selection" in w for w in partial.warnings)

    registry, suite = _registry()
    full = run(
        project, registry=registry, opts=RunOptions(no_llm=False), llm_client=ReviewScriptedClient()
    )
    assert full.retried == 1
    assert _hint_searched(suite)


class _BetaStaysAGap(ReviewScriptedClient):
    """Row 13 finds nothing even after the retry, so `--refresh` would pick it up."""

    @staticmethod
    def _choose_reply(user: str) -> str:
        if "betaretry" in user.casefold():
            return json.dumps({**_NONE_FIT, "missing_skill": "a missing skill"})
        return ReviewScriptedClient._choose_reply(user)


def test_refresh_keeps_a_row_the_reviewer_retried(project: Path) -> None:
    _first_run_with_retry_on_13(project)
    run(
        project,
        registry=_registry()[0],
        opts=RunOptions(no_llm=False),
        llm_client=_BetaStaysAGap(),
    )
    before = [r for r in _ledger_lines(project, "ledger.jsonl") if r["row_id"] == "13"][-1]
    assert before["verdict"] == "gap"

    run(
        project,
        registry=_registry()[0],
        opts=RunOptions(no_llm=False, refresh=True),
        llm_client=_BetaStaysAGap(),
    )
    after = [r for r in _ledger_lines(project, "ledger.jsonl") if r["row_id"] == "13"][-1]

    assert after["fingerprint"] == before["fingerprint"]
    assert "this is about software testing" in str(after["hint"])


def test_a_changed_profile_is_not_answered_from_the_old_answers(project: Path) -> None:
    """Reuse must notice that the cleaned text a unit is searched with has changed."""
    run(
        project,
        registry=_registry()[0],
        opts=RunOptions(no_llm=False, fresh=True),
        llm_client=ReviewScriptedClient(),
    )
    (project / "tag.yaml").write_text("groups: false\ncleaners:\n  - ' procedure$'\n")

    again = run(
        project,
        registry=_registry()[0],
        opts=RunOptions(no_llm=False),
        llm_client=ReviewScriptedClient(),
    )

    assert again.searched > 0


def test_the_second_look_reports_its_own_progress(project: Path) -> None:
    ticks: list[tuple[str, int, int]] = []
    run(
        project,
        registry=_registry()[0],
        opts=RunOptions(no_llm=False, fresh=True),
        llm_client=ReviewScriptedClient(),
        progress=lambda _message: None,
        step_progress=lambda label, done, total: ticks.append((label, done, total)),
    )
    second_look = [(done, total) for label, done, total in ticks if "step 4/4" in label]
    assert second_look == [(n, len(_GAP_KEYWORDS)) for n in range(1, len(_GAP_KEYWORDS) + 1)]


def test_a_project_setting_changes_the_retry_limit(project: Path) -> None:
    (project / "tag.yaml").write_text("groups: false\nmax_retries: 0\n")
    _first_run_with_retry_on_13(project)

    summary = run(
        project,
        registry=_registry()[0],
        opts=RunOptions(no_llm=False),
        llm_client=ReviewScriptedClient(),
    )

    assert summary.retried == 0
    assert any("not retried again" in w for w in summary.warnings)
    out_path = Path(summary.outputs[0])
    _, ws = _open_review(out_path)
    row = _row_by_id(ws, "13", _header_index(ws, "Row ID"))
    assert ws.cell(row=row, column=_header_index(ws, "Decision ▼")).value == "Retry with hint"  # type: ignore[attr-defined]
    previous = str(ws.cell(row=row, column=_header_index(ws, "Previous decision")).value)  # type: ignore[attr-defined]
    assert previous.startswith("Not retried")
