"""Offline e2e of the model steps (PR3): rewrite + choose + verdict over a run.

A scripted client answers from the *content* of each prompt (never call
order), so the same fixture gives the same verdicts whether `--workers` is 1
or N. The fixture has five leaf units behind one keyword each:

- "alpha" (performance) -> exact, corroborated -> accepted
- "beta"  (performance) -> partial, with covers -> adjacent
- "gamma" (performance) -> none, still none after widening -> gap (reported)
- "delta" (definition)  -> none, still none after widening -> gap (not reported:
  `definition` isn't in `report_kinds`)
- "epsilon" (performance) -> the suite finds no candidates at all -> gap,
  with *no* model call (rule: zero candidates is never worth a model call)
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import openpyxl
import pytest

from talent_angels.llm.protocol import LLMResult, LLMUsage, Message
from talent_angels.suites.schema import SuiteSchema
from talent_angels.tagging.models import RunOptions
from talent_angels.tagging.runner import run
from tests.fakes.tagging_suite import FakeCandidate, FakeNode, FakeToolResult, table_registry

_SCHEMA = SuiteSchema(
    skill_rel_types=("HAS_SKILL",),
    optional_rel_values=frozenset(),
    group_rel_type=None,
    group_node_kinds=frozenset(),
)

_DISTINCTIVE_DESCRIPTION = "DISTINCTIVE_SUITE_DESCRIPTION_TEXT_12345"


# --- fixture construction (a tiny {n, r, m} tree built in code) -------------


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


_TREE = [
    # (child_id, child_labels, child_text, parent_id, parent_labels, parent_text)
    (1, ["Section"], "Section I", 0, ["Document"], "Fixture Model Steps Doc"),
    (2, ["Item"], "Title: Fixture Model Steps Doc", 1, ["Section"], "Section I"),
    (10, ["Item"], "Performances", 1, ["Section"], "Section I"),
    (11, ["Item"], "Alpha: handle the alpha procedure", 10, ["Item"], "Performances"),
    (12, ["Item"], "Beta: handle the beta procedure", 10, ["Item"], "Performances"),
    (13, ["Item"], "Gamma: handle the gamma procedure", 10, ["Item"], "Performances"),
    (14, ["Item"], "Epsilon: handle the epsilon procedure", 10, ["Item"], "Performances"),
    (20, ["Item"], "Glossary", 1, ["Section"], "Section I"),
    (21, ["Item"], "Delta: glossary definition of delta", 20, ["Item"], "Glossary"),
]


def _build_fixture() -> list[dict[str, object]]:
    rows = []
    edge_id = 1000
    for child_id, child_labels, child_text, parent_id, parent_labels, parent_text in _TREE:
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
    # Simplify the unit count: no group-level ("heading + its steps") units.
    (project_dir / "tag.yaml").write_text("groups: false\n")
    return project_dir


# --- fake suite: one fixed candidate per (unit kind), none for "epsilon" ----


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
        if "epsilon" in text.casefold():
            return FakeToolResult(warnings=["not_found"])
        node = FakeNode(
            id=f"{self.name}:{kind}:fixed",
            kind="Skill",
            label=text,
            properties={"description": _DISTINCTIVE_DESCRIPTION},
        )
        return FakeToolResult(
            candidates=[FakeCandidate(node=node, confidence=0.9, method="exact_pref")]
        )

    def get_neighbors(self, node_id: str, rel_types: list[str] | None = None) -> FakeToolResult:
        return FakeToolResult(warnings=["node_not_found"])


def _registry() -> tuple[object, KeywordSuite]:
    esco = KeywordSuite("esco")
    return table_registry({"esco": esco}, default="esco"), esco


# --- scripted model client: answers from prompt content, not call order ----


_REWRITE_ROW_RE = re.compile(r'<row id="(r\d+)"[^>]*>([^<]*)</row>')


class ContentScriptedClient:
    provider = "litellm"
    model = "test-model"

    def __init__(self) -> None:
        self.calls: list[list[Message]] = []

    def complete(
        self, messages: list[Message], *, tools: list[dict[str, object]] | None = None
    ) -> LLMResult:
        self.calls.append(messages)
        user = messages[1].content
        if "<row id=" in user:
            text = self._rewrite_reply(user)
        else:
            text = self._choose_reply(user)
        return LLMResult(
            text=text,
            provider=self.provider,
            model=self.model,
            usage=LLMUsage(input_tokens=5, output_tokens=5),
        )

    @staticmethod
    def _rewrite_reply(user: str) -> str:
        rows = {}
        for row_id, row_text in _REWRITE_ROW_RE.findall(user):
            keyword = row_text.strip().split(":")[0].strip().split()[0].lower()
            rows[row_id] = {"labels": [f"{keyword} label"], "description": f"{keyword} description"}
        return json.dumps({"rows": rows})

    @staticmethod
    def _choose_reply(user: str) -> str:
        lowered = user.casefold()
        if "alpha" in lowered:
            return json.dumps({"fit": "exact", "picks": [{"n": 1}], "reason": "alpha matches"})
        if "beta" in lowered:
            return json.dumps(
                {
                    "fit": "partial",
                    "picks": [{"n": 1, "covers": "partially covers beta"}],
                    "reason": "beta is adjacent",
                }
            )
        if "gamma" in lowered:
            return json.dumps(
                {
                    "fit": "none",
                    "picks": [],
                    "reason": "no match for gamma",
                    "missing_skill": "gamma skill",
                }
            )
        if "delta" in lowered:
            return json.dumps(
                {
                    "fit": "none",
                    "picks": [],
                    "reason": "no match for delta",
                    "missing_skill": "delta skill",
                }
            )
        return json.dumps({"fit": "none", "picks": [], "reason": "no match"})


def _ledger_text(project_dir: Path) -> str:
    return (project_dir / ".ta-tag" / "ledger.jsonl").read_text(encoding="utf-8")


def _verdicts_by_unit_suite(project_dir: Path) -> dict[tuple[str, str], str]:
    out: dict[tuple[str, str], str] = {}
    for line in _ledger_text(project_dir).splitlines():
        record = json.loads(line)
        out[(record["unit_key"], record["suite"])] = record["verdict"]
    return out


# --- tests -------------------------------------------------------------


def test_model_on_produces_the_expected_verdict_mix(project: Path) -> None:
    registry, _suite = _registry()
    client = ContentScriptedClient()
    summary = run(
        project, registry=registry, opts=RunOptions(no_llm=False, fresh=True), llm_client=client
    )

    verdicts = _verdicts_by_unit_suite(project)
    by_verdict = {}
    for line in _ledger_text(project).splitlines():
        record = json.loads(line)
        by_verdict.setdefault(record["verdict"], []).append(record["clean_text"])

    assert any("alpha" in text for text in by_verdict.get("accepted", []))
    assert any("beta" in text for text in by_verdict.get("adjacent", []))
    gap_texts = by_verdict.get("gap", [])
    assert any("gamma" in text for text in gap_texts)
    assert any("delta" in text for text in gap_texts)
    assert any("epsilon" in text for text in gap_texts)
    assert summary.model_calls > 0
    assert len(verdicts) == 5


def test_zero_candidates_never_calls_the_model(project: Path) -> None:
    registry, _suite = _registry()
    client = ContentScriptedClient()
    run(project, registry=registry, opts=RunOptions(no_llm=False, fresh=True), llm_client=client)

    # Epsilon may appear in a *rewrite* batch (it shares a parent with alpha/
    # beta/gamma), but never in a *choose* call — zero candidates never reach
    # the choose step at all.
    choose_calls = [c for c in client.calls if "<row id=" not in c[1].content]
    assert all("epsilon" not in c[1].content.casefold() for c in choose_calls)


def test_gaps_jsonl_only_lists_report_kinds(project: Path) -> None:
    registry, _suite = _registry()
    client = ContentScriptedClient()
    run(project, registry=registry, opts=RunOptions(no_llm=False, fresh=True), llm_client=client)

    gaps_path = project / ".ta-tag" / "gaps.jsonl"
    gaps = [json.loads(line) for line in gaps_path.read_text(encoding="utf-8").splitlines()]
    kinds = {g["kind"] for g in gaps}
    texts = " ".join(g["clean_text"] for g in gaps)
    assert "definition" not in kinds  # delta's kind, excluded from report_kinds
    assert "gamma" in texts
    assert "delta" not in texts


def test_no_candidate_descriptions_leak_into_the_ledger_or_workbook(project: Path) -> None:
    registry, _suite = _registry()
    client = ContentScriptedClient()
    summary = run(
        project, registry=registry, opts=RunOptions(no_llm=False, fresh=True), llm_client=client
    )

    assert _DISTINCTIVE_DESCRIPTION not in _ledger_text(project)

    [output] = summary.outputs
    workbook = openpyxl.load_workbook(output)
    for sheet in workbook.worksheets:
        for row in sheet.iter_rows():
            for cell in row:
                assert cell.value != _DISTINCTIVE_DESCRIPTION
                assert not (isinstance(cell.value, str) and _DISTINCTIVE_DESCRIPTION in cell.value)


def test_second_run_makes_zero_model_calls_and_zero_searches(project: Path) -> None:
    registry, _suite = _registry()
    run(
        project,
        registry=registry,
        opts=RunOptions(no_llm=False, fresh=True),
        llm_client=ContentScriptedClient(),
    )

    registry2, _suite2 = _registry()
    client2 = ContentScriptedClient()
    second = run(project, registry=registry2, opts=RunOptions(no_llm=False), llm_client=client2)

    assert client2.calls == []
    assert second.searched == 0
    assert second.model_calls == 0
    assert second.reused == second.units


def test_refresh_only_retries_non_final_verdicts(project: Path) -> None:
    registry, _suite = _registry()
    run(
        project,
        registry=registry,
        opts=RunOptions(no_llm=False, fresh=True),
        llm_client=ContentScriptedClient(),
    )

    registry2, _suite2 = _registry()
    client2 = ContentScriptedClient()
    refreshed = run(
        project,
        registry=registry2,
        opts=RunOptions(no_llm=False, refresh=True),
        llm_client=client2,
    )

    # Only beta (adjacent), gamma/delta/epsilon (gap) are retried; alpha
    # (accepted) is reused untouched.
    assert refreshed.reused == 1
    assert refreshed.searched == 4
    assert len(client2.calls) > 0


def test_workers_two_matches_workers_one(tmp_path: Path) -> None:
    results = {}
    for workers, label in ((1, "one"), (2, "two")):
        project_dir = tmp_path / f"project-{label}"
        (project_dir / "inputs").mkdir(parents=True)
        (project_dir / "inputs" / "standard.json").write_text(json.dumps(_build_fixture()))
        (project_dir / "tag.yaml").write_text("groups: false\n")

        registry, _suite = _registry()
        client = ContentScriptedClient()
        run(
            project_dir,
            registry=registry,
            opts=RunOptions(no_llm=False, fresh=True, workers=workers),
            llm_client=client,
        )
        results[label] = _verdicts_by_unit_suite(project_dir)

    assert results["one"] == results["two"]


def test_isolation_memory_home_and_runlog_untouched(project: Path) -> None:
    home = Path(os.environ["TA_AGENTS_HOME"])
    runlog_path = Path(os.environ["RUNLOG_PATH"])

    def snapshot() -> list[tuple[str, int, int]]:
        if not home.exists():
            return []
        return sorted(
            (str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in home.rglob("*") if p.is_file()
        )

    before = snapshot()
    registry, _suite = _registry()
    run(
        project,
        registry=registry,
        opts=RunOptions(no_llm=False, fresh=True),
        llm_client=ContentScriptedClient(),
    )
    assert snapshot() == before
    assert not runlog_path.exists()
    assert (project / ".ta-tag" / "runlog.jsonl").is_file()


def test_progress_names_every_phase_and_counts_the_model_step(project: Path) -> None:
    """The model step is most of a real run; it must report progress, not go silent."""
    registry, _ = _registry()
    messages: list[str] = []
    run(
        project,
        registry=registry,  # type: ignore[arg-type]
        opts=RunOptions(no_llm=False, workers=2),
        llm_client=ContentScriptedClient(),
        progress=messages.append,
    )
    text = "\n".join(messages)
    assert "step 1/4 skill phrases" in text
    assert "step 1/4 skill phrases, batch" in text  # step 1 counts its batches too
    assert "step 2/4 rows" in text
    assert "step 3/4 model choosing among candidates" in text


def test_a_live_bar_gets_every_tick_of_each_counted_step(project: Path) -> None:
    registry, _ = _registry()
    ticks: list[tuple[str, int, int]] = []
    run(
        project,
        registry=registry,  # type: ignore[arg-type]
        opts=RunOptions(no_llm=False, workers=2),
        llm_client=ContentScriptedClient(),
        progress=lambda _message: None,
        step_progress=lambda label, done, total: ticks.append((label, done, total)),
    )
    for step in ("step 1/4", "step 2/4", "step 3/4"):
        counts = [(done, total) for label, done, total in ticks if step in label]
        assert counts, step
        assert counts[-1][0] == counts[-1][1]  # each counted step ends full
    searched = [done for label, done, _ in ticks if "step 2/4" in label]
    assert searched == list(range(1, len(searched) + 1))  # one tick per row


def test_without_a_terminal_progress_is_plain_lines(capsys: pytest.CaptureFixture[str]) -> None:
    from talent_angels.tagging.cli import _ProgressView

    with _ProgressView(live=False) as view:
        view.message("doc: step 1/4 skill phrases for 3 rows…")
        view.step("doc: step 2/4 searching taxonomies", 1, 3)  # ignored without a bar
    assert capsys.readouterr().err == "doc: step 1/4 skill phrases for 3 rows…\n"
