"""Offline end-to-end: `runner.run` / `cli.main` / `tui` dispatch over the fixture.

Three fake suites stand in for the registry: `esco` answers every routed kind,
`onet` supports none of them (-> `n/a` for every unit), `down` fails to open
at all (-> `unavailable` for every unit, never cached).
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import openpyxl
import pytest

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

FIXTURE = Path(__file__).parent / "fixtures" / "tagging" / "standard_small.json"

_SEARCHABLE_KINDS = frozenset({"skill", "task", "knowledge", "work_style", "occupation"})


@dataclass
class AnyKindSuite:
    """Answers every query for a configured set of kinds with one fixed hit."""

    name: str
    hit_kinds: frozenset[str] = field(default_factory=lambda: frozenset(_SEARCHABLE_KINDS))
    calls: list[tuple[str, str | None]] = field(default_factory=list)

    @property
    def suite_schema(self) -> SuiteSchema:
        return _SCHEMA

    def search_nodes(self, text: str, kind: str | None = None) -> FakeToolResult:
        self.calls.append((text, kind))
        if kind is None or kind not in self.hit_kinds:
            return FakeToolResult(warnings=[f"unknown_kind:{kind}"])
        node_kind = "Occupation" if kind == "occupation" else "Skill"
        node = FakeNode(id=f"{self.name}:{kind}:fixed", kind=node_kind, label=text)
        return FakeToolResult(
            candidates=[FakeCandidate(node=node, confidence=0.9, method="exact_pref")]
        )

    def get_neighbors(self, node_id: str, rel_types: list[str] | None = None) -> FakeToolResult:
        return FakeToolResult(warnings=["node_not_found"])


@pytest.fixture
def project(tmp_path: Path) -> Path:
    project_dir = tmp_path / "project"
    (project_dir / "inputs").mkdir(parents=True)
    shutil.copy(FIXTURE, project_dir / "inputs" / "standard.json")
    return project_dir


def _registry():
    esco = AnyKindSuite("esco")
    onet = AnyKindSuite("onet", hit_kinds=frozenset())
    return table_registry(
        {"esco": esco, "onet": onet}, default="esco", broken={"down": ConnectionError("down")}
    )


def test_first_run_produces_expected_verdict_mix(project: Path) -> None:
    registry = _registry()
    summary = run(project, registry=registry, opts=RunOptions())

    assert summary.units > 0
    assert summary.reused == 0
    assert summary.searched == summary.units * 3
    assert summary.counts["onet"].get("n/a") == summary.units
    assert summary.counts["down"].get("unavailable") == summary.units
    assert summary.counts["esco"].get("review", 0) > 0

    [output] = summary.outputs
    assert Path(output).is_file()
    workbook = openpyxl.load_workbook(output)
    assert workbook.sheetnames == ["Summary", "Review", "Standard", "Gaps"]


def test_second_run_reuses_cacheable_suites(project: Path) -> None:
    registry = _registry()
    first = run(project, registry=registry, opts=RunOptions())

    registry2 = _registry()
    second = run(project, registry=registry2, opts=RunOptions())

    # `esco`/`onet` verdicts (review/n-a) are cacheable; `down` (unavailable)
    # is rule 1 of tag-v1 ("not cached") and is re-attempted every run.
    assert second.reused == first.units * 2
    assert second.searched == first.units


def test_fresh_ignores_the_ledger(project: Path) -> None:
    registry = _registry()
    run(project, registry=registry, opts=RunOptions())

    registry2 = _registry()
    second = run(project, registry=registry2, opts=RunOptions(fresh=True))
    assert second.reused == 0
    assert second.searched == second.units * 3


def test_locked_workbook_raises_workbook_open_error(project: Path) -> None:
    doc = read_document(project / "inputs" / "standard.json")
    out_dir = project / "out"
    out_dir.mkdir(parents=True)
    (out_dir / f"~${doc.key}-tagged.xlsx").write_text("lock")

    with pytest.raises(WorkbookOpenError):
        run(project, registry=_registry(), opts=RunOptions())


def test_cli_main_returns_0_and_prints_outputs(
    project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from talent_angels.tagging.cli import main as tag_main

    exit_code = tag_main([str(project), "--no-llm"], registry=_registry())
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "tagged.xlsx" in out


def test_cli_prints_reviews_applied_and_retried_when_nonzero(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from talent_angels.tagging.cli import _print_summary
    from talent_angels.tagging.models import RunSummary

    _print_summary(RunSummary(project="p", reviews_applied=3, retried=1))
    out = capsys.readouterr().out
    assert "reviews_applied=3 retried=1" in out

    capsys.readouterr()
    _print_summary(RunSummary(project="p"))
    out = capsys.readouterr().out
    assert "reviews_applied" not in out


def test_tui_dispatches_tag_without_opening_the_chat_loop(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from talent_angels.tui import app as tui_app

    def _boom(*_args: object, **_kwargs: object) -> str:
        raise AssertionError("the chat loop must not run for `tag`")

    monkeypatch.setattr(tui_app.Console, "input", _boom)
    exit_code = tui_app.main(["tag", str(project), "--no-llm"], registry=_registry())
    assert exit_code == 0


def test_isolation_memory_home_and_runlog_untouched(project: Path) -> None:
    home = Path(os.environ["TA_AGENTS_HOME"])
    runlog_path = Path(os.environ["RUNLOG_PATH"])

    def snapshot() -> list[tuple[str, int, int]]:
        # Names, sizes and mtimes: a rewritten USER.md or memory.db must fail too.
        if not home.exists():
            return []
        return sorted(
            (str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in home.rglob("*") if p.is_file()
        )

    before = snapshot()
    run(project, registry=_registry(), opts=RunOptions())
    assert snapshot() == before
    assert not runlog_path.exists()
    assert (project / ".ta-tag" / "runlog.jsonl").is_file()


@dataclass
class FailsOnSearchSuite(AnyKindSuite):
    """Opens fine but every search raises — e.g. Neo4j dropping mid-run."""

    def search_nodes(self, text: str, kind: str | None = None) -> FakeToolResult:
        raise ConnectionError("graph went away")


def test_a_suite_failing_mid_run_is_unavailable_not_a_crash(project: Path) -> None:
    registry = table_registry(
        {"esco": AnyKindSuite("esco"), "flaky": FailsOnSearchSuite("flaky")}, default="esco"
    )
    summary = run(project, registry=registry, opts=RunOptions())
    assert summary.counts["flaky"] == {"unavailable": summary.units}
    assert summary.counts["esco"].get("review", 0) > 0


def test_the_same_document_twice_is_tagged_once(project: Path) -> None:
    shutil.copy(FIXTURE, project / "inputs" / "renamed-copy.json")
    summary = run(project, registry=_registry(), opts=RunOptions())
    assert len(summary.documents) == 1
    assert any("same document" in warning for warning in summary.warnings)


def test_a_suite_down_for_every_row_is_called_out(project: Path) -> None:
    summary = run(project, registry=_registry(), opts=RunOptions())
    assert any(w.startswith("down: unavailable for every row") for w in summary.warnings)


def test_a_plain_rerun_keeps_no_history_copy(project: Path) -> None:
    run(project, registry=_registry(), opts=RunOptions())
    run(project, registry=_registry(), opts=RunOptions())
    assert not (project / "out" / "history").exists()


def test_a_newer_numbers_save_stops_the_run_before_any_work(project: Path) -> None:
    """Numbers saves edits as `.numbers`: the decisions must not be silently ignored."""
    import time

    run(project, registry=_registry(), opts=RunOptions())
    doc = read_document(project / "inputs" / "standard.json")
    out_path = project / "out" / f"{doc.key}-tagged.xlsx"
    time.sleep(0.01)
    numbers_copy = out_path.with_suffix(".numbers")
    numbers_copy.write_text("numbers document")

    with pytest.raises(WorkbookOpenError, match="Export To > Excel"):
        run(project, registry=_registry(), opts=RunOptions())

    # Once exported over the xlsx (now the newer file), the run goes ahead.
    os.utime(out_path, None)
    os.utime(numbers_copy, (0, 0))
    run(project, registry=_registry(), opts=RunOptions())


# --- macOS + Numbers ---------------------------------------------------------------


def test_mac_system_files_in_inputs_are_ignored_quietly(project: Path) -> None:
    inputs = project / "inputs"
    (inputs / ".DS_Store").write_bytes(b"\x00\x00\x00\x01Bud1")
    (inputs / "._standard.json").write_bytes(b"\x00\x05\x16\x07")
    (inputs / "Icon\r").write_bytes(b"")
    (inputs / "~$standard.xlsx").write_bytes(b"lock")

    summary = run(project, registry=_registry(), opts=RunOptions())

    assert summary.units > 0
    assert not any("recognizes" in w for w in summary.warnings)


@pytest.mark.parametrize("as_package", [False, True])
def test_a_numbers_input_says_how_to_export_it(tmp_path: Path, as_package: bool) -> None:
    from talent_angels.tagging.readers.base import ReaderError

    inputs = tmp_path / "project" / "inputs"
    inputs.mkdir(parents=True)
    target = inputs / "course.numbers"
    if as_package:  # older Numbers saves a folder that Finder shows as one file
        (target / "Index").mkdir(parents=True)
    else:
        target.write_bytes(b"PK\x03\x04")

    with pytest.raises(ReaderError, match="Export To > Excel"):
        run(tmp_path / "project", registry=_registry(), opts=RunOptions())


def test_an_export_saved_under_another_name_is_pointed_out(project: Path) -> None:
    import time

    run(project, registry=_registry(), opts=RunOptions())
    doc = read_document(project / "inputs" / "standard.json")
    out_path = project / "out" / f"{doc.key}-tagged.xlsx"
    time.sleep(0.01)
    stray = out_path.with_name(f"{out_path.stem} 2.xlsx")
    shutil.copy(out_path, stray)

    summary = run(project, registry=_registry(), opts=RunOptions())

    assert any(stray.name in w and "only" in w for w in summary.warnings)


def test_a_reference_document_in_inputs_gets_a_plain_note(project: Path) -> None:
    (project / "inputs" / "Standard original.docx").write_bytes(b"PK\x03\x04")
    (project / "inputs" / "context").mkdir()
    (project / "inputs" / "context" / "notes.pdf").write_bytes(b"%PDF")

    summary = run(project, registry=_registry(), opts=RunOptions())

    notes = [w for w in summary.warnings if "Standard original.docx" in w]
    assert notes and "reference document, not tagged" in notes[0]
    assert not any("notes.pdf" in w for w in summary.warnings)  # context/ is never scanned
