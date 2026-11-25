"""Esc stops a long slash command at the next item: nothing written, session unchanged."""

from __future__ import annotations

import io
import shutil
import sys
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import pytest
from rich.console import Console

from talent_angels.engine.registry import Result
from talent_angels.llm import protocol
from talent_angels.llm.protocol import TurnCancelled, bind_cancel, check_cancelled
from talent_angels.memory.paths import outputs_dir
from talent_angels.session import slash
from talent_angels.session.copy import STOPPED
from talent_angels.session.kernel import handle_line
from talent_angels.session.store import new_session
from talent_angels.skills.connect import discover_report
from talent_angels.skills.connect.discover import discover
from talent_angels.suites import SuiteRegistry
from talent_angels.tagging import runner
from talent_angels.tagging.ledger import Ledger
from talent_angels.tagging.rewrite import Rewrite
from talent_angels.tui import working
from tests.fakes.suite import suite_factory
from tests.fakes.tagging_suite import table_registry
from tests.test_discover import PathSuite, _graph, _items, _registry
from tests.test_tagging_run import FIXTURE, AnyKindSuite
from tests.test_tagging_run import _registry as tag_registry


@pytest.fixture
def esc() -> Iterator[threading.Event]:
    """The reader's Esc: bound as the TUI binds it, reset afterwards."""
    event = threading.Event()
    bind_cancel(event)
    yield event
    protocol._CANCEL.set(None)


def _no_runner(*_a: object, **_k: object) -> None:
    raise AssertionError("a slash command must not run a chat turn")


def _run(line: str, registry: object) -> tuple[str, object]:
    state = new_session()
    before = state.model_dump()
    reply = handle_line(state, line, runner=_no_runner, registry=registry)  # type: ignore[arg-type]
    return reply.text, (before, state.model_dump())


def _written() -> list[Path]:
    return sorted(p for p in outputs_dir().glob("*/*") if p.is_file())


def test_a_slow_command_stops_at_the_next_boundary_writes_nothing_and_leaves_the_session(
    esc: threading.Event, monkeypatch: pytest.MonkeyPatch
) -> None:
    done: list[int] = []

    def slow(_argument: str | None, _registry: SuiteRegistry) -> list[str]:
        for item in range(10):
            check_cancelled()  # the boundary
            done.append(item)
            if item == 2:
                esc.set()  # Esc while item 2 is being worked on
        raise AssertionError("never reached: the loop must stop")

    monkeypatch.setitem(slash._RUNNERS, "discover", slow)
    text, (before, after) = _run("/discover anything.csv", None)

    assert text == STOPPED == "Stopped. Nothing was written."
    assert done == [0, 1, 2]  # item 2 finished, item 3 never started
    assert before == after and after["transcript"] == []  # not even the typed line is kept
    assert _written() == []


def test_without_esc_the_same_runner_completes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(slash._RUNNERS, "discover", lambda *_a: ["all done"])
    assert _run("/discover x.csv", None)[0] == "all done"


def test_turn_cancelled_is_not_an_exception_so_no_catch_all_can_eat_it() -> None:
    assert not issubclass(TurnCancelled, Exception)
    check_cancelled()  # no event bound: a no-op (the CLI, the API, tests)


class _Esc(PathSuite):
    """A graph that presses Esc after ``after`` neighbour lookups."""

    def __init__(self, base: PathSuite, event: threading.Event, after: int) -> None:
        super().__init__(list(base.nodes.values()), base.edges)
        self.event, self.after = event, after

    def get_neighbors(self, node_id: str, rel_types: list[str] | None = None):  # type: ignore[override]
        if len(self.calls) + 1 >= self.after:
            self.event.set()
        return super().get_neighbors(node_id, rel_types)


def test_discover_stops_between_items_and_writes_no_files(
    esc: threading.Event, tmp_path: Path
) -> None:
    items = tmp_path / "items.txt"
    items.write_text("web developer\nJavaScript\nPython\nSQL\ndata analyst\n", encoding="utf-8")
    graph = _Esc(_graph(), esc, after=3)
    reg = SuiteRegistry({"test": suite_factory("test", graph)}, default="test")

    text, (before, after) = _run(f"/discover {items} --suites test --depth 1", reg)

    assert text == STOPPED and before == after
    assert _written() == []
    assert len(graph.calls) < 12  # it stopped early, it did not finish the map


def test_the_discover_core_raises_at_the_first_boundary_once_esc_is_set(
    esc: threading.Event,
) -> None:
    esc.set()
    graph = _graph()
    reg = SuiteRegistry({"test": suite_factory("test", graph)}, default="test")
    with pytest.raises(TurnCancelled):
        discover(reg, _items("web developer", "JavaScript"), depth=1, breadth=4)
    assert graph.calls == []  # nothing was mapped


def _tag_project(tmp_path: Path) -> Path:
    project = tmp_path / "project"
    (project / "inputs").mkdir(parents=True)
    shutil.copy(FIXTURE, project / "inputs" / "standard.json")
    return project


def test_tag_stops_at_a_unit_and_never_overwrites_the_workbook_with_a_partial_one(
    esc: threading.Event, tmp_path: Path
) -> None:
    project = _tag_project(tmp_path)
    text, _ = _run(f"/tag {project} --no-llm --fresh", tag_registry())
    assert ".xlsx" in text
    [workbook] = project.rglob("*-tagged.xlsx")
    finished = workbook.read_bytes()

    class PressesEsc(AnyKindSuite):
        def search_nodes(self, text: str, kind: str | None = None):  # type: ignore[override]
            if len(self.calls) >= 2:
                esc.set()
            return super().search_nodes(text, kind)

    reg = table_registry({"esco": PressesEsc("esco")}, default="esco", broken={})
    text, (before, after) = _run(f"/tag {project} --no-llm --fresh --suites esco", reg)

    assert text == STOPPED and before == after
    assert workbook.read_bytes() == finished  # the earlier complete workbook is untouched


def test_tag_cancelled_before_any_unit_creates_no_workbook(
    esc: threading.Event, tmp_path: Path
) -> None:
    project = _tag_project(tmp_path)
    esc.set()
    text, _ = _run(f"/tag {project} --no-llm", tag_registry())
    assert text == STOPPED and not list(project.rglob("*.xlsx"))


# ---- Esc is only reported once nothing can be written any more -------------------------------


def _fake_terminal(monkeypatch: pytest.MonkeyPatch) -> None:
    """A terminal that presses Esc on the first look at the keyboard."""
    keys = ["\x1b"]

    class Stdin:
        def fileno(self) -> int:
            return 0

        def read(self, _n: int) -> str:
            return keys.pop()

    monkeypatch.setattr(working, "_can_watch_keys", lambda: True)
    monkeypatch.setattr(working.termios, "tcgetattr", lambda _f: None)
    monkeypatch.setattr(working.tty, "setcbreak", lambda _f: None)
    monkeypatch.setattr(working.sys, "stdin", Stdin())
    monkeypatch.setattr(
        working.select, "select", lambda r, _w, _x, _t: (r, [], []) if keys else ([], [], [])
    )


def _slow_writer(cancel: threading.Event, wrote: list[str]):  # noqa: ANN202
    def work() -> str:
        cancel.wait(5)
        time.sleep(0.3)  # the worker is still running when Esc is seen
        wrote.append("file")
        return "done"

    return work


def test_a_file_writing_command_is_waited_for_after_esc(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_terminal(monkeypatch)
    cancel, wrote = threading.Event(), []
    console = Console(file=io.StringIO())
    result = working.run_with_status(
        console, "x", _slow_writer(cancel, wrote), cancel=cancel, wait_on_cancel=True
    )
    assert cancel.is_set() and wrote == ["file"] and result == "done"  # finished before we returned


def test_a_chat_turn_is_still_abandoned_at_once_after_esc(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_terminal(monkeypatch)
    cancel, wrote = threading.Event(), []
    with pytest.raises(working.Cancelled):
        working.run_with_status(
            Console(file=io.StringIO()), "x", _slow_writer(cancel, wrote), cancel=cancel
        )
    assert wrote == []  # returned without waiting for the worker


def test_map_writes_neither_file_when_esc_comes_while_the_page_is_built(
    esc: threading.Event, monkeypatch: pytest.MonkeyPatch
) -> None:
    def page(_data: object) -> str:
        esc.set()
        return "<html></html>"

    monkeypatch.setattr(slash, "map_page", page)
    text, _ = _run("/map JavaScript --suites test --depth 1", _registry())
    assert text == STOPPED and _written() == []


@pytest.mark.parametrize("name", ["roadmap", "successors"])
def test_roadmap_and_successors_write_neither_file_when_esc_comes_before_the_write(
    esc: threading.Event, monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    def page(_data: object) -> str:
        esc.set()
        return "<html></html>"

    monkeypatch.setattr(slash, f"{name}_page", page)
    fake = Result(status="ok", output={"suites": {}, "policy": "p", "steps": []})
    monkeypatch.setattr(slash, "run", lambda *_a, **_k: fake)  # the engine's answer, not a walk
    text, _ = _run(f"/{name} a | b" if name == "roadmap" else f"/{name} a", None)
    assert text == STOPPED and _written() == []


def test_discover_outputs_are_all_or_nothing_after_esc(
    esc: threading.Event, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = discover(_registry(), _items("web developer", "JavaScript"), depth=1, breadth=4)

    def late_page(_data: object) -> str:
        esc.set()  # Esc while the last output is being built
        return "<html></html>"

    monkeypatch.setattr(discover_report, "render_html", late_page)
    with pytest.raises(TurnCancelled):
        discover_report.write_outputs(data, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_no_ledger_line_is_appended_after_esc(esc: threading.Event, tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / ".ta-tag")
    esc.set()
    for write in (
        lambda: ledger.rewrite_append("k", Rewrite()),
        lambda: ledger.review_append({"review_key": "k"}),
        lambda: runner._append_gaps(ledger, [{"row_id": "1"}]),
    ):
        with pytest.raises(TurnCancelled):
            write()
    assert not list((tmp_path / ".ta-tag").glob("*"))


def test_tag_in_the_chat_never_swaps_the_process_stdout(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    project = _tag_project(tmp_path)
    real = sys.stdout
    seen: list[bool] = []

    class Watching(AnyKindSuite):
        def search_nodes(self, text: str, kind: str | None = None):  # type: ignore[override]
            seen.append(sys.stdout is real)
            return super().search_nodes(text, kind)

    reg = table_registry({"esco": Watching("esco")}, default="esco", broken={})
    text, _ = _run(f"/tag {project} --no-llm --fresh --suites esco", reg)
    assert "units=" in text and seen and all(seen)
    bad, _ = _run(f"/tag {project} --bogus", reg)
    assert bad.startswith("/tag:") and "--bogus" in bad
    assert capsys.readouterr().err == ""  # argparse printed nothing on the terminal
