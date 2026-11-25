"""Esc stops a long slash command at the next item: nothing written, session unchanged."""

from __future__ import annotations

import shutil
import threading
from collections.abc import Iterator
from pathlib import Path

import pytest

from talent_angels.llm import protocol
from talent_angels.llm.protocol import TurnCancelled, bind_cancel, check_cancelled
from talent_angels.memory.paths import outputs_dir
from talent_angels.session import slash
from talent_angels.session.copy import STOPPED
from talent_angels.session.kernel import handle_line
from talent_angels.session.store import new_session
from talent_angels.skills.connect.discover import discover
from talent_angels.suites import SuiteRegistry
from tests.fakes.suite import suite_factory
from tests.fakes.tagging_suite import table_registry
from tests.test_discover import PathSuite, _graph, _items
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
