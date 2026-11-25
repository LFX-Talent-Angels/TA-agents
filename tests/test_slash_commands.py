"""/map /discover /roadmap /successors /tag in the chat: parse, summary, files, no turn clock."""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import pytest

from talent_angels.llm import protocol
from talent_angels.memory.paths import outputs_dir
from talent_angels.session import slash
from talent_angels.session.commands import parse_command
from talent_angels.session.copy import COMMANDS_BLOCK
from talent_angels.session.kernel import handle_line
from talent_angels.session.store import new_session
from tests.fakes.order_suite import registry
from tests.fakes.suite import fake_registry
from tests.test_roadmap import A, B, _esco
from tests.test_successors import X
from tests.test_successors import _esco as _skills
from tests.test_tagging_run import FIXTURE
from tests.test_tagging_run import _registry as tag_registry

SAMPLE = Path(__file__).parent / "fixtures" / "discover_uca_sample.csv"


def _no_runner(*_a: object, **_k: object) -> None:
    raise AssertionError("a slash command must not run a chat turn")


def _say(line: str, reg: object) -> str:
    state = new_session()
    reply = handle_line(state, line, runner=_no_runner, registry=reg)  # type: ignore[arg-type]
    assert [t.role for t in state.transcript] == ["user", "assistant"]
    return reply.text


def _files(command: str) -> list[Path]:
    return sorted(p for p in outputs_dir().glob(f"{command}-*/*") if p.is_file())


@pytest.mark.parametrize("name", ["map", "discover", "roadmap", "successors", "tag"])
def test_each_command_parses_and_is_in_help(name: str) -> None:
    parsed = parse_command(f"/{name} some thing --depth 2")
    assert parsed is not None and parsed.name == name and parsed.argument == "some thing --depth 2"
    assert f"`/{name}" in COMMANDS_BLOCK


def test_roadmap_summary_files_and_byte_identical_json() -> None:
    reg = registry(esco=_esco())
    first = _say(f"/roadmap {A.id} | {B.id} --per-step 2", reg)
    assert "learn-order-v1" in first and "algebra" in first

    [page, path] = _files("roadmap")
    assert path.name == "roadmap.json" and json.loads(path.read_text())["per_step"] == 2
    assert page.name == "roadmap.html" and "Foundations" in page.read_text()
    assert str(path) in first and str(page) in first
    before = path.read_bytes()
    assert _say(f"/roadmap {A.id} | {B.id} --per-step 2", reg) == first
    assert path.read_bytes() == before
    assert "timestamp" not in path.read_text()


def test_successors_and_map_write_json_and_name_the_top_items() -> None:
    text = _say(f"/successors {X.id} --depth 1 --breadth 2", registry(esco=_skills()))
    assert "esco: ok" in text
    [page, out] = _files("successors")
    assert page.name == "successors.html" and "<!doctype html>" in page.read_text()
    assert [n["id"] for n in json.loads(out.read_text())["suites"]["esco"]["nodes"]][:2] == [
        "esco:x",
        "esco:sb",
    ]

    reg = fake_registry()
    text = _say("/map software developer --suites fake --depth 1", reg)
    assert "Map for" in text and "fake mapped" in text and "Top:" in text
    route = _say("/map --route software developer | programmer --suites fake", reg)
    assert "software developer -> programmer" in route
    assert sorted(p.name for p in _files("map")) == ["map.html"] * 2 + ["map.json"] * 2
    assert all(
        "prefers-color-scheme:dark" in p.read_text() for p in _files("map") if p.suffix == ".html"
    )


def test_discover_writes_the_json_xlsx_and_html_reports() -> None:
    text = _say(f"/discover {SAMPLE} --suites fake --depth 1", fake_registry())
    assert "items:" in text and "guesses:" in text
    assert {p.suffix for p in _files("discover")} == {".json", ".xlsx", ".html"}
    assert all(str(p) in text for p in _files("discover"))


def test_tag_runs_the_batch_without_a_model(tmp_path: Path) -> None:
    project = tmp_path / "project"
    (project / "inputs").mkdir(parents=True)
    shutil.copy(FIXTURE, project / "inputs" / "standard.json")
    text = _say(f"/tag {project} --no-llm", tag_registry())
    assert "units=" in text and ".xlsx" in text
    assert list(project.rglob("*.xlsx"))


@pytest.mark.parametrize(
    ("line", "fragment"),
    [
        ("/map", "give a name or id"),
        ("/map a | b", "give a name or id"),
        ("/map --route onlyone", "FROM | TO"),
        ("/map x --depth 9", "--depth"),
        ("/map x --bogus", "unrecognized"),
        ("/map x --suites nope", "unknown suite"),
        ("/roadmap just-one", "FROM | TO"),
        ("/roadmap a | b --per-step 0", "--per-step"),
        ("/successors", "skill"),
        ("/successors x --depth zero", "--depth"),
        ("/discover", "file"),
        ("/discover /no/such/file.csv", "/no/such/file.csv"),
        (f"/discover {SAMPLE} --suites nope", "unknown suite"),
        ("/tag /no/such/dir", "no such path"),
        ("/tag", "path"),
    ],
)
def test_bad_input_is_one_line_never_a_traceback(line: str, fragment: str) -> None:
    text = _say(line, fake_registry())
    assert "\n" not in text and "Traceback" not in text
    assert text.startswith(f"/{line.split()[0][1:]}") and fragment in text


def test_a_command_runs_outside_the_turn_clock_and_is_never_a_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TA_TURN_DEADLINE_SECONDS", "0.05")
    seen: list[float | None] = []

    def slow(_arg: str | None, _reg: object) -> list[str]:
        time.sleep(0.2)  # four times the turn deadline
        seen.append(protocol.seconds_left())
        return ["done"]

    monkeypatch.setitem(slash._RUNNERS, "tag", slow)
    assert _say("/tag whatever", fake_registry()) == "done"
    assert seen == [None]  # no turn clock was running; _no_runner proves no turn either


def test_the_tui_loop_dispatches_a_command_with_its_registry(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from talent_angels.tui.app import main

    monkeypatch.setenv("LLM_PROVIDER", "none")
    monkeypatch.setenv("LLM_MODEL", "stub")
    answers = iter([f"/roadmap {A.id} | {B.id}", "/quit"])
    monkeypatch.setattr("rich.console.Console.input", lambda self, *a, **k: next(answers))

    assert main([], registry=registry(esco=_esco())) == 0
    out = capsys.readouterr().out
    assert "Learning order from" in out and "Running" in out
    assert [p.suffix for p in _files("roadmap")] == [".html", ".json"]


def test_an_unexpected_error_never_echoes_its_message_into_the_chat(monkeypatch, caplog) -> None:
    def boom(argument: object, registry: object) -> list[str]:
        raise RuntimeError("authorization failed: sk-secret-123")

    monkeypatch.setitem(slash._RUNNERS, "tag", boom)
    with caplog.at_level("WARNING"):
        reply = slash.run_slash("tag", "project/", fake_registry())
    assert reply == "/tag failed (RuntimeError); the details are in the log."
    assert "sk-secret" not in reply
    assert "/tag failed" in caplog.text


def test_chat_summaries_use_plain_words_for_status_codes() -> None:
    text = _say("/map zzzz-no-such-thing --suites fake --depth 1", fake_registry())
    assert "fake no clear match" in text and "not_found" not in text
