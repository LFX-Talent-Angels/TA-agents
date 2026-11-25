"""`tag` as a job: the root restriction, no model without the flag, cancel, busy, interrupted.

The job thread is joined (`_join`), never polled with sleep. `TA_AGENTS_HOME` and `TA_TAG_ROOT`
point at tmp dirs so nothing touches a real home.
"""

from __future__ import annotations

import dataclasses
import json
import os
import shutil
import subprocess
import sys
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from talent_angels.engine import CAPABILITIES, RunContext, jobs
from talent_angels.engine.jobs import (
    JobBusy,
    JobRejected,
    JobState,
    cancel_job,
    job_status,
    start_job,
)
from tests.fakes.suite import fake_registry
from tests.test_tagging_run import FIXTURE
from tests.test_tagging_run import _registry as tag_registry


@pytest.fixture(autouse=True)
def _env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    root = tmp_path / "root"
    root.mkdir()
    monkeypatch.setenv("TA_AGENTS_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("TA_TAG_ROOT", str(root))
    monkeypatch.delenv("TA_JOBS_ALLOW_MODEL", raising=False)
    yield root
    for job_id, event in list(jobs._events.items()):  # never leave a looping job behind
        event.set()
        jobs._join(job_id)


def _project(root: Path, name: str = "proj") -> str:
    inputs = root / name / "inputs"
    inputs.mkdir(parents=True)
    shutil.copy(FIXTURE, inputs / "standard.json")
    return name


def _looping(monkeypatch: pytest.MonkeyPatch) -> threading.Event:
    """Make `tag` a function that ticks forever, until a progress hook raises."""
    started = threading.Event()

    def spin(_inp: Any, _reg: Any, ctx: RunContext) -> dict[str, Any]:
        started.set()
        while True:
            ctx.progress("tick")

    monkeypatch.setitem(CAPABILITIES, "tag", dataclasses.replace(CAPABILITIES["tag"], fn=spin))
    return started


def test_a_tag_job_runs_without_a_model_and_writes_its_state(_env: Path) -> None:
    name = _project(_env)
    state = start_job("tag", {"path": name, "limit": 3}, registry=tag_registry())
    assert state.status == "running" and state.call
    assert state.call.input["path"] == name and state.call.input["no_llm"] is True
    jobs._join(state.id)

    done = job_status(state.id)
    assert done.status == "done" and done.finished_at and done.result
    assert done.result.status == "ok" and done.result.output
    assert done.result.output["units"] > 0
    assert list((_env / name).rglob("*.xlsx"))
    saved = json.loads((jobs.jobs_dir() / f"{state.id}.json").read_text())
    assert saved["status"] == "done" and saved["pid"] == os.getpid()
    assert not list(jobs.jobs_dir().glob("*.tmp"))  # the atomic write left no temp file


def test_the_default_run_never_reaches_a_model(_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(**_k: object) -> None:
        raise AssertionError("a model client was requested")

    monkeypatch.setattr("talent_angels.llm.factory.get_llm_client", forbidden)
    state = start_job("tag", {"path": _project(_env)}, registry=tag_registry())
    jobs._join(state.id)
    assert job_status(state.id).status == "done"


@pytest.mark.parametrize("bad", ["../x", "../root/proj", "proj/../../x", "/etc", "/", "nope"])
def test_a_path_outside_the_root_or_missing_is_refused(_env: Path, bad: str) -> None:
    _project(_env)
    with pytest.raises(JobRejected) as err:
        start_job("tag", {"path": bad}, registry=tag_registry())
    assert err.value.result.status == "invalid_input"
    assert not jobs.jobs_dir().exists() or not list(jobs.jobs_dir().glob("*.json"))


def test_a_symlink_that_leaves_the_root_is_refused(_env: Path, tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    (outside / "inputs").mkdir(parents=True)
    shutil.copy(FIXTURE, outside / "inputs" / "standard.json")
    (_env / "link").symlink_to(outside)
    with pytest.raises(JobRejected):
        start_job("tag", {"path": "link"}, registry=tag_registry())

    name = _project(_env)  # a link inside the project may not leave the root either
    (_env / name / "inputs" / "stolen.json").symlink_to(outside / "inputs" / "standard.json")
    with pytest.raises(JobRejected):
        start_job("tag", {"path": name}, registry=tag_registry())


def test_a_symlink_inside_the_root_is_fine(_env: Path) -> None:
    name = _project(_env)
    (_env / "alias").symlink_to(_env / name)
    state = start_job("tag", {"path": "alias"}, registry=tag_registry())
    jobs._join(state.id)
    assert job_status(state.id).status == "done"


def test_a_profile_path_must_be_inside_the_root_too(_env: Path) -> None:
    with pytest.raises(JobRejected):
        start_job(
            "tag", {"path": _project(_env), "profile": "/etc/passwd"}, registry=fake_registry()
        )


def test_a_model_run_needs_the_server_flag(_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    name = _project(_env)
    with pytest.raises(JobRejected) as err:
        start_job("tag", {"path": name, "no_llm": False}, registry=tag_registry())
    assert "TA_JOBS_ALLOW_MODEL" in (err.value.result.error or "")
    assert not jobs.jobs_dir().exists() or not list(jobs.jobs_dir().glob("*.json"))

    monkeypatch.setenv("TA_JOBS_ALLOW_MODEL", "1")
    seen: list[bool] = []

    def fake_tag(inp: Any, _reg: Any, _ctx: RunContext) -> dict[str, Any]:
        seen.append(inp.no_llm)
        return {}

    monkeypatch.setitem(CAPABILITIES, "tag", dataclasses.replace(CAPABILITIES["tag"], fn=fake_tag))
    state = start_job("tag", {"path": name, "no_llm": False}, registry=tag_registry())
    jobs._join(state.id)
    assert seen == [False]


def test_bad_input_an_unknown_capability_and_a_bad_id_start_nothing(_env: Path) -> None:
    for name, payload, status in (
        ("tag", {"path": "x", "extra": 1}, "invalid_input"),
        ("tag", {}, "invalid_input"),
        ("shell", {}, "unknown_capability"),
    ):
        with pytest.raises(JobRejected) as err:
            start_job(name, payload, registry=tag_registry())
        assert err.value.result.status == status
    for bad in ("../../etc/passwd", "x", "", "A" * 32, "0" * 31, "0" * 33):
        assert job_status(bad).status == "not_found"
        assert cancel_job(bad).status == "not_found"
    assert job_status("0" * 32).status == "not_found"  # well formed, unknown


def test_cancel_stops_the_job_between_ticks(monkeypatch: pytest.MonkeyPatch) -> None:
    started = _looping(monkeypatch)
    state = start_job("tag", {"path": "."}, registry=fake_registry())
    assert started.wait(5)
    asked = cancel_job(state.id)
    assert asked.cancel_requested
    jobs._join(state.id)

    end = job_status(state.id)
    assert end.status == "cancelled" and end.result and end.result.status == "cancelled"
    assert end.cancel_requested and end.finished_at
    assert cancel_job(state.id).status == "cancelled"  # harmless on a finished job
    assert not (jobs.jobs_dir() / f"{state.id}.cancel").exists()  # the marker is cleaned up


def test_cancel_through_the_marker_file_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    """What another process (a second worker, the MCP server) does: touch `<id>.cancel`."""
    started = _looping(monkeypatch)
    state = start_job("tag", {"path": "."}, registry=fake_registry())
    assert started.wait(5)
    (jobs.jobs_dir() / f"{state.id}.cancel").touch()
    jobs._join(state.id)
    assert job_status(state.id).status == "cancelled"


def test_a_second_job_while_one_runs_is_busy(monkeypatch: pytest.MonkeyPatch) -> None:
    started = _looping(monkeypatch)
    first = start_job("tag", {"path": "."}, registry=fake_registry())
    assert started.wait(5)
    with pytest.raises(JobBusy):
        start_job("tag", {"path": "."}, registry=fake_registry())
    cancel_job(first.id)
    jobs._join(first.id)
    again = start_job("tag", {"path": "."}, registry=fake_registry())  # free again
    cancel_job(again.id)
    jobs._join(again.id)


def test_progress_and_a_failing_capability_are_recorded(monkeypatch: pytest.MonkeyPatch) -> None:
    def fn(_inp: Any, _reg: Any, ctx: RunContext) -> dict[str, Any]:
        ctx.progress("one")
        assert ctx.step is not None
        ctx.step("units", 1, 2)
        raise RuntimeError("sk-or-v1-SECRET")

    monkeypatch.setitem(CAPABILITIES, "tag", dataclasses.replace(CAPABILITIES["tag"], fn=fn))
    state = start_job("tag", {"path": "."}, registry=fake_registry())
    jobs._join(state.id)
    end = job_status(state.id)
    assert end.status == "failed" and end.result and end.result.error == "RuntimeError"
    assert "SECRET" not in end.model_dump_json()
    assert end.progress == ["one"] and end.step == {"label": "units", "done": 1, "total": 2}


def test_a_running_file_with_a_dead_process_is_interrupted() -> None:
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    child.wait()  # reaped: its pid is dead
    ghost = JobState(id="a" * 32, capability="tag", status="running", pid=child.pid, started_at="x")
    jobs._write(ghost)
    assert job_status(ghost.id).status == "interrupted"
    assert json.loads((jobs.jobs_dir() / f"{ghost.id}.json").read_text())["status"] == "interrupted"
    # this process, but no thread behind it (a restart that kept the pid): also interrupted
    mine = JobState(
        id="b" * 32, capability="tag", status="running", pid=os.getpid(), started_at="x"
    )
    jobs._write(mine)
    assert job_status(mine.id).status == "interrupted"
