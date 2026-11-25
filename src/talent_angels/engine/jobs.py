"""Long capabilities as jobs: start, then status, then result; cancel at any time.

One in-process thread per job and one JSON file, `<TA home>/jobs/<id>.json`, so a UI or an MCP
client can poll, and another process (a second uvicorn worker, the MCP server) can read the state
or cancel through the `<id>.cancel` marker file. One running job per process: that also keeps two
tag runs off the same project ledger.

Guardrails for `tag` (the HTTP API has no auth): `path` must resolve inside the tag root
(`TA_TAG_ROOT`, else `<TA home>/tag`), symlinks included; a model run needs
`TA_JOBS_ALLOW_MODEL=1` on the server, so a caller cannot spend money on its own.
shortcut: no upload; add a multipart endpoint that writes into tag_root()/uploads/<job id>/ when a
remote UI needs it (python-multipart is not a dependency).
"""

from __future__ import annotations

import contextlib
import os
import re
import threading
import time
import uuid
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel

from talent_angels.engine.inputs import TagRequest
from talent_angels.engine.registry import (
    Call,
    JobCancelled,
    Result,
    RunContext,
    prepare,
    run,
)
from talent_angels.memory.paths import home
from talent_angels.suites import SuiteRegistry

__all__ = [
    "JobBusy",
    "JobRejected",
    "JobState",
    "cancel_job",
    "job_status",
    "jobs_dir",
    "start_job",
    "tag_root",
]

JOB_ID = re.compile(r"^[0-9a-f]{32}$")
JobStatus = Literal["running", "done", "failed", "cancelled", "interrupted", "not_found"]
_KEEP_LINES = 20
_WRITE_EVERY = 1.0  # seconds between progress writes; the final write always happens


class JobState(BaseModel):
    id: str
    capability: str
    status: JobStatus
    call: Call | None = None
    pid: int = 0
    started_at: str = ""  # UTC ISO, to the second (jobs are not cited artifacts)
    finished_at: str | None = None
    cancel_requested: bool = False
    progress: list[str] = []
    step: dict[str, Any] | None = None  # {"label", "done", "total"}
    result: Result | None = None


class JobBusy(RuntimeError):
    """A job of this process is still running."""


class JobRejected(Exception):
    """The job was not started: `result` says why (unknown_capability or invalid_input)."""

    def __init__(self, result: Result) -> None:
        super().__init__(result.error or result.status)
        self.result = result


_threads: dict[str, threading.Thread] = {}
_events: dict[str, threading.Event] = {}
_lock = threading.Lock()


def jobs_dir() -> Path:
    return home() / "jobs"


def tag_root() -> Path:
    return Path(os.environ.get("TA_TAG_ROOT") or home() / "tag").expanduser().resolve()


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _file(job_id: str) -> Path:
    return jobs_dir() / f"{job_id}.json"


def _marker(job_id: str) -> Path:
    return jobs_dir() / f"{job_id}.cancel"


def _write(state: JobState) -> None:
    path = _file(state.id)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(state.model_dump_json(indent=2), encoding="utf-8")
    os.replace(tmp, path)  # atomic: a reader never sees half a file


def _inside_root(relative: str, what: str) -> Path:
    """`relative` resolved under the tag root, or ValueError. `resolve()` follows symlinks, so
    a link out of the root is refused too."""
    root = tag_root()
    if Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise ValueError(f"{what} must be relative to the tag root, with no `..`")
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(root) or not resolved.exists():
        raise ValueError(f"{what} must be an existing path inside the tag root")
    for here, dirs, files in os.walk(resolved):  # a link inside the folder may not leave either
        for name in (*dirs, *files):
            link = Path(here, name)
            if link.is_symlink() and not link.resolve().is_relative_to(root):
                raise ValueError(f"{what} holds a link that leaves the tag root")
    return resolved


def _checked_tag(inp: TagRequest) -> TagRequest:
    from talent_angels.tagging.profile import builtin_profile_names

    path = _inside_root(inp.path, "path")
    profile = inp.profile
    if profile is not None and profile not in builtin_profile_names():
        profile = str(_inside_root(profile, "profile"))
    if not inp.no_llm and os.environ.get("TA_JOBS_ALLOW_MODEL") != "1":
        raise ValueError(
            "model runs are disabled on this server; set TA_JOBS_ALLOW_MODEL=1 or use no_llm"
        )
    return inp.model_copy(update={"path": str(path), "profile": profile})


def start_job(capability: str, payload: Mapping[str, Any], *, registry: SuiteRegistry) -> JobState:
    """Validate, then start the job thread. Raises JobRejected (404/422) or JobBusy (409)."""
    prepared = prepare(capability, payload)
    if isinstance(prepared, Result):
        raise JobRejected(prepared)
    cap, asked = prepared
    inp = asked
    if isinstance(asked, TagRequest):
        try:
            inp = _checked_tag(asked)  # absolute and inside the root; the call keeps the request
        except ValueError as exc:
            raise JobRejected(Result(status="invalid_input", error=str(exc))) from exc
    with _lock:
        if any(t.is_alive() for t in _threads.values()):
            raise JobBusy("a job is already running in this process")
        job_id = uuid.uuid4().hex
        event = threading.Event()
        state = JobState(
            id=job_id,
            capability=cap.name,
            status="running",
            call=Call(capability=cap.name, input=asked.model_dump(mode="json", by_alias=True)),
            pid=os.getpid(),
            started_at=_now(),
        )
        _write(state)
        thread = threading.Thread(
            target=_work,
            args=(state, inp, registry, event),
            daemon=True,
            name=f"ta-job-{job_id[:8]}",
        )
        _threads[job_id], _events[job_id] = thread, event
        thread.start()
    return state.model_copy(deep=True)  # the worker keeps mutating its own copy


def _work(state: JobState, inp: BaseModel, registry: SuiteRegistry, event: threading.Event) -> None:
    from talent_angels.llm.protocol import bind_cancel

    bind_cancel(event)  # model calls stop once the job is cancelled
    marker = _marker(state.id)
    last = 0.0

    def check() -> None:
        if event.is_set() or marker.exists():
            event.set()
            raise JobCancelled

    def save() -> None:
        nonlocal last
        if time.monotonic() - last >= _WRITE_EVERY:
            last = time.monotonic()
            _write(state)

    def progress(line: str) -> None:
        check()
        state.progress = [*state.progress, line][-_KEEP_LINES:]
        save()

    def step(label: str, done: int, total: int) -> None:
        check()
        state.step = {"label": label, "done": done, "total": total}
        save()

    ctx = RunContext(progress=progress, step=step, cancel=event)
    try:
        result = run(state.capability, inp, registry=registry, ctx=ctx, allow_long=True)
    except BaseException as exc:  # noqa: BLE001 - a job thread must always write its end
        result = Result(status="error", call=state.call, error=type(exc).__name__)
    state.result = result
    state.status = (
        "cancelled"
        if result.status == "cancelled"
        else "failed"
        if result.status in ("error", "invalid_input", "unknown_suite")
        else "done"
    )
    state.cancel_requested = state.cancel_requested or event.is_set()
    state.finished_at = _now()
    _write(state)
    with contextlib.suppress(OSError):
        marker.unlink(missing_ok=True)


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # another user's live process
    return True


def job_status(job_id: str) -> JobState:
    """The job's state. A bad or unknown id is `not_found` (which also blocks traversal)."""
    if not JOB_ID.match(job_id) or not _file(job_id).exists():
        return JobState(id=job_id[:64], capability="", status="not_found")
    try:
        state = JobState.model_validate_json(_file(job_id).read_text(encoding="utf-8"))
    except (OSError, ValueError):  # not one of ours: report it as unknown, never its content
        return JobState(id=job_id, capability="", status="not_found")
    if state.status == "running":
        state.cancel_requested = state.cancel_requested or _marker(job_id).exists()
        thread = _threads.get(job_id)
        local_dead = state.pid == os.getpid() and (thread is None or not thread.is_alive())
        if local_dead or (state.pid != os.getpid() and not _alive(state.pid)):
            state.status, state.finished_at = "interrupted", _now()
            _write(state)
    return state


def cancel_job(job_id: str) -> JobState:
    """Ask a running job to stop at its next progress tick. Harmless twice or when finished."""
    state = job_status(job_id)
    if state.status != "running":
        return state
    _marker(job_id).touch()  # what another process sees
    if job_id in _events:
        _events[job_id].set()
    return state.model_copy(update={"cancel_requested": True})


def _join(job_id: str, timeout: float = 30.0) -> None:
    """Tests only: wait for a local job thread instead of polling with sleep."""
    thread = _threads.get(job_id)
    if thread is not None:
        thread.join(timeout)
