"""Append-only JSONL run log — the source of truth for cost/accuracy numbers.

Local file, not a cloud dependency (MVP plan Sec 2.6): reproducible without
LangSmith/Langfuse, which stay optional exporters layered on top later.
"""

from __future__ import annotations

import os
from pathlib import Path

from talent_angels.memory.paths import default_runlog_path
from talent_angels.runlog.models import RunLogRecord


def runlog_path() -> Path:
    """``RUNLOG_PATH`` or ``<memory home>/runlog.jsonl``.

    The run-log holds users' raw questions, so it lives in the memory home that
    ``/reset-all`` erases rather than wherever the process was launched.
    """
    raw = os.environ.get("RUNLOG_PATH", "").strip()
    return Path(raw).expanduser() if raw else default_runlog_path()


def append_record(record: RunLogRecord, *, path: Path | None = None) -> None:
    target = path or runlog_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as f:
        f.write(record.model_dump_json())
        f.write("\n")


def read_records(*, path: Path | None = None, last: int | None = None) -> list[RunLogRecord]:
    """Load JSONL turns. ``last`` keeps only the newest N records."""
    target = path or runlog_path()
    if not target.is_file():
        return []
    rows: list[RunLogRecord] = []
    for line in target.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(RunLogRecord.model_validate_json(line))
    if last is None or last <= 0:
        return rows
    return rows[-last:]
