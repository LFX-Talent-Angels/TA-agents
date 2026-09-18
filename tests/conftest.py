"""Repo-wide test isolation: no test run should touch real local state."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _isolate_memory_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("TA_MEMORY_DIR", str(tmp_path / "memory"))
