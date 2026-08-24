"""Storing an OpenRouter key: verify first, write once, never echo it back."""

from __future__ import annotations

import stat
from pathlib import Path

import pytest

from talent_angels.session.credentials import (
    ENV_NAME,
    CredentialError,
    have_key,
    looks_like_openrouter_key,
    store,
)

KEY = "sk-or-v1-" + "a" * 40


def test_obvious_mispastes_are_caught_without_a_network_call() -> None:
    assert looks_like_openrouter_key(KEY) is True
    assert looks_like_openrouter_key("sk-or-short") is False
    assert looks_like_openrouter_key("sk-ant-api03-something-long-enough-here") is False
    assert looks_like_openrouter_key("") is False


def test_store_replaces_a_previous_value_rather_than_appending(tmp_path: Path) -> None:
    """A second /login must not leave two keys in the file, where the loser
    silently wins or loses depending on the reader."""
    env = tmp_path / ".env"
    env.write_text(f"NEO4J_URI=bolt://localhost:7687\n{ENV_NAME}=sk-or-v1-old\n", encoding="utf-8")

    store(KEY, root=tmp_path)

    lines = env.read_text(encoding="utf-8").splitlines()
    assert [line for line in lines if line.startswith(f"{ENV_NAME}=")] == [f"{ENV_NAME}={KEY}"]
    assert "NEO4J_URI=bolt://localhost:7687" in lines


def test_store_creates_the_file_and_locks_it_to_the_owner(tmp_path: Path) -> None:
    path = store(KEY, root=tmp_path)

    assert path.exists()
    mode = stat.S_IMODE(path.stat().st_mode)
    assert mode == 0o600, f"credential file is group/world readable: {mode:o}"


def test_store_exports_to_the_running_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without this the key only takes effect on the next launch, which reads
    as the login having silently failed."""
    monkeypatch.delenv(ENV_NAME, raising=False)
    assert have_key() is False

    store(KEY, root=tmp_path)

    assert have_key() is True


def test_unwritable_destination_reports_instead_of_raising_oserror(tmp_path: Path) -> None:
    unwritable = tmp_path / "locked"
    unwritable.mkdir()
    unwritable.chmod(stat.S_IRUSR | stat.S_IXUSR)
    try:
        with pytest.raises(CredentialError, match="could not write"):
            store(KEY, root=unwritable)
    finally:
        unwritable.chmod(stat.S_IRWXU)
