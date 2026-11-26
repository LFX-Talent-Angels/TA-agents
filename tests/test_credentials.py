"""Storing provider credentials: verify first, write once, never echo keys back."""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

import talent_angels.session.credentials as credentials
from talent_angels.session.credentials import (
    ENV_NAME,
    CredentialError,
    looks_like_openrouter_key,
    save_settings,
)

KEY = "sk-or-v1-" + "a" * 40


def test_obvious_mispastes_are_caught_without_a_network_call() -> None:
    assert looks_like_openrouter_key(KEY) is True
    assert looks_like_openrouter_key("sk-or-short") is False
    assert looks_like_openrouter_key("sk-ant-api03-something-long-enough-here") is False
    assert looks_like_openrouter_key("") is False


def test_save_settings_replaces_entire_file_rather_than_appending(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A second /connect must not leave stale keys from an earlier provider."""
    env = tmp_path / "llm.env"
    env.write_text(f'OPENAI_API_KEY="sk-old"\n{ENV_NAME}="sk-or-v1-old"\n', encoding="utf-8")

    monkeypatch.delenv(ENV_NAME, raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-old")
    save_settings({ENV_NAME: KEY, "LLM_PROVIDER": "litellm"}, path=env)

    lines = env.read_text(encoding="utf-8").splitlines()
    assert lines == [f'{ENV_NAME}="{KEY}"', 'LLM_PROVIDER="litellm"']
    assert "OPENAI_API_KEY" not in os.environ


def test_save_settings_creates_the_file_and_locks_it_to_the_owner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(ENV_NAME, raising=False)
    path = save_settings({ENV_NAME: KEY}, path=tmp_path / "llm.env")

    assert path.exists()
    mode = stat.S_IMODE(path.stat().st_mode)
    assert mode == 0o600, f"credential file is group/world readable: {mode:o}"


def test_save_settings_keeps_the_previous_file_when_secure_replace_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = tmp_path / "llm.env"
    previous = f'{ENV_NAME}="sk-or-v1-old"\n'
    env.write_text(previous, encoding="utf-8")
    env.chmod(0o600)

    def fail_replace(source: str, destination: Path) -> None:
        raise OSError("replace failed")

    monkeypatch.setattr(credentials.os, "replace", fail_replace)
    monkeypatch.delenv(ENV_NAME, raising=False)

    with pytest.raises(CredentialError, match="could not write"):
        save_settings({ENV_NAME: KEY}, path=env)

    assert env.read_text(encoding="utf-8") == previous
    assert ENV_NAME not in os.environ


def test_save_settings_exports_to_the_running_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(ENV_NAME, raising=False)
    save_settings({ENV_NAME: KEY}, path=tmp_path / "llm.env")
    assert os.environ.get(ENV_NAME) == KEY


def test_unwritable_destination_reports_instead_of_raising_oserror(tmp_path: Path) -> None:
    unwritable = tmp_path / "locked"
    unwritable.mkdir()
    unwritable.chmod(stat.S_IRUSR | stat.S_IXUSR)
    try:
        with pytest.raises(CredentialError, match="could not write"):
            save_settings({ENV_NAME: KEY}, path=unwritable / "llm.env")
    finally:
        unwritable.chmod(stat.S_IRWXU)
