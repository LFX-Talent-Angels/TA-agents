"""Local `.env` loading: file fills gaps, shell exports win."""

from __future__ import annotations

import os

import pytest

from talent_angels.env import load_local_dotenv


def test_load_local_dotenv_sets_unset_keys(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("LLM_PROVIDER=litellm\nLLM_MODEL=from-file\n", encoding="utf-8")
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.chdir(tmp_path)

    loaded = load_local_dotenv()

    assert loaded == env_file
    assert os.environ["LLM_PROVIDER"] == "litellm"
    assert os.environ["LLM_MODEL"] == "from-file"


def test_load_local_dotenv_does_not_override_shell(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / ".env").write_text("LLM_PROVIDER=litellm\n", encoding="utf-8")
    monkeypatch.setenv("LLM_PROVIDER", "none")
    monkeypatch.chdir(tmp_path)

    load_local_dotenv()

    assert os.environ["LLM_PROVIDER"] == "none"
