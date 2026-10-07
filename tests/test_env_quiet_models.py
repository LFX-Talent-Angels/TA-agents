"""Loading the environment quiets local model loading, without overriding the user."""

from __future__ import annotations

import os

import pytest

from talent_angels.env import load_local_dotenv


def test_quiet_defaults_are_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HF_HUB_VERBOSITY", raising=False)
    monkeypatch.delenv("HF_HUB_DISABLE_PROGRESS_BARS", raising=False)
    load_local_dotenv()
    assert os.environ["HF_HUB_VERBOSITY"] == "error"
    assert os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] == "1"


def test_an_export_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TRANSFORMERS_VERBOSITY", "info")
    load_local_dotenv()
    assert os.environ["TRANSFORMERS_VERBOSITY"] == "info"
