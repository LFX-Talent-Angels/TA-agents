"""Tests for Bring Your Own LLM (`/connect`, `ta-agent setup`, `llm.env`, `/model`)."""

from __future__ import annotations

import io
import os
import stat
import sys
from pathlib import Path
from typing import Any

import litellm
import pytest

from talent_angels.env import load_local_dotenv
from talent_angels.session import credentials, switch
from talent_angels.session.copy import NO_MODEL_NOTICE
from talent_angels.session.credentials import (
    CredentialError,
    save_settings,
    settings_for,
    verify_model,
)
from talent_angels.tui import connect


class _FakeConsole:
    def __init__(self, answers: list[str]) -> None:
        self.answers = list(answers)
        self.printed: list[str] = []
        self.password_prompts: list[bool] = []

    def print(self, msg: object = "", **_kw: object) -> None:
        self.printed.append(str(msg))

    def input(self, prompt: str = "", *, password: bool = False) -> str:
        self.password_prompts.append(password)
        if not self.answers:
            raise EOFError
        return self.answers.pop(0)


def test_settings_for_per_provider() -> None:
    assert settings_for("none") == {"LLM_PROVIDER": "none"}

    s_openai = settings_for("openai", "gpt-4o-mini", "sk-openai-1234567890")
    assert s_openai == {
        "LLM_PROVIDER": "litellm",
        "LLM_MODEL": "openai/gpt-4o-mini",
        "OPENAI_API_KEY": "sk-openai-1234567890",
    }

    s_ollama = settings_for("ollama", "", None, {"OLLAMA_API_BASE": "http://localhost:11434"})
    assert s_ollama == {
        "LLM_PROVIDER": "litellm",
        "LLM_MODEL": "ollama/llama3.1",
        "OLLAMA_API_BASE": "http://localhost:11434",
    }

    s_azure = settings_for(
        "azure",
        "my-dep",
        "az-key-123",
        {"AZURE_API_BASE": "https://ex.openai.azure.com", "AZURE_API_VERSION": "2024-10-21"},
    )
    assert s_azure["LLM_MODEL"] == "azure/my-dep"
    assert s_azure["AZURE_API_KEY"] == "az-key-123"
    assert s_azure["AZURE_API_BASE"] == "https://ex.openai.azure.com"


def test_verify_model_routes_openrouter_and_litellm_safely(monkeypatch: pytest.MonkeyPatch) -> None:
    seen_verify: list[str] = []
    monkeypatch.setattr(credentials, "verify", lambda key: (seen_verify.append(key), "ok")[1])
    assert verify_model("openrouter", {"OPENROUTER_API_KEY": "sk-or-v1-test"}) == "ok"
    assert seen_verify == ["sk-or-v1-test"]

    captured: list[dict[str, Any]] = []

    def ok_completion(**kwargs: object) -> dict[str, Any]:
        captured.append(dict(kwargs))
        return {"model": "openai/gpt-4o-mini", "choices": [{"message": {"content": "OK"}}]}

    monkeypatch.setattr(litellm, "completion", ok_completion)
    s = settings_for("openai", "gpt-4o-mini", "sk-secret-value")
    assert verify_model("openai", s) == "the model answered"
    assert captured[0]["api_key"] == "sk-secret-value"
    assert captured[0]["max_tokens"] == 8
    assert captured[0]["num_retries"] == 0

    secret = "sk-secret-leak-check-999"

    class AuthenticationError(Exception):
        pass

    def bad_completion(**_kwargs: object) -> None:
        raise AuthenticationError(f"bad key {secret}")

    monkeypatch.setattr(litellm, "completion", bad_completion)
    with pytest.raises(CredentialError) as exc_info:
        verify_model("openai", settings_for("openai", "gpt-4o-mini", secret))
    assert secret not in str(exc_info.value)
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None


def test_ask_and_save_enter_ollama_and_bad_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TA_AGENTS_HOME", str(tmp_path))

    # Enter at provider -> LLM_PROVIDER=none.
    c_enter = _FakeConsole([""])
    msg = connect.ask_and_save(c_enter, first_run=True)  # type: ignore[arg-type]
    assert msg == NO_MODEL_NOTICE
    assert (tmp_path / "llm.env").read_text(encoding="utf-8") == 'LLM_PROVIDER="none"\n'

    # Ollama asks no key.
    monkeypatch.setattr(connect, "verify_model", lambda *_a, **_k: "the model answered")
    c_ollama = _FakeConsole(["6", "", ""])
    msg2 = connect.ask_and_save(c_ollama)  # type: ignore[arg-type]
    assert msg2.startswith("Saved: ollama/llama3.1")
    assert not any(c_ollama.password_prompts)
    assert stat.S_IMODE((tmp_path / "llm.env").stat().st_mode) == 0o600

    # Bad key stores nothing (keeps previous file).
    before = (tmp_path / "llm.env").read_text(encoding="utf-8")

    def fail_verify(*_a: object, **_k: object) -> str:
        raise CredentialError("the provider rejected the key")

    monkeypatch.setattr(connect, "verify_model", fail_verify)
    c_bad = _FakeConsole(["1", "", "sk-wrong-key-value"])
    msg3 = connect.ask_and_save(c_bad)  # type: ignore[arg-type]
    assert c_bad.password_prompts == [False, False, True]
    assert msg3 == "Not stored: the provider rejected the key."
    assert (tmp_path / "llm.env").read_text(encoding="utf-8") == before


def test_setup_main_reads_stdin_rejects_api_key_flag_and_sets_exit_codes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TA_AGENTS_HOME", str(tmp_path))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(connect, "verify_model", lambda *_a, **_k: "the model answered")
    monkeypatch.setattr(sys, "stdin", io.StringIO("sk-from-stdin\n"))

    assert connect.setup_main(["--provider", "openai", "--model", "gpt-4o-mini"]) == 0
    assert "sk-from-stdin" in (tmp_path / "llm.env").read_text(encoding="utf-8")

    # Never accepts --api-key on argv.
    with pytest.raises(SystemExit) as sys_exc:
        connect.setup_main(["--provider", "openai", "--api-key", "sk-bad"])
    assert sys_exc.value.code == 2

    # Missing terminal with no --provider exits 2.
    monkeypatch.setattr(connect, "is_interactive", lambda: False)
    assert connect.setup_main([]) == 2


def test_load_local_dotenv_precedence_shell_beats_llm_env_beats_dotenv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TA_AGENTS_HOME", str(tmp_path / "ta-home"))
    (tmp_path / ".env").write_text(
        "LLM_PROVIDER=none\nLLM_MODEL=from-dotenv\nOPENAI_API_KEY=from-dotenv\n",
        encoding="utf-8",
    )
    save_settings(
        {"LLM_PROVIDER": "litellm", "LLM_MODEL": "from-llm-env", "OPENAI_API_KEY": "from-llm-env"},
        path=tmp_path / "ta-home" / "llm.env",
    )
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "from-shell")

    load_local_dotenv()
    assert os.environ["LLM_PROVIDER"] == "litellm"
    assert os.environ["LLM_MODEL"] == "from-llm-env"
    assert os.environ["OPENAI_API_KEY"] == "from-shell"


def test_switch_resolve_keeps_active_provider_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "litellm")
    monkeypatch.setenv("LLM_MODEL", "anthropic/claude-haiku-4-5")
    assert switch.resolve("claude-sonnet-4-6", []).model == "anthropic/claude-sonnet-4-6"
    assert switch.resolve("openai/gpt-4o-mini", []).model == "openai/gpt-4o-mini"

    monkeypatch.setenv("LLM_PROVIDER", "none")
    monkeypatch.setenv("LLM_MODEL", "stub")
    assert switch.resolve("google/gemma-4-31b-it:free", []).model == (
        "openrouter/google/gemma-4-31b-it:free"
    )
