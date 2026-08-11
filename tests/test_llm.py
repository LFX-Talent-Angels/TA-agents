"""LLM layer tests — stub client behavior and provider factory selection."""

import pytest

from talent_angels.llm import Message, get_llm_client
from talent_angels.llm.stub_client import StubLLMClient


def test_stub_client_echoes_last_user_message_with_zero_usage() -> None:
    client = StubLLMClient()
    messages = [
        Message(role="system", content="You are a helpful assistant."),
        Message(role="user", content="software developer"),
    ]

    result = client.complete(messages)

    assert result.text == "software developer"
    assert result.provider == "none"
    assert result.usage.input_tokens == 0
    assert result.usage.output_tokens == 0


def test_stub_client_handles_no_user_message() -> None:
    client = StubLLMClient()
    result = client.complete([Message(role="system", content="system only")])
    assert result.text == ""


def test_factory_defaults_to_stub_when_provider_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    client = get_llm_client()
    assert isinstance(client, StubLLMClient)
    assert client.provider == "none"


def test_factory_returns_stub_for_explicit_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "none")
    client = get_llm_client()
    assert isinstance(client, StubLLMClient)


def test_factory_rejects_unsupported_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "not-a-real-provider")
    with pytest.raises(ValueError, match="Unsupported LLM_PROVIDER"):
        get_llm_client()


def test_factory_requires_model_for_anthropic(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.delenv("LLM_MODEL", raising=False)
    with pytest.raises(ValueError, match="LLM_MODEL must be set"):
        get_llm_client()


def test_factory_requires_api_key_for_anthropic(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("LLM_MODEL", "claude-sonnet-5")
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(ValueError, match="requires LLM_API_KEY or ANTHROPIC_API_KEY"):
        get_llm_client()
