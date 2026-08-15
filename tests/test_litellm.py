"""LiteLLM adapter tests without making provider requests."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from talent_angels.llm import Message
from talent_angels.llm.litellm_client import (
    LOCAL_MODEL_COST_MAP_ENV,
    LiteLLMClient,
    ensure_local_model_cost_map,
)
from talent_angels.runlog import estimate_llm_cost_usd

MODEL = "openrouter/nvidia/nemotron-3-ultra-550b-a55b:free"


def test_litellm_maps_request_text_and_exclusive_usage_buckets() -> None:
    captured: dict[str, Any] = {}

    def complete(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {
            "model": "nvidia/nemotron-3-ultra-550b-a55b:free",
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "Found it."},
                }
            ],
            "usage": {
                "prompt_tokens": 21,
                "completion_tokens": 13,
                "prompt_tokens_details": {"cached_tokens": 5},
                "cache_write_tokens": 2,
                "completion_tokens_details": {"reasoning_tokens": 8},
            },
        }

    client = LiteLLMClient(
        model=MODEL,
        reasoning_enabled=True,
        completion_fn=complete,
    )

    result = client.complete(
        [
            Message(role="system", content="Return only grounded facts."),
            Message(role="user", content="Find software developer."),
        ]
    )

    assert captured == {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "Return only grounded facts."},
            {"role": "user", "content": "Find software developer."},
        ],
        "max_tokens": 1024,
        "extra_body": {"reasoning": {"enabled": True}},
    }
    assert result.text == "Found it."
    assert result.provider == "litellm"
    assert result.model == "nvidia/nemotron-3-ultra-550b-a55b:free"
    assert result.usage.input_tokens == 14
    assert result.usage.output_tokens == 13
    assert result.usage.cache_read_input_tokens == 5
    assert result.usage.cache_creation_input_tokens == 2
    assert result.usage.reasoning_tokens == 8

    cost = estimate_llm_cost_usd(
        result.usage,
        "priced-test-model",
        rate_card={
            "version": "test",
            "models": {
                "priced-test-model": {
                    "input_per_million": 1_000_000.0,
                    "output_per_million": 2_000_000.0,
                    "cache_read_per_million": 100_000.0,
                    "cache_write_5m_per_million": 1_250_000.0,
                }
            },
        },
    )
    assert cost.total == 43.0
    assert cost.llm == 47.0


def test_litellm_accepts_model_response_objects() -> None:
    class Response:
        def model_dump(self) -> dict[str, object]:
            return {
                "model": "actual-model",
                "choices": [{"finish_reason": "stop", "message": {"content": "ok"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 2},
            }

    def completion_fn(**_kwargs: object) -> object:
        return Response()

    result = LiteLLMClient(model=MODEL, completion_fn=completion_fn).complete(
        [Message(role="user", content="hello")]
    )

    assert result.text == "ok"
    assert result.model == "actual-model"
    assert result.usage.input_tokens == 1
    assert result.usage.output_tokens == 2


def test_litellm_maps_tool_calls() -> None:
    def complete(**kwargs: object) -> dict[str, object]:
        assert kwargs.get("tools")
        assert "tool_choice" not in kwargs
        assert "extra_body" not in kwargs
        return {
            "model": "actual-model",
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call-1",
                                "function": {
                                    "name": "search_nodes",
                                    "arguments": '{"text":"software developer"}',
                                },
                            }
                        ],
                    },
                }
            ],
            "usage": {"prompt_tokens": 4, "completion_tokens": 3},
        }

    result = LiteLLMClient(model=MODEL, completion_fn=complete).complete(
        [Message(role="user", content="skills?")],
        tools=[{"type": "function", "function": {"name": "search_nodes"}}],
    )

    assert result.text == ""
    assert result.tool_calls[0].name == "search_nodes"
    assert result.tool_calls[0].arguments == {"text": "software developer"}


def test_litellm_omits_reasoning_when_disabled() -> None:
    def complete(**kwargs: object) -> dict[str, object]:
        assert "reasoning" not in kwargs
        assert "extra_body" not in kwargs
        return {
            "model": "actual-model",
            "choices": [{"finish_reason": "stop", "message": {"content": "ok"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }

    result = LiteLLMClient(model=MODEL, completion_fn=complete).complete(
        [Message(role="user", content="hello")]
    )

    assert result.text == "ok"


def test_litellm_rejects_provider_error_choice() -> None:
    def complete(**_kwargs: object) -> dict[str, object]:
        return {
            "model": "actual-model",
            "choices": [
                {
                    "finish_reason": "error",
                    "message": {"content": "partial"},
                    "error": {"message": "upstream failed"},
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }

    with pytest.raises(RuntimeError, match="generation error"):
        LiteLLMClient(model=MODEL, completion_fn=complete).complete(
            [Message(role="user", content="hello")]
        )


@pytest.mark.parametrize(
    "response",
    [
        {"model": "actual-model", "choices": []},
        {
            "model": "actual-model",
            "choices": [{"finish_reason": "stop", "message": {"content": "ok"}}],
        },
        {
            "model": "actual-model",
            "choices": [{"finish_reason": "stop", "message": {"content": "ok"}}],
            "usage": {"prompt_tokens": "one", "completion_tokens": 2},
        },
    ],
)
def test_litellm_rejects_malformed_measurement_response(
    response: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="LiteLLM response"):
        LiteLLMClient(model=MODEL, completion_fn=lambda **_kwargs: response).complete(
            [Message(role="user", content="hello")]
        )


def test_litellm_rejects_provider_error_envelope() -> None:
    def complete(**_kwargs: object) -> dict[str, object]:
        return {
            "error": {"message": "upstream failed"},
            "model": "actual-model",
            "choices": [{"finish_reason": "stop", "message": {"content": "partial"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }

    with pytest.raises(RuntimeError, match="generation error"):
        LiteLLMClient(model=MODEL, completion_fn=complete).complete(
            [Message(role="user", content="hello")]
        )


def test_litellm_rejects_cache_tokens_exceeding_prompt() -> None:
    def complete(**_kwargs: object) -> dict[str, object]:
        return {
            "model": "actual-model",
            "choices": [{"finish_reason": "stop", "message": {"content": "ok"}}],
            "usage": {
                "prompt_tokens": 3,
                "completion_tokens": 1,
                "prompt_tokens_details": {"cached_tokens": 5},
            },
        }

    with pytest.raises(ValueError, match="cache token"):
        LiteLLMClient(model=MODEL, completion_fn=complete).complete(
            [Message(role="user", content="hello")]
        )


def test_ensure_local_model_cost_map_sets_litellm_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(LOCAL_MODEL_COST_MAP_ENV, raising=False)

    ensure_local_model_cost_map()

    assert os.environ[LOCAL_MODEL_COST_MAP_ENV].lower() == "true"


def test_litellm_complete_does_not_print_provider_list(
    capsys: pytest.CaptureFixture[str],
) -> None:
    def complete(**_kwargs: object) -> dict[str, object]:
        print("Provider List: OpenAI, Anthropic")
        return {
            "model": "actual-model",
            "choices": [{"finish_reason": "stop", "message": {"content": "ok"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }

    result = LiteLLMClient(model=MODEL, completion_fn=complete).complete(
        [Message(role="user", content="hello")]
    )

    captured = capsys.readouterr()
    assert result.text == "ok"
    assert "Provider List" not in captured.out
    assert "Provider List" not in captured.err


def test_importing_litellm_uses_bundled_cost_map_without_network() -> None:
    src = Path(__file__).resolve().parents[1] / "src"
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(src), env.get("PYTHONPATH", "")])
    env.pop(LOCAL_MODEL_COST_MAP_ENV, None)
    env.pop("OPENROUTER_API_KEY", None)
    env.pop("LLM_API_KEY", None)
    script = (
        "import httpx\n"
        "def _blocked(*_args, **_kwargs):\n"
        "    raise AssertionError('LiteLLM attempted a remote cost-map fetch')\n"
        "httpx.get = _blocked\n"
        "from talent_angels.llm.litellm_client import LiteLLMClient\n"
        "LiteLLMClient(model='openrouter/test-model')\n"
        "from litellm.litellm_core_utils.get_model_cost_map import _cost_map_source_info\n"
        "assert _cost_map_source_info.source == 'local'\n"
        "assert _cost_map_source_info.is_env_forced is True\n"
        "print('ok')\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "ok" in completed.stdout


def test_openrouter_transform_forwards_reasoning_extra_body() -> None:
    ensure_local_model_cost_map()
    from litellm.llms.openrouter.chat.transformation import OpenrouterConfig

    body = OpenrouterConfig().transform_request(
        model="nvidia/nemotron-3-ultra-550b-a55b:free",
        messages=[{"role": "user", "content": "ping"}],
        optional_params={
            "max_tokens": 16,
            "extra_body": {"reasoning": {"enabled": True}},
        },
        litellm_params={},
        headers={},
    )

    assert body["reasoning"] == {"enabled": True}
    assert body["usage"] == {"include": True}
    assert body["max_tokens"] == 16
