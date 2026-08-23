"""Unified LiteLLM adapter behind TA-agents' internal LLMClient protocol."""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from typing import Any

from talent_angels.llm.protocol import LLMResult, LLMUsage, Message

DEFAULT_MAX_TOKENS = 1024
LOCAL_MODEL_COST_MAP_ENV = "LITELLM_LOCAL_MODEL_COST_MAP"

CompletionFn = Callable[..., object]


def ensure_local_model_cost_map() -> None:
    """Force LiteLLM to use its bundled cost map.

    TA-agents prices calls from ``runlog/rate_card.yaml``. LiteLLM otherwise
    fetches a remote model-cost JSON during ``import litellm``, which is unused
    here and fails offline. The env var is read at import time, so this must
    run before LiteLLM is imported.
    """

    os.environ[LOCAL_MODEL_COST_MAP_ENV] = "True"


ensure_local_model_cost_map()


def _mapping(value: object) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump()
        if isinstance(dumped, Mapping):
            return dumped
    raise ValueError("LiteLLM response must be a mapping or model response")


def _required_int(value: object, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"LiteLLM response requires non-negative integer {field}")
    return value


def _optional_int(value: object, field: str) -> int:
    if value is None:
        return 0
    return _required_int(value, field)


class LiteLLMClient:
    """Call any LiteLLM-supported provider and preserve measurable usage."""

    provider = "litellm"

    def __init__(
        self,
        model: str,
        *,
        reasoning_enabled: bool = False,
        completion_fn: CompletionFn | None = None,
    ) -> None:
        self.model = model
        self.reasoning_enabled = reasoning_enabled
        if completion_fn is None:
            ensure_local_model_cost_map()
            from litellm import completion

            completion_fn = completion
        self._completion = completion_fn

    def complete(self, messages: list[Message]) -> LLMResult:
        kwargs: dict[str, object] = {
            "model": self.model,
            "messages": [message.model_dump() for message in messages],
            "max_tokens": DEFAULT_MAX_TOKENS,
        }
        if self.reasoning_enabled:
            # OpenRouter-only fields are merged from extra_body in LiteLLM 1.96.2.
            kwargs["extra_body"] = {"reasoning": {"enabled": True}}

        body = _mapping(self._completion(**kwargs))
        if body.get("error") is not None:
            raise RuntimeError("LiteLLM response contained a provider generation error")

        response_model = body.get("model")
        if not isinstance(response_model, str) or not response_model:
            raise ValueError("LiteLLM response requires the actual model name")

        choices = body.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ValueError("LiteLLM response did not contain a text choice")
        first_choice = _mapping(choices[0])
        if first_choice.get("finish_reason") == "error" or first_choice.get("error") is not None:
            raise RuntimeError("LiteLLM response contained a provider generation error")
        message = _mapping(first_choice.get("message"))
        text = message.get("content")
        if not isinstance(text, str):
            raise ValueError("LiteLLM response did not contain a text choice")

        usage = _mapping(body.get("usage"))
        prompt_tokens = _required_int(usage.get("prompt_tokens"), "prompt_tokens")
        output_tokens = _required_int(usage.get("completion_tokens"), "completion_tokens")
        prompt_details_value = usage.get("prompt_tokens_details")
        prompt_details = _mapping(prompt_details_value) if prompt_details_value is not None else {}
        completion_details_value = usage.get("completion_tokens_details")
        completion_details = (
            _mapping(completion_details_value) if completion_details_value is not None else {}
        )
        cache_read = _optional_int(
            usage.get("cache_read_input_tokens", prompt_details.get("cached_tokens")),
            "cache_read_input_tokens",
        )
        cache_creation = _optional_int(
            usage.get("cache_creation_input_tokens", usage.get("cache_write_tokens")),
            "cache_creation_input_tokens",
        )
        uncached_input = prompt_tokens - cache_read - cache_creation
        if uncached_input < 0:
            raise ValueError("LiteLLM response cache token details exceed prompt_tokens")
        reasoning_tokens = _optional_int(
            completion_details.get("reasoning_tokens", usage.get("reasoning_tokens")),
            "reasoning_tokens",
        )

        return LLMResult(
            text=text,
            provider=self.provider,
            model=response_model,
            usage=LLMUsage(
                input_tokens=uncached_input,
                output_tokens=output_tokens,
                reasoning_tokens=reasoning_tokens,
                cache_read_input_tokens=cache_read,
                cache_creation_input_tokens=cache_creation,
            ),
        )
