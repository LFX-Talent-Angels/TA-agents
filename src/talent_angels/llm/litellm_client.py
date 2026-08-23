"""Unified LiteLLM adapter behind TA-agents' internal LLMClient protocol."""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping
from typing import Any

from talent_angels.llm.protocol import LLMResult, LLMUsage, Message, ToolInvocation

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


def _parse_tool_calls(value: object) -> list[ToolInvocation]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError("LiteLLM response tool_calls must be a list")
    parsed: list[ToolInvocation] = []
    for item in value:
        payload = _mapping(item)
        function = payload.get("function", payload)
        fn = _mapping(function) if function is not None else {}
        name = fn.get("name") or payload.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError("LiteLLM tool call requires a name")
        raw_args = fn.get("arguments", payload.get("arguments", {}))
        arguments: dict[str, object]
        if raw_args is None or raw_args == "":
            arguments = {}
        elif isinstance(raw_args, dict):
            arguments = dict(raw_args)
        elif isinstance(raw_args, str):
            loaded = json.loads(raw_args) if raw_args.strip() else {}
            if not isinstance(loaded, dict):
                raise ValueError("LiteLLM tool call arguments must be an object")
            arguments = loaded
        else:
            raise ValueError("LiteLLM tool call arguments must be an object")
        call_id = payload.get("id")
        parsed.append(
            ToolInvocation(
                id=call_id if isinstance(call_id, str) else "",
                name=name,
                arguments=arguments,
            )
        )
    return parsed


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
            os.environ.setdefault("LITELLM_LOG", "ERROR")
            import contextlib
            import io

            # LiteLLM prints a "Provider List" banner on import; keep the CLI clean.
            quiet = contextlib.redirect_stdout(io.StringIO())
            with quiet, contextlib.redirect_stderr(io.StringIO()):
                from litellm import completion

            completion_fn = completion
        self._completion = completion_fn

    def complete(
        self, messages: list[Message], *, tools: list[dict[str, object]] | None = None
    ) -> LLMResult:
        kwargs: dict[str, object] = {
            "model": self.model,
            "messages": [message.model_dump(exclude_none=True) for message in messages],
            "max_tokens": DEFAULT_MAX_TOKENS,
        }
        if tools:
            # OpenAI-shaped tools only. Do not send tool_choice="auto": some
            # OpenRouter upstreams (Nvidia) parse that as a named function call
            # and return 400 "missing field `function`".
            kwargs["tools"] = tools
        if self.reasoning_enabled and not tools:
            # Reasoning extra_body plus tools has broken several OpenRouter
            # providers; keep reasoning for plain completions only.
            kwargs["extra_body"] = {"reasoning": {"enabled": True}}

        try:
            raw = self._completion(**kwargs)
        except Exception as first:
            if "extra_body" in kwargs:
                kwargs.pop("extra_body", None)
                try:
                    raw = self._completion(**kwargs)
                except Exception as exc:
                    raise RuntimeError(f"LiteLLM provider request failed: {exc}") from exc
            else:
                raise RuntimeError(f"LiteLLM provider request failed: {first}") from first
        body = _mapping(raw)
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
        if text is None:
            text = ""
        if not isinstance(text, str):
            raise ValueError("LiteLLM response did not contain a text choice")
        tool_calls = _parse_tool_calls(message.get("tool_calls"))
        if not text and not tool_calls:
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
            tool_calls=tool_calls,
        )
