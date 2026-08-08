"""`LLM_PROVIDER=anthropic` — the one real provider path wired for Gate A.

Only this module imports the `anthropic` SDK; the rest of the codebase talks
to `LLMClient` (protocol.py) so a second provider is a second file here, not
a rewrite of the assistant.
"""

from __future__ import annotations

import os
from typing import Any

import anthropic

from talent_angels.llm.protocol import LLMResult, LLMUsage, Message

DEFAULT_MAX_TOKENS = 1024


class AnthropicLLMClient:
    provider = "anthropic"

    def __init__(self, model: str, api_key: str | None = None) -> None:
        self.model = model
        key = api_key or os.environ.get("LLM_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise ValueError(
                "LLM_PROVIDER=anthropic requires LLM_API_KEY or ANTHROPIC_API_KEY to be set"
            )
        self._client = anthropic.Anthropic(api_key=key)

    def complete(self, messages: list[Message]) -> LLMResult:
        system_text = "\n".join(m.content for m in messages if m.role == "system")
        turns: list[dict[str, Any]] = [
            {"role": m.role, "content": m.content} for m in messages if m.role != "system"
        ]

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": turns,
            "max_tokens": DEFAULT_MAX_TOKENS,
        }
        if system_text:
            kwargs["system"] = system_text

        response = self._client.messages.create(**kwargs)
        text = "".join(block.text for block in response.content if block.type == "text")
        usage = LLMUsage(
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            cache_read_input_tokens=getattr(response.usage, "cache_read_input_tokens", 0) or 0,
            cache_creation_input_tokens=getattr(response.usage, "cache_creation_input_tokens", 0)
            or 0,
        )
        return LLMResult(text=text, provider=self.provider, model=self.model, usage=usage)
