"""Build an `LLMClient` from environment variables (`LLM_PROVIDER`, `LLM_MODEL`).

This is the only place in the codebase that chooses a provider implementation
by name. Everything else — assistant, skills — depends on `LLMClient` only.
"""

from __future__ import annotations

import os

from talent_angels.llm.protocol import LLMClient
from talent_angels.llm.stub_client import StubLLMClient

SUPPORTED_PROVIDERS = ("none", "anthropic", "litellm")
KEY_PLACEHOLDER = "ADD_REPLACEMENT_KEY_HERE"


def get_answer_mode() -> str:
    """Return the configured answer shape; default is zero-token structured text."""
    return os.environ.get("ANSWER_MODE", "structured").strip() or "structured"


def get_llm_client() -> LLMClient:
    provider = os.environ.get("LLM_PROVIDER", "none").strip().lower()
    model = os.environ.get("LLM_MODEL", "").strip()

    if provider == "none":
        return StubLLMClient(model=model or "stub")

    if provider == "anthropic":
        if not model:
            raise ValueError("LLM_MODEL must be set when LLM_PROVIDER=anthropic")
        from talent_angels.llm.anthropic_client import AnthropicLLMClient

        return AnthropicLLMClient(model=model)

    if provider == "litellm":
        if not model:
            raise ValueError("LLM_MODEL must be set when LLM_PROVIDER=litellm")
        if model.startswith("openrouter/"):
            openrouter_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
            if not openrouter_key or openrouter_key == KEY_PLACEHOLDER:
                raise ValueError("LiteLLM OpenRouter models require OPENROUTER_API_KEY to be set")
        from talent_angels.llm.litellm_client import LiteLLMClient

        reasoning_enabled = os.environ.get("LLM_REASONING_ENABLED", "false").strip().lower()
        return LiteLLMClient(
            model=model,
            reasoning_enabled=reasoning_enabled in {"1", "true", "yes", "on"},
        )

    raise ValueError(
        f"Unsupported LLM_PROVIDER={provider!r}; supported: {', '.join(SUPPORTED_PROVIDERS)}"
    )
