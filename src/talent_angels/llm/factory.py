"""Build an `LLMClient` from environment variables (`LLM_PROVIDER`, `LLM_MODEL`).

This is the only place in the codebase that chooses a provider implementation
by name. Everything else — assistant, skills — depends on `LLMClient` only.
"""

from __future__ import annotations

import os

from talent_angels.llm.protocol import LLMClient
from talent_angels.llm.stub_client import StubLLMClient

SUPPORTED_PROVIDERS = ("none", "anthropic")


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

    raise ValueError(
        f"Unsupported LLM_PROVIDER={provider!r}; supported: {', '.join(SUPPORTED_PROVIDERS)}"
    )
