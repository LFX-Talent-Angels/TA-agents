"""Provider-agnostic LLM interface (MVP plan Sec 2.8).

The assistant and skills depend only on this Protocol — never on a vendor
SDK directly. Swap providers via env (`LLM_PROVIDER`, `LLM_MODEL`) through
`talent_angels.llm.factory.get_llm_client`, no code changes required.
"""

from __future__ import annotations

import os
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field

DEFAULT_LLM_TIMEOUT_SECONDS = 60.0
DEFAULT_LLM_NUM_RETRIES = 1


class LLMError(RuntimeError):
    """Any failure to get a usable completion: transport, provider, or parsing.

    Providers raise a zoo of types (``ValueError`` for a malformed body,
    ``anthropic.RateLimitError``, ``TimeoutError``…). Callers catch this one
    type and degrade to the deterministic path; a subclass of ``RuntimeError``
    so every existing ``except RuntimeError`` fallback keeps working.
    """


def llm_timeout_seconds() -> float:
    """Per-request timeout. Library defaults (600-6000 s) would hang a turn."""
    raw = os.environ.get("LLM_TIMEOUT_SECONDS", "").strip()
    try:
        value = float(raw) if raw else DEFAULT_LLM_TIMEOUT_SECONDS
    except ValueError:
        value = DEFAULT_LLM_TIMEOUT_SECONDS
    return value if value > 0 else DEFAULT_LLM_TIMEOUT_SECONDS


def llm_num_retries() -> int:
    raw = os.environ.get("LLM_NUM_RETRIES", "").strip()
    try:
        return max(0, int(raw)) if raw else DEFAULT_LLM_NUM_RETRIES
    except ValueError:
        return DEFAULT_LLM_NUM_RETRIES


class ToolInvocation(BaseModel):
    id: str = ""
    name: str
    arguments: dict[str, object] = Field(default_factory=dict)


class Message(BaseModel):
    role: str  # "system" | "user" | "assistant" | "tool"
    content: str = ""
    tool_call_id: str | None = None
    tool_calls: list[ToolInvocation] | None = None


class LLMUsage(BaseModel):
    """OTel GenAI-aligned token usage (MVP plan Sec 2.6)."""

    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0


class LLMResult(BaseModel):
    text: str
    provider: str
    model: str
    usage: LLMUsage = Field(default_factory=LLMUsage)
    tool_calls: list[ToolInvocation] = Field(default_factory=list)


def uses_chat_phrasing(client: object | None) -> bool:
    """True for a real model client (not the ``none``/stub/test providers)."""
    if client is None:
        return False
    provider = str(getattr(client, "provider", "none") or "none").lower()
    return provider not in {"none", "stub", "test"}


@runtime_checkable
class LLMClient(Protocol):
    provider: str
    model: str

    def complete(
        self, messages: list[Message], *, tools: list[dict[str, object]] | None = None
    ) -> LLMResult:
        """Return drafted text plus usage. Zero tokens for the `none` stub."""
        ...
