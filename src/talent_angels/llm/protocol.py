"""Provider-agnostic LLM interface (MVP plan Sec 2.8).

The assistant and skills depend only on this Protocol — never on a vendor
SDK directly. Swap providers via env (`LLM_PROVIDER`, `LLM_MODEL`) through
`talent_angels.llm.factory.get_llm_client`, no code changes required.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field


class Message(BaseModel):
    role: str  # "system" | "user" | "assistant"
    content: str


class LLMUsage(BaseModel):
    """OTel GenAI-aligned token usage (MVP plan Sec 2.6)."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0


class LLMResult(BaseModel):
    text: str
    provider: str
    model: str
    usage: LLMUsage = Field(default_factory=LLMUsage)


@runtime_checkable
class LLMClient(Protocol):
    provider: str
    model: str

    def complete(self, messages: list[Message]) -> LLMResult:
        """Return drafted text plus usage. Zero tokens for the `none` stub."""
        ...
