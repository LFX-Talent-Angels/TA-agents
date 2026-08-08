"""`LLM_PROVIDER=none` — deterministic, zero-token stub.

Default for CI and offline graph demos: Locate is driven by `search_nodes`
(deterministic tool), so most turns never need this at all. When the
assistant does ask for a drafted answer, the stub echoes the last user
message instead of calling out to a real model.
"""

from __future__ import annotations

from talent_angels.llm.protocol import LLMResult, LLMUsage, Message


class StubLLMClient:
    provider = "none"

    def __init__(self, model: str = "stub") -> None:
        self.model = model

    def complete(self, messages: list[Message]) -> LLMResult:
        last_user = next((m.content for m in reversed(messages) if m.role == "user"), "")
        return LLMResult(
            text=last_user,
            provider=self.provider,
            model=self.model,
            usage=LLMUsage(),
        )
