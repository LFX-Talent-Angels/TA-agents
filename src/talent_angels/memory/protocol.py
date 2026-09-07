"""Provider-agnostic user-memory interface (mirrors `talent_angels.llm.protocol`).

The assistant depends only on this Protocol — never on a vendor SDK directly.
Swap providers via env (`MEMORY_PROVIDER`) through
`talent_angels.memory.factory.get_memory_client`, no code changes required.

Memory is *user* knowledge (role, goals, preferences), not taxonomy fact.
ARCHITECTURE.md rule #2 still holds: only graph data is cited as taxonomy
fact, so retrieved memories may shape *phrasing and emphasis* but are never
presented as taxonomy evidence.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import BaseModel


class UserMemory(BaseModel):
    """A single fact the assistant has learned about a user."""

    id: str
    text: str
    score: float = 0.0


@runtime_checkable
class MemoryClient(Protocol):
    """Store and retrieve per-user memories for personalization."""

    provider: str

    def add(self, messages: list[dict[str, str]], *, user_id: str) -> None:
        """Extract and persist facts from a conversation for `user_id`."""
        ...

    def search(self, query: str, *, user_id: str, top_k: int = 5) -> list[UserMemory]:
        """Return the memories for `user_id` most relevant to `query`."""
        ...

    def get_all(self, *, user_id: str) -> list[UserMemory]:
        """Return every stored memory for `user_id`."""
        ...

    def delete_all(self, *, user_id: str) -> None:
        """Erase every memory for `user_id` (right-to-be-forgotten)."""
        ...
