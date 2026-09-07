"""`MEMORY_PROVIDER=mem0` — real user memory via the open-source mem0 library.

Self-hosted `mem0.Memory`, not the managed Platform client: no `MEM0_API_KEY`,
no SaaS dependency. mem0 does the work the stub cannot — LLM fact extraction,
embedding-based retrieval, and semantic update/dedupe of contradictions.

mem0 needs its own LLM + embedder. It defaults to OpenAI (`OPENAI_API_KEY`),
which is independent of this repo's `LLM_PROVIDER`; point it elsewhere with
`MEM0_CONFIG` (JSON) if you'd rather not add a second provider.

The `mem0` import is deferred to call time so `MEMORY_PROVIDER=none` never pays
for the dependency — same pattern as `llm.factory`'s Anthropic import.
"""

from __future__ import annotations

import json
import os
from typing import Any

from talent_angels.memory.protocol import UserMemory


def _to_memories(response: Any) -> list[UserMemory]:
    """Normalize mem0's response shape into `UserMemory`.

    mem0 returns `{"results": [...]}` on current versions and a bare list on
    older ones; tolerate both rather than pinning to one release.
    """
    items = response.get("results", []) if isinstance(response, dict) else (response or [])
    memories: list[UserMemory] = []
    for item in items:
        text = item.get("memory") or item.get("text") or ""
        if not text:
            continue
        memories.append(
            UserMemory(
                id=str(item.get("id", "")),
                text=text,
                score=float(item.get("score") or 0.0),
            )
        )
    return memories


class Mem0MemoryClient:
    provider = "mem0"

    def __init__(self) -> None:
        from mem0 import Memory  # deferred: optional dependency

        raw_config = os.environ.get("MEM0_CONFIG", "").strip()
        if raw_config:
            try:
                config = json.loads(raw_config)
            except json.JSONDecodeError as exc:
                raise ValueError(f"MEM0_CONFIG is not valid JSON: {exc}") from exc
            self._memory = Memory.from_config(config)
        else:
            self._memory = Memory()

    def add(self, messages: list[dict[str, str]], *, user_id: str) -> None:
        self._memory.add(messages, user_id=user_id)

    def search(self, query: str, *, user_id: str, top_k: int = 5) -> list[UserMemory]:
        return _to_memories(self._memory.search(query, user_id=user_id, limit=top_k))

    def get_all(self, *, user_id: str) -> list[UserMemory]:
        return _to_memories(self._memory.get_all(user_id=user_id))

    def delete_all(self, *, user_id: str) -> None:
        self._memory.delete_all(user_id=user_id)
