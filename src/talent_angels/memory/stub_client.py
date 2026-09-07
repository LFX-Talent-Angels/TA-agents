"""`MEMORY_PROVIDER=none` — deterministic, dependency-free in-process memory.

Default for CI and offline runs, mirroring `llm.stub_client.StubLLMClient`:
the integration is exercised end-to-end with zero network calls, zero tokens,
and no vector store. Facts are whole user utterances rather than LLM-extracted
claims, and `search` is token-overlap scoring rather than embeddings — enough
to prove the wiring, deliberately not enough to pass for real personalization.

Process-lifetime only: nothing here persists across restarts (same tradeoff as
`assistant.cache.ResultCache`). Use `MEMORY_PROVIDER=mem0` for real memory.
"""

from __future__ import annotations

from talent_angels.memory.protocol import UserMemory

_MIN_FACT_LEN = 3


class StubMemoryClient:
    provider = "none"

    def __init__(self) -> None:
        self._store: dict[str, list[UserMemory]] = {}
        self._counter = 0

    def add(self, messages: list[dict[str, str]], *, user_id: str) -> None:
        facts = self._store.setdefault(user_id, [])
        for message in messages:
            if message.get("role") != "user":
                continue  # only the user's own words are treated as user facts
            text = (message.get("content") or "").strip()
            if len(text) < _MIN_FACT_LEN:
                continue
            if any(existing.text.lower() == text.lower() for existing in facts):
                continue  # naive dedupe; real semantic merge is mem0's job
            self._counter += 1
            facts.append(UserMemory(id=f"stub-{self._counter}", text=text, score=1.0))

    def search(self, query: str, *, user_id: str, top_k: int = 5) -> list[UserMemory]:
        terms = {t for t in query.lower().split() if t}
        if not terms:
            return []
        scored: list[UserMemory] = []
        for fact in self._store.get(user_id, []):
            fact_terms = {t for t in fact.text.lower().split() if t}
            overlap = len(terms & fact_terms)
            if overlap:
                scored.append(
                    UserMemory(id=fact.id, text=fact.text, score=overlap / len(terms))
                )
        scored.sort(key=lambda m: m.score, reverse=True)
        return scored[:top_k]

    def get_all(self, *, user_id: str) -> list[UserMemory]:
        return list(self._store.get(user_id, []))

    def delete_all(self, *, user_id: str) -> None:
        self._store.pop(user_id, None)
