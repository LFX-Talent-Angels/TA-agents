"""In-process TTL result cache — the efficiency experiment (MVP plan Sec 2.7).

Keyed by (suite, capability, normalized query). Process-lifetime only, no
shared/cross-process cache — enough to measure the mechanism's effect on
repeated identical Locate queries within one run of the golden set.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from talent_angels.contracts import AgentResult

_Key = tuple[str, str, str]


@dataclass
class _Entry:
    result: AgentResult
    expires_at: float


class ResultCache:
    def __init__(self, ttl_seconds: float = 300.0) -> None:
        self._ttl = ttl_seconds
        self._store: dict[_Key, _Entry] = {}

    @staticmethod
    def _key(suite: str, capability: str, question: str) -> _Key:
        return (suite, capability, question.strip().lower())

    def get(self, suite: str, capability: str, question: str) -> AgentResult | None:
        key = self._key(suite, capability, question)
        entry = self._store.get(key)
        if entry is None:
            return None
        if entry.expires_at < time.monotonic():
            del self._store[key]
            return None
        return entry.result

    def set(self, suite: str, capability: str, question: str, result: AgentResult) -> None:
        key = self._key(suite, capability, question)
        self._store[key] = _Entry(result=result, expires_at=time.monotonic() + self._ttl)
