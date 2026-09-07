"""User memory: what the assistant knows about the person it is talking to.

Distinct from the two stores that already exist: `runlog` records what the
system did, `assistant.cache` memoizes identical queries. Neither carries user
identity, so today every turn treats the user as a stranger. This package is
the missing piece — the "session/checkpoint store" open item in ARCHITECTURE.md.

`MEMORY_PROVIDER=none` (default) is a dependency-free stub for CI and offline
runs. `MEMORY_PROVIDER=mem0` uses open-source mem0 for LLM fact extraction,
embedding retrieval, and semantic dedupe.

Memory shapes *how* an answer is phrased, never *what* is claimed as taxonomy
fact — ARCHITECTURE.md rule #2 is unchanged.
"""

from talent_angels.memory.factory import get_memory_client
from talent_angels.memory.protocol import MemoryClient, UserMemory
from talent_angels.memory.stub_client import StubMemoryClient

__all__ = ["MemoryClient", "StubMemoryClient", "UserMemory", "get_memory_client"]
