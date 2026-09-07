"""Build a `MemoryClient` from environment variables (`MEMORY_PROVIDER`).

The only place in the codebase that chooses a memory provider by name —
everything else depends on the `MemoryClient` protocol. Mirrors
`talent_angels.llm.factory.get_llm_client`.
"""

from __future__ import annotations

import os

from talent_angels.memory.protocol import MemoryClient
from talent_angels.memory.stub_client import StubMemoryClient

SUPPORTED_PROVIDERS = ("none", "mem0")


def get_memory_client() -> MemoryClient:
    provider = os.environ.get("MEMORY_PROVIDER", "none").strip().lower() or "none"

    if provider == "none":
        return StubMemoryClient()

    if provider == "mem0":
        from talent_angels.memory.mem0_client import Mem0MemoryClient

        return Mem0MemoryClient()

    raise ValueError(
        f"Unsupported MEMORY_PROVIDER={provider!r}; supported: {', '.join(SUPPORTED_PROVIDERS)}"
    )
