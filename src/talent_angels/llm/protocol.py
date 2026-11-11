"""Provider-agnostic LLM interface (MVP plan Sec 2.8).

The assistant and skills depend only on this Protocol — never on a vendor
SDK directly. Swap providers via env (`LLM_PROVIDER`, `LLM_MODEL`) through
`talent_angels.llm.factory.get_llm_client`, no code changes required.
"""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field

#: One call. Short, because a turn makes several calls and has its own limit.
DEFAULT_LLM_TIMEOUT_SECONDS = 20.0
DEFAULT_LLM_NUM_RETRIES = 1
#: A whole turn (one line typed). Calls after it are skipped and the code
#: fallbacks answer: a slow provider minute must not become a ten-minute turn.
DEFAULT_TURN_DEADLINE_SECONDS = 45.0

_DEADLINE: ContextVar[float | None] = ContextVar("turn_deadline", default=None)
_CANCEL: ContextVar[threading.Event | None] = ContextVar("turn_cancel", default=None)


def _turn_deadline_seconds() -> float:
    raw = os.environ.get("TA_TURN_DEADLINE_SECONDS", "").strip()
    try:
        value = float(raw) if raw else DEFAULT_TURN_DEADLINE_SECONDS
    except ValueError:
        value = DEFAULT_TURN_DEADLINE_SECONDS
    return value if value > 0 else DEFAULT_TURN_DEADLINE_SECONDS


@contextmanager
def turn_deadline(seconds: float | None = None) -> Iterator[None]:
    """Bound every model call made in this context (and copies of it) by one clock.

    A deadline already set by an outer turn is kept, not extended.
    """
    if _DEADLINE.get() is not None:
        yield
        return
    token = _DEADLINE.set(time.monotonic() + (seconds or _turn_deadline_seconds()))
    try:
        yield
    finally:
        _DEADLINE.reset(token)


def bind_cancel(event: threading.Event) -> None:
    """Mark this context's turn as cancellable by ``event`` (the TUI's Esc)."""
    _CANCEL.set(event)


def turn_cancelled() -> bool:
    event = _CANCEL.get()
    return event is not None and event.is_set()


def seconds_left() -> float | None:
    """Time left in this turn, or None outside a turn."""
    deadline = _DEADLINE.get()
    return None if deadline is None else deadline - time.monotonic()


class LLMError(RuntimeError):
    """Any failure to get a usable completion: transport, provider, or parsing.

    Providers raise a zoo of types (``ValueError`` for a malformed body,
    ``anthropic.RateLimitError``, ``TimeoutError``…). Callers catch this one
    type and degrade to the deterministic path; a subclass of ``RuntimeError``
    so every existing ``except RuntimeError`` fallback keeps working.
    """


def _configured_timeout() -> float:
    raw = os.environ.get("LLM_TIMEOUT_SECONDS", "").strip()
    try:
        value = float(raw) if raw else DEFAULT_LLM_TIMEOUT_SECONDS
    except ValueError:
        value = DEFAULT_LLM_TIMEOUT_SECONDS
    return value if value > 0 else DEFAULT_LLM_TIMEOUT_SECONDS


def llm_timeout_seconds() -> float:
    """Per-request timeout, never past the turn's deadline.

    Library defaults (600-6000 s) would hang a turn.
    """
    configured = _configured_timeout()
    left = seconds_left()
    return configured if left is None else max(1.0, min(configured, left))


def llm_num_retries() -> int:
    """Retries per call; none when the turn has no room left for one."""
    raw = os.environ.get("LLM_NUM_RETRIES", "").strip()
    try:
        configured = max(0, int(raw)) if raw else DEFAULT_LLM_NUM_RETRIES
    except ValueError:
        configured = DEFAULT_LLM_NUM_RETRIES
    left = seconds_left()
    if left is not None and left < 2 * _configured_timeout():
        return 0
    return configured


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
