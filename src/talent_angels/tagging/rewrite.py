"""Model call rw-1: per parent-group rewrite of a client line into search phrases.

One call covers up to 6 units sharing a `group_key` (the runner does the
chunking); local ids `r1..rN` map 1:1 to `units`' order so the model never
sees — and can't echo back — a real `unit_key`. A bad reply (JSON/LLMError)
gets one retry after splitting the group in half; a unit that still fails
falls back to an empty `Rewrite` (`queries_for` then uses its own default:
`seed_phrases` or the cleaned text).
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

from pydantic import BaseModel, Field

from talent_angels.assistant.llm_call import measure_complete
from talent_angels.llm import LLMClient, LLMError, Message
from talent_angels.runlog import StageUsage
from talent_angels.tagging import jsonio
from talent_angels.tagging.models import TagUnit
from talent_angels.tagging.prompts import REWRITE_SYSTEM, rewrite_message

_SEP = "\x1f"


class Rewrite(BaseModel):
    """One row's model-proposed search phrases. Empty = fallback to defaults."""

    labels: list[str] = Field(default_factory=list)
    description: str | None = None


class _RewriteReply(BaseModel):
    rows: dict[str, Rewrite] = Field(default_factory=dict)


def rewrite_cache_key(unit: TagUnit, model: str, prompts_version: str) -> str:
    """Cache key for `<project>/.ta-tag/rewrites.jsonl` — same shape as `keys.fingerprint`."""
    parts = [unit.unit_key, unit.clean_text, *unit.context_path, model, prompts_version]
    return f"rw_{hashlib.sha256(_SEP.join(parts).encode('utf-8')).hexdigest()[:24]}"


def _call(
    client: LLMClient, units: Sequence[TagUnit], doc_context: Sequence[str]
) -> tuple[_RewriteReply, StageUsage]:
    messages = [
        Message(role="system", content=REWRITE_SYSTEM),
        Message(role="user", content=rewrite_message(units, doc_context)),
    ]
    result, stage = measure_complete(client, messages, stage="tag_rewrite")
    try:
        reply = jsonio.parse_reply(result.text, _RewriteReply)
    except ValueError as exc:
        # Billed even though unusable: keep the usage so the run's cost is true.
        raise _BadReply(stage) from exc
    return reply, stage


class _BadReply(ValueError):
    def __init__(self, stage: StageUsage) -> None:
        super().__init__("unusable rewrite reply")
        self.stage = stage


def _attempt(
    client: LLMClient,
    units: Sequence[TagUnit],
    doc_context: Sequence[str],
    *,
    allow_split: bool,
) -> tuple[dict[str, Rewrite], list[StageUsage]]:
    if not units:
        return {}, []
    try:
        reply, stage = _call(client, units, doc_context)
    except (LLMError, ValueError) as exc:
        spent = [exc.stage] if isinstance(exc, _BadReply) else []
        if allow_split and len(units) > 1:
            mid = len(units) // 2
            left_map, left_stages = _attempt(client, units[:mid], doc_context, allow_split=False)
            right_map, right_stages = _attempt(client, units[mid:], doc_context, allow_split=False)
            return {**left_map, **right_map}, [*spent, *left_stages, *right_stages]
        return {unit.unit_key: Rewrite() for unit in units}, spent

    result: dict[str, Rewrite] = {}
    for index, unit in enumerate(units, start=1):
        row = reply.rows.get(f"r{index}")
        result[unit.unit_key] = row if row is not None else Rewrite()
    return result, [stage]


def rewrite_group(
    client: LLMClient, units: list[TagUnit], doc_context: list[str]
) -> tuple[dict[str, Rewrite], list[StageUsage]]:
    """Rewrite every unit in one parent group (<= 6 units) in a single call.

    Returns a `Rewrite` for every `unit.unit_key` passed in — never a partial
    dict — so the caller can apply the cache/fallback uniformly.
    """
    return _attempt(client, units, doc_context, allow_split=True)
