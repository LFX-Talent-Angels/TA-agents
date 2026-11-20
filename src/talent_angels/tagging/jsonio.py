"""Parse a model's reply into a typed object: fences, brace-match, validate.

The one place every model-step parser (`rewrite.py`, `choose.py`) goes
through, so "the model wrapped it in ```json" or "there's trailing prose
after the object" are handled once.
"""

from __future__ import annotations

import json
import re
from typing import TypeVar

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE | re.MULTILINE)


def _strip_fences(text: str) -> str:
    return _FENCE_RE.sub("", text.strip()).strip()


def _first_json_object(text: str) -> str | None:
    """The first balanced `{...}` in `text`, honouring strings and escapes.

    Same brace-depth scan as `assistant/agent_loop.py`'s own helper — kept
    separate here so `tagging/` never imports from `assistant/`.
    """
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape = False
    for i, ch in enumerate(text[start:], start):
        if escape:
            escape = False
            continue
        if ch == "\\" and in_string:
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def parse_reply(text: str, model: type[T]) -> T:
    """Strip fences, take the first balanced JSON object, validate into `model`.

    Raises `ValueError` on any failure (no object found, bad JSON, schema
    mismatch) so callers can catch one exception type and fall back.
    """
    cleaned = _strip_fences(text)
    raw = _first_json_object(cleaned)
    if raw is None:
        raise ValueError("no JSON object found in model reply")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in model reply: {exc}") from exc
    try:
        return model.model_validate(payload)
    except ValidationError as exc:
        raise ValueError(f"model reply did not match {model.__name__}: {exc}") from exc
