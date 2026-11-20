"""`jsonio.parse_reply`: fences, brace-matching, schema validation."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from talent_angels.tagging.jsonio import parse_reply


class _Reply(BaseModel):
    fit: str
    n: int = 0


def test_parses_plain_json() -> None:
    result = parse_reply('{"fit": "exact", "n": 2}', _Reply)
    assert result.fit == "exact"
    assert result.n == 2


def test_strips_fenced_json_block() -> None:
    text = """```json
    {"fit": "partial", "n": 1}
    ```"""
    result = parse_reply(text, _Reply)
    assert result.fit == "partial"
    assert result.n == 1


def test_strips_bare_fence_without_json_tag() -> None:
    text = '```\n{"fit": "none"}\n```'
    result = parse_reply(text, _Reply)
    assert result.fit == "none"


def test_takes_first_balanced_object_ignoring_trailing_prose() -> None:
    text = 'Sure, here you go: {"fit": "exact", "n": 3} Let me know if you need more.'
    result = parse_reply(text, _Reply)
    assert result.fit == "exact"
    assert result.n == 3


def test_braces_inside_strings_do_not_confuse_the_scan() -> None:
    text = '{"fit": "exact { not a brace } still a string", "n": 9}'
    result = parse_reply(text, _Reply)
    assert result.fit == "exact { not a brace } still a string"
    assert result.n == 9


def test_escaped_quote_inside_string_does_not_end_the_string_early() -> None:
    text = r'{"fit": "a \"quoted\" word { with brace }", "n": 4}'
    result = parse_reply(text, _Reply)
    assert result.n == 4


def test_no_json_object_raises_value_error() -> None:
    with pytest.raises(ValueError, match="no JSON object"):
        parse_reply("not json at all", _Reply)


def test_invalid_json_raises_value_error() -> None:
    with pytest.raises(ValueError, match="invalid JSON"):
        parse_reply("{not: valid json}", _Reply)


def test_schema_mismatch_raises_value_error() -> None:
    with pytest.raises(ValueError, match="_Reply"):
        parse_reply('{"n": "not an int for fit which is also missing"}', _Reply)
