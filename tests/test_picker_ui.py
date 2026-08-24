"""Matching for the model picker: find it from any part of the name."""

from __future__ import annotations

import types

import pytest

import talent_angels.tui.picker_ui as picker_ui
from talent_angels.tui.picker_ui import Option, filter_options, subsequence_match

MODELS = [
    "poolside/laguna-s-2.1:free",
    "nvidia/nemotron-3-ultra-550b-a55b:free",
    "nvidia/nemotron-3-nano-30b-a3b:free",
    "google/gemma-4-31b-it:free",
    "cohere/north-mini-code:free",
]
OPTIONS = [Option(key=name, label=name) for name in MODELS]


def _keys(query: str) -> list[str]:
    return [option.key for option in filter_options(OPTIONS, query)]


def test_matches_the_model_name_without_the_vendor() -> None:
    """Ids read `vendor/model`, and the vendor is the half nobody remembers."""
    assert _keys("laguna") == ["poolside/laguna-s-2.1:free"]
    assert _keys("gemma") == ["google/gemma-4-31b-it:free"]


def test_matches_characters_in_order_when_no_substring_fits() -> None:
    """A half-remembered name should still land."""
    assert subsequence_match("nemultra", "nvidia/nemotron-3-ultra-550b-a55b:free") is True
    assert _keys("nemultra") == ["nvidia/nemotron-3-ultra-550b-a55b:free"]


def test_exact_substrings_rank_above_looser_matches() -> None:
    """Both nemotron models match "nemotron"; the ordering must be stable and
    put the literal match first so the cursor starts on it."""
    matches = _keys("nemotron-3-nano")
    assert matches[0] == "nvidia/nemotron-3-nano-30b-a3b:free"


def test_ordering_is_required_not_just_membership() -> None:
    """Without order, a few common letters would match everything."""
    assert subsequence_match("anugal", "poolside/laguna-s-2.1:free") is False


def test_empty_query_keeps_every_option() -> None:
    assert _keys("") == MODELS
    assert _keys("   ") == MODELS


def test_no_match_returns_nothing_rather_than_everything() -> None:
    assert _keys("zzzz") == []


def test_bare_escape_does_not_read_a_second_byte(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(picker_ui.sys, "stdin", types.SimpleNamespace(fileno=lambda: 7))
    monkeypatch.setattr(
        picker_ui,
        "select_module",
        types.SimpleNamespace(select=lambda *_: ([], [], [])),
        raising=False,
    )
    monkeypatch.setattr(picker_ui.os, "read", lambda *_: b"\x1b", raising=False)

    assert picker_ui._read_key() == "\x1b"


def test_down_arrow_reads_the_full_escape_sequence(monkeypatch: pytest.MonkeyPatch) -> None:
    chunks = iter([b"\x1b", b"[", b"B"])
    monkeypatch.setattr(picker_ui.sys, "stdin", types.SimpleNamespace(fileno=lambda: 7))
    monkeypatch.setattr(
        picker_ui,
        "select_module",
        types.SimpleNamespace(select=lambda *_: ([7], [], [])),
        raising=False,
    )
    monkeypatch.setattr(picker_ui.os, "read", lambda *_: next(chunks), raising=False)

    assert picker_ui._read_key() == "\x1b[B"
