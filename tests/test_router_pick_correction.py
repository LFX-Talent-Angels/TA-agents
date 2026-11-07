"""A corrected pick ("actually I meant 5") is a pick, not a search."""

from __future__ import annotations

import pytest

from talent_angels.session.router import route_line


@pytest.mark.parametrize(
    ("text", "number"),
    [
        ("actually I meant 5", 5),
        ("I meant 3", 3),
        ("no, number 2", 2),
        ("sorry, make it #4", 4),
        ("pick 7", 7),
        ("number 12", 12),
        ("5", 5),
    ],
)
def test_corrections_are_picks(text: str, number: int) -> None:
    routed = route_line(text)
    assert (routed.kind, routed.pick) == ("pick", number)


@pytest.mark.parametrize(
    "text", ["top 5 skills for a plumber", "I have 5 years in retail", "5 jobs for a chef"]
)
def test_sentences_with_numbers_are_not_picks(text: str) -> None:
    assert route_line(text).kind != "pick"
