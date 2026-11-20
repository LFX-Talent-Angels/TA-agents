"""A miss is phrased without inventing titles or codes (BX-01)."""

from __future__ import annotations

import pytest

from talent_angels.session.phrase import _looks_like_invented_miss


@pytest.mark.parametrize(
    "text",
    [
        "**SSYK 7612 – Skorstensfejare** This is the Swedish occupation code for chimney sweeps.",
        "SOC 15-1250 covers Software Developers and Testers.",
        "I don't have access to the JobTech API or the SSYK database in real time.",
        "Standard SOC codes follow formats like 00-0000 through 99-0000.",
    ],
)
def test_invented_misses_are_caught(text: str) -> None:
    assert _looks_like_invented_miss(text)


def test_an_honest_miss_passes() -> None:
    assert not _looks_like_invented_miss(
        "I couldn't find that on the map. Try another title or name the taxonomy."
    )
