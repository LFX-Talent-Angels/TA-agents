"""Which suite's groups are offered as "which area?" when several report some."""

from __future__ import annotations

from talent_angels.session.narrow import area_choices
from talent_angels.skills.locate.areas import Area


def _areas(suite: str, *labels: str) -> list[Area]:
    return [Area(suite=suite, code=str(i), label=label, count=3) for i, label in enumerate(labels)]


def test_esco_groups_win_over_bls_even_though_bls_sorts_first() -> None:
    by_suite = {
        "bls": _areas("bls", "Architecture and Engineering", "Management"),
        "esco": _areas("esco", "Civil engineers", "Mechanical engineers"),
        "jobtech": _areas("jobtech", "Chefsyrken", "Yrken med krav på högskolekompetens"),
    }
    choices = area_choices(by_suite, query="engineer", kind="occupation")
    assert [c.label for c in choices] == ["Civil engineers", "Mechanical engineers"]
    assert {c.suite for c in choices} == {"esco"}


def test_a_suite_with_one_group_is_skipped() -> None:
    by_suite = {"esco": _areas("esco", "Civil engineers"), "bls": _areas("bls", "A", "B")}
    assert [c.suite for c in area_choices(by_suite, query="q", kind=None)] == ["bls", "bls"]
