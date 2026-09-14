"""Suite selection: override, named source, else all attached."""

from __future__ import annotations

import pytest

from talent_angels.assistant.suite_select import (
    named_suites,
    named_unattached,
    select_suites,
)
from talent_angels.suites import UnknownSuiteError

_ATTACHED = ("esco", "onet")


def test_generic_question_selects_every_attached_suite() -> None:
    assert select_suites(available=_ATTACHED, question="what is a software developer") == (
        "esco",
        "onet",
    )


def test_override_wins_even_when_the_question_names_another_source() -> None:
    assert select_suites(
        available=_ATTACHED,
        override="onet",
        question="where is firefighter in ESCO",
    ) == ("onet",)


def test_named_esco_narrows() -> None:
    assert select_suites(available=_ATTACHED, question="where is firefighter in ESCO") == ("esco",)


def test_named_onet_aliases_narrow() -> None:
    assert select_suites(available=_ATTACHED, question="O*NET software developer") == ("onet",)
    assert select_suites(available=_ATTACHED, question="onet importance of programming") == (
        "onet",
    )


def test_both_names_keep_both_attached() -> None:
    assert select_suites(available=_ATTACHED, question="compare ESCO and O*NET nurses") == (
        "esco",
        "onet",
    )


def test_named_source_that_is_not_attached_is_ignored() -> None:
    assert named_suites("look this up in SFIA", _ATTACHED) == ()
    assert select_suites(available=_ATTACHED, question="look this up in SFIA") == _ATTACHED


def test_suite_name_is_a_token_not_a_substring() -> None:
    assert named_suites("bayonet", _ATTACHED) == ()
    assert select_suites(available=_ATTACHED, question="bayonet") == _ATTACHED
    assert named_suites("sonet mixer", _ATTACHED) == ()
    assert named_suites("shop at tesco", _ATTACHED) == ()  # esco not inside tesco


def test_o_net_spellings_still_narrow() -> None:
    assert select_suites(available=_ATTACHED, question="in O NET") == ("onet",)
    assert select_suites(available=_ATTACHED, question="o-net software") == ("onet",)


def test_unknown_override_raises() -> None:
    with pytest.raises(UnknownSuiteError, match="sfia"):
        select_suites(available=_ATTACHED, override="sfia")


def test_named_unattached_suite_is_reported() -> None:
    assert named_unattached("look this up in SFIA", _ATTACHED) == ("sfia",)
    assert named_unattached("software developer", _ATTACHED) == ()


def test_empty_available_raises() -> None:
    with pytest.raises(ValueError, match="no suites attached"):
        select_suites(available=())
