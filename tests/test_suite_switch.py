"""'show me that in <suite>' and suite names in any case (BX-14, BX-24)."""

from __future__ import annotations

import pytest

from talent_angels.assistant.suite_select import select_suites
from talent_angels.session.kernel import _switch_suite
from talent_angels.suites import SuiteRegistry
from tests.test_suite_registry import _factory


@pytest.mark.parametrize(
    ("text", "suite"),
    [
        ("show me that in jobtech", "jobtech"),
        ("and in O*NET?", "onet"),
        ("same in BLS", "bls"),
        ("i jobtech", "jobtech"),
    ],
)
def test_bare_switches(text: str, suite: str) -> None:
    assert _switch_suite(text) == suite


@pytest.mark.parametrize(
    "text", ["what does a baker do in bls?", "show me nurses in esco", "baker in lightcast"]
)
def test_new_questions_are_not_switches(text: str) -> None:
    assert _switch_suite(text) is None


def test_suite_names_in_any_case() -> None:
    events: list[str] = []
    registry = SuiteRegistry(
        {"esco": _factory("esco", events), "jobtech": _factory("jobtech", events)}, default="esco"
    )
    assert registry.resolve("JobTech") == "jobtech"
    with registry.open("JOBTECH") as runtime:
        assert runtime.name == "jobtech"
    assert select_suites(available=("esco", "jobtech"), override="JobTech") == ("jobtech",)
