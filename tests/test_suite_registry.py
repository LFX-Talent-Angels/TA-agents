"""Offline tests for suite selection and lifecycle."""

from __future__ import annotations

from contextlib import contextmanager

import pytest

from talent_angels.suites import SuiteRegistry, SuiteRuntime, UnknownSuiteError
from tests.fakes.taxonomy import FakeToolResult


class FakeSuite:
    def search_nodes(self, text: str, kind: str | None = None) -> FakeToolResult:
        return FakeToolResult()


def _factory(name: str, events: list[str]):
    @contextmanager
    def open_runtime():
        events.append(f"open:{name}")
        try:
            yield SuiteRuntime(name=name, suite=FakeSuite(), health_check=lambda: True)
        finally:
            events.append(f"close:{name}")

    return open_runtime


def test_registry_is_lazy_and_opens_the_default_suite() -> None:
    events: list[str] = []
    registry = SuiteRegistry(
        {"esco": _factory("esco", events), "onet": _factory("onet", events)},
        default="esco",
    )

    assert events == []
    assert registry.available == ("esco", "onet")
    assert registry.default == "esco"

    with registry.open() as runtime:
        assert runtime.name == "esco"
        assert runtime.is_reachable() is True
        assert events == ["open:esco"]

    assert events == ["open:esco", "close:esco"]


def test_registry_opens_only_the_explicit_suite() -> None:
    events: list[str] = []
    registry = SuiteRegistry(
        {"esco": _factory("esco", events), "onet": _factory("onet", events)},
        default="esco",
    )

    with registry.open("onet") as runtime:
        assert runtime.name == "onet"

    assert events == ["open:onet", "close:onet"]


@pytest.mark.parametrize("name", ["sfia", ""])
def test_registry_rejects_unknown_suite_with_available_names(name: str) -> None:
    registry = SuiteRegistry({"esco": _factory("esco", [])}, default="esco")

    with pytest.raises(UnknownSuiteError, match="unknown.*available: esco"):
        registry.open(name)


def test_registry_rejects_empty_or_missing_default() -> None:
    with pytest.raises(ValueError, match="at least one"):
        SuiteRegistry({}, default="esco")

    with pytest.raises(ValueError, match="default suite 'sfia' is not registered"):
        SuiteRegistry({"esco": _factory("esco", [])}, default="sfia")
