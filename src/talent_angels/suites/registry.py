"""Typed suite selection and lifecycle, independent of concrete taxonomies."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass

from talent_angels.suites.protocol import SuiteTools

HealthCheck = Callable[[], bool]


@dataclass(frozen=True)
class SuiteRuntime:
    """An opened taxonomy suite plus infrastructure owned by its adapter."""

    name: str
    suite: SuiteTools
    health_check: HealthCheck

    def is_reachable(self) -> bool:
        """Report adapter infrastructure availability without exposing its driver."""
        return self.health_check()


SuiteFactory = Callable[[], AbstractContextManager[SuiteRuntime]]


class UnknownSuiteError(ValueError):
    """Raised when a plan requests a suite that is not registered."""


class SuiteRegistry:
    """Map suite names to lazy, context-managed runtime factories."""

    def __init__(self, factories: Mapping[str, SuiteFactory], *, default: str) -> None:
        if not factories:
            raise ValueError("suite registry requires at least one suite")
        if default not in factories:
            raise ValueError(f"default suite {default!r} is not registered")
        self._factories = dict(factories)
        self._default = default

    @property
    def available(self) -> tuple[str, ...]:
        return tuple(sorted(self._factories))

    @property
    def default(self) -> str:
        return self._default

    def open(self, name: str | None = None) -> AbstractContextManager[SuiteRuntime]:
        selected = self._default if name is None else name
        try:
            factory = self._factories[selected]
        except KeyError as exc:
            available = ", ".join(self.available)
            raise UnknownSuiteError(f"unknown suite {selected!r}; available: {available}") from exc
        return factory()


@contextmanager
def _open_default_esco() -> Iterator[SuiteRuntime]:
    """Load the optional concrete adapter only when ESCO is opened."""
    from talent_angels.suites.esco import open_esco_runtime

    with open_esco_runtime() as runtime:
        yield runtime


def default_suite_registry() -> SuiteRegistry:
    """Build the MVP registry; constructing it performs no database work."""
    return SuiteRegistry({"esco": _open_default_esco}, default="esco")
