"""Typed suite selection and lifecycle, independent of concrete taxonomies."""

from __future__ import annotations

import importlib.util
import logging
from collections.abc import Callable, Iterator, Mapping
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass

from talent_angels.suites.protocol import SuiteTools

_log = logging.getLogger(__name__)

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
        """Suite names in registration order.

        The order is what users see first: answers, pick lists, areas and the
        first goal saved all follow it. It is the order suites were added
        (ESCO, O*NET, SFIA, then BLS and JobTech), not alphabetical, so a new
        suite never jumps ahead of the ones users know.
        """
        return tuple(self._factories)

    @property
    def default(self) -> str:
        return self._default

    def resolve(self, name: str) -> str:
        """The registered name for ``name`` in any case ("JobTech" -> "jobtech")."""
        if name in self._factories:
            return name
        folded = {key.casefold(): key for key in self._factories}
        return folded.get(name.strip().casefold(), name)

    def open(self, name: str | None = None) -> AbstractContextManager[SuiteRuntime]:
        selected = self._default if name is None else self.resolve(name)
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


@contextmanager
def _open_default_onet() -> Iterator[SuiteRuntime]:
    """Load the optional concrete adapter only when O*NET is opened."""
    from talent_angels.suites.onet import open_onet_runtime

    with open_onet_runtime() as runtime:
        yield runtime


@contextmanager
def _open_default_sfia() -> Iterator[SuiteRuntime]:
    """Load the optional concrete adapter only when SFIA is opened."""
    from talent_angels.suites.sfia import open_sfia_runtime

    with open_sfia_runtime() as runtime:
        yield runtime


@contextmanager
def _open_default_bls() -> Iterator[SuiteRuntime]:
    """Load the optional concrete adapter only when BLS is opened."""
    from talent_angels.suites.bls import open_bls_runtime

    with open_bls_runtime() as runtime:
        yield runtime


@contextmanager
def _open_default_jobtech() -> Iterator[SuiteRuntime]:
    """Load the optional concrete adapter only when JobTech is opened."""
    from talent_angels.suites.jobtech import open_jobtech_runtime

    with open_jobtech_runtime() as runtime:
        yield runtime


def _suite_installed(name: str) -> bool:
    """False only when an importable ta-taxonomies clearly lacks this suite (an old pin)."""
    try:
        importlib.util.find_spec("ta_taxonomies.suites")
    except ModuleNotFoundError:
        # No package to inspect: stay lazy, and opening the suite says why.
        return True
    try:
        return importlib.util.find_spec(f"ta_taxonomies.suites.{name}.tools") is not None
    except ModuleNotFoundError:
        return False


def with_neighbor_cache(factory: SuiteFactory) -> SuiteFactory:
    """Wrap one suite factory so every open() serves get_neighbors from SQLite.

    Applied in ``default_suite_registry`` — the single place the TUI, API, MCP
    edge, and CLI all build their registry — rather than inside each edge. A
    per-edge opt-in is exactly the shape of bug this closes: the cache layer
    existed, was tested, and never fired for a real user because no edge
    remembered to pass it. Here it cannot be forgotten.
    """

    @contextmanager
    def _open() -> Iterator[SuiteRuntime]:
        from talent_angels.memory.cache import CachedSuite

        with factory() as runtime:
            yield SuiteRuntime(
                name=runtime.name,
                suite=CachedSuite(runtime.suite),
                health_check=runtime.health_check,
            )

    return _open


def default_suite_registry(*, neighbor_cache: bool = True) -> SuiteRegistry:
    """Build the registry; constructing it performs no database work.

    Every attached suite is available to the assistant. ``open(name)`` is the
    debug override (CLI ``--suite`` / API ``suite``). Omit the override to
    search all attached taxonomies. Suites stay separate graphs.

    ``neighbor_cache=True`` (the default) wraps each suite in ``CachedSuite``,
    so repeated Connect queries for the same occupation are served from
    ``memory.db`` instead of re-querying Neo4j — Sprint 6's "no repeated graph
    queries". Pass ``False`` where a cache would corrupt the measurement or the
    expectation: the ``bench`` command exists to measure cache effect, and
    ``quality`` scores live answers whose golden values assume current graph
    data. Tests that construct ``SuiteRegistry`` directly are unaffected.
    """
    factories: dict[str, SuiteFactory] = {
        "esco": _open_default_esco,
        "onet": _open_default_onet,
        "sfia": _open_default_sfia,
        "bls": _open_default_bls,
        "jobtech": _open_default_jobtech,
    }
    missing = [name for name in factories if name != "esco" and not _suite_installed(name)]
    if missing:
        _log.warning(
            "ta-taxonomies has no %s suite; not attached (update the ta-taxonomies pin)",
            ", ".join(missing),
        )
        factories = {name: f for name, f in factories.items() if name not in missing}
    if neighbor_cache:
        factories = {name: with_neighbor_cache(factory) for name, factory in factories.items()}
    return SuiteRegistry(factories, default="esco")
