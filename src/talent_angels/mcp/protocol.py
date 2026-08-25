"""The suite-contract surface this edge needs, expressed structurally.

TA-agents never imports the concrete taxonomy library at module scope — the
offline test suite runs without it installed — so the four contract tools are
described here as protocols and satisfied by any object with the right shape.

The Pathfind and Evaluate halves are declared here rather than added to
``talent_angels.suites.protocol``:

``SuiteTools`` deliberately covers only the Locate and Connect slices,
because ``MeasuredSuite`` (the run-log wrapper the assistant passes around)
implements only those two. Widening it would break that call path, so these two
live here and are checked at runtime instead.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol, TypeGuard, runtime_checkable

from talent_angels.skills.connect.reveal import TaxonomyEdge
from talent_angels.skills.locate.resolve import TaxonomyNode

__all__ = [
    "PathEnumeratingSuite",
    "PathScoringSuite",
    "TaxonomyEdge",
    "TaxonomyNode",
    "supports_enumerate_paths",
    "supports_score_paths",
]


class TaxonomyPath(Protocol):
    @property
    def node_ids(self) -> Sequence[str]: ...

    @property
    def edges(self) -> Sequence[TaxonomyEdge]: ...


class TaxonomyPruning(Protocol):
    @property
    def considered(self) -> int: ...

    @property
    def returned(self) -> int: ...

    @property
    def pruned(self) -> int: ...


class ScoredTaxonomyPath(Protocol):
    @property
    def path(self) -> TaxonomyPath: ...

    @property
    def score(self) -> float: ...

    @property
    def policy(self) -> Any: ...


class PathsResult(Protocol):
    @property
    def nodes(self) -> Sequence[TaxonomyNode]: ...

    @property
    def paths(self) -> Sequence[TaxonomyPath]: ...

    @property
    def pruning(self) -> TaxonomyPruning | None: ...

    @property
    def evidence(self) -> Sequence[str]: ...

    @property
    def warnings(self) -> Sequence[str]: ...


class ScoredPathsResult(Protocol):
    @property
    def scored_paths(self) -> Sequence[ScoredTaxonomyPath]: ...

    @property
    def evidence(self) -> Sequence[str]: ...

    @property
    def warnings(self) -> Sequence[str]: ...


@runtime_checkable
class PathEnumeratingSuite(Protocol):
    """The Pathfind slice of the suite contract."""

    def enumerate_paths(
        self,
        from_id: str,
        to_id: str,
        *,
        max_depth: int = 4,
        max_paths: int = 20,
    ) -> PathsResult: ...


@runtime_checkable
class PathScoringSuite(Protocol):
    """The Evaluate slice of the suite contract.

    ``paths`` is typed loosely because the concrete contract model is not
    imported here: the suite re-validates whatever it receives, and this edge
    hands back exactly the route shape ``enumerate_paths`` produced.
    """

    def score_paths(self, paths: list[Any], policy: Any) -> ScoredPathsResult: ...


def supports_enumerate_paths(suite: object) -> TypeGuard[PathEnumeratingSuite]:
    """Registered suites need only implement Locate and Connect (see module docstring)."""
    return callable(getattr(suite, "enumerate_paths", None))


def supports_score_paths(suite: object) -> TypeGuard[PathScoringSuite]:
    return callable(getattr(suite, "score_paths", None))
