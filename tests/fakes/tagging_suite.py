"""A table-driven fake suite for the tagging pipeline's offline tests.

Answers are keyed by (query casefold, kind); a missing key means "not found".
Shaped like `ta_taxonomies`'s `ToolResult`/`Candidate`/`Node` (simple
dataclasses, see `tests/fakes/taxonomy.py`) so `search.py`/`runner.py` never
know this isn't the real suite.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field

from talent_angels.suites import SuiteFactory, SuiteRegistry, SuiteRuntime
from talent_angels.suites.schema import SuiteSchema

_SCHEMA = SuiteSchema(
    skill_rel_types=("HAS_SKILL",),
    optional_rel_values=frozenset(),
    group_rel_type=None,
    group_node_kinds=frozenset(),
)


@dataclass
class FakeNode:
    id: str
    kind: str
    label: str
    source: str = "fake"
    source_id: str = ""
    properties: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.source_id:
            self.source_id = f"urn:{self.id}"


@dataclass
class FakeCandidate:
    node: FakeNode
    confidence: float
    method: str


@dataclass
class FakeToolResult:
    candidates: list[FakeCandidate] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class Answer:
    """One scripted answer for a (query, kind) pair."""

    hits: list[tuple[str, str, str, str]] = field(default_factory=list)  # id, kind, label, method
    warnings: list[str] = field(default_factory=list)


class TableSuite:
    """`search_nodes` driven entirely by a `{(query, kind): Answer}` table."""

    def __init__(
        self,
        name: str,
        table: dict[tuple[str, str | None], Answer],
        *,
        unsupported_kinds: frozenset[str] = frozenset(),
        raise_on: frozenset[str] = frozenset(),
        raise_as: type[Exception] = RuntimeError,
    ) -> None:
        self.name = name
        self._table = table
        self._unsupported_kinds = unsupported_kinds
        self._raise_on = raise_on
        self._raise_as = raise_as
        self.calls: list[tuple[str, str | None]] = []

    @property
    def suite_schema(self) -> SuiteSchema:
        return _SCHEMA

    def search_nodes(self, text: str, kind: str | None = None) -> FakeToolResult:
        self.calls.append((text, kind))
        if text in self._raise_on:
            raise self._raise_as(f"{self.name}: boom on {text!r}")
        if kind is not None and kind in self._unsupported_kinds:
            return FakeToolResult(warnings=[f"unknown_kind:{kind}"])
        answer = self._table.get((text.casefold(), kind))
        if answer is None:
            return FakeToolResult(warnings=["not_found"])
        candidates = [
            FakeCandidate(
                node=FakeNode(id=hit_id, kind=hit_kind, label=label), confidence=0.9, method=method
            )
            for hit_id, hit_kind, label, method in answer.hits
        ]
        return FakeToolResult(candidates=candidates, warnings=list(answer.warnings))

    def get_neighbors(self, node_id: str, rel_types: list[str] | None = None) -> FakeToolResult:
        return FakeToolResult(warnings=["node_not_found"])


def suite_factory(suite: TableSuite, *, reachable: bool = True) -> SuiteFactory:
    @contextmanager
    def _open() -> Iterator[SuiteRuntime]:
        yield SuiteRuntime(name=suite.name, suite=suite, health_check=lambda: reachable)  # type: ignore[arg-type]

    return _open


def unopenable_factory(error: Exception) -> SuiteFactory:
    @contextmanager
    def _open() -> Iterator[SuiteRuntime]:
        raise error
        yield  # pragma: no cover - unreachable, keeps this a generator

    return _open


def table_registry(
    suites: dict[str, TableSuite],
    *,
    default: str,
    broken: dict[str, Exception] | None = None,
) -> SuiteRegistry:
    """Build a `SuiteRegistry` over `TableSuite`s, plus optional unopenable ones."""
    factories: dict[str, SuiteFactory] = {
        name: suite_factory(suite) for name, suite in suites.items()
    }
    for name, error in (broken or {}).items():
        factories[name] = unopenable_factory(error)
    return SuiteRegistry(factories, default=default)
