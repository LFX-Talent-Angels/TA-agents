"""SuiteSchema — per-suite relation type and node kind metadata.

The canonical definition lives in ta_taxonomies.contract.schema so suite
implementations (EscoSuite, OnetSuite, ...) can reference it without a
circular import. This module re-exports it when ta_taxonomies is installed,
and provides an identical fallback definition when it is not (e.g. in the
API/CLI smoke test that verifies TA-agents imports without the concrete
taxonomy package).
"""

from __future__ import annotations

try:
    from ta_taxonomies.contract.schema import SuiteSchema as SuiteSchema
except ImportError:
    from dataclasses import dataclass, field

    @dataclass(frozen=True)  # type: ignore[no-redef]
    class SuiteSchema:  # type: ignore[no-redef]
        """Declares what relation types and property values a suite uses per concept."""

        skill_rel_types: tuple[str, ...]
        optional_rel_values: frozenset[str] = field(default_factory=frozenset)
        group_rel_type: str | None = None
        group_node_kinds: frozenset[str] = field(default_factory=frozenset)


__all__ = ["SuiteSchema"]
