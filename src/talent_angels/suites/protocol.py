"""Structural taxonomy tool surface currently consumed by TA-agents."""

from __future__ import annotations

from typing import Protocol

from talent_angels.skills.connect import ConnectableSuite
from talent_angels.skills.locate.resolve import SearchableSuite
from talent_angels.suites.schema import SuiteSchema


class SuiteTools(SearchableSuite, ConnectableSuite, Protocol):
    """Implemented Locate and Connect slices of the taxonomy suite contract."""

    @property
    def suite_schema(self) -> SuiteSchema:
        """Suite-specific relation types, optional-value aliases, and group kinds.

        Every suite implementation in TA-taxonomies must return one SuiteSchema.
        TA-agents callers read from it instead of hardcoding taxonomy strings.
        """
        ...
