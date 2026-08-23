"""Structural taxonomy tool surface currently consumed by TA-agents."""

from __future__ import annotations

from typing import Protocol

from talent_angels.skills.connect import ConnectableSuite
from talent_angels.skills.locate.resolve import SearchableSuite


class SuiteTools(SearchableSuite, ConnectableSuite, Protocol):
    """Implemented Locate and Connect slices of the taxonomy suite contract."""
