"""Areas: how a broad search splits into occupation groups, and narrowing to one.

"engineer" matches hundreds of titles; the first page of a pick list says
little about them. The suite counts its title matches per occupation group
(ISCO unit group on ESCO, SOC major group on O*NET), so the assistant can ask
"which area?" with real category names. Choosing one re-runs the same search
inside that group only. Counts and names are graph data; nothing here guesses.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict

from talent_angels.contracts import AgentResult
from talent_angels.skills.locate.resolve import SearchableSuite, SearchResult, locate

#: Areas offered per suite, largest first.
AREAS_SHOWN = 8
#: An area worth offering holds at least this many matching titles, and at
#: least two such areas must exist: "Health care assistants (1 title)" five
#: times over is a list of titles, not a choice of areas.
MIN_AREA_TITLES = 3


class Area(BaseModel):
    """One occupation group a search matched, with how many titles fall in it."""

    model_config = ConfigDict(frozen=True)

    suite: str
    code: str
    label: str
    count: int


class AreaRequest(BaseModel):
    """Re-run ``query`` inside one area of one suite."""

    model_config = ConfigDict(frozen=True)

    suite: str
    code: str
    query: str
    kind: str | None = "occupation"


def areas_from(raw: object, suite_name: str) -> list[Area]:
    """The areas a suite reported for one search, or [] when it reported none."""
    meta: Any = getattr(raw, "meta", None) or {}
    groups = meta.get("groups") if isinstance(meta, dict) else None
    if not groups:
        return []
    areas: list[Area] = []
    for group in groups[:AREAS_SHOWN]:
        code = str(group.get("code") or "").strip()
        if not code or int(group.get("count") or 0) < MIN_AREA_TITLES:
            continue
        areas.append(
            Area(
                suite=suite_name,
                code=code,
                label=str(group.get("label") or code),
                count=int(group.get("count") or 0),
            )
        )
    return areas if len(areas) >= 2 else []


def area_summary(
    suite: SearchableSuite, suite_name: str, query: str, *, kind: str | None
) -> list[Area]:
    """Search once more and read the suite's group counts. [] if unsupported."""
    return areas_from(suite.search_nodes(query, kind=kind), suite_name)


class _GroupScoped:
    """A suite whose search only looks inside one group (for ``locate``)."""

    def __init__(self, search_group: Any, code: str) -> None:
        self._search_group = search_group
        self._code = code

    def search_nodes(self, text: str, kind: str | None = None) -> SearchResult:
        del kind  # a group holds occupations only
        result: SearchResult = self._search_group(text, self._code)
        return result


def search_area(suite: object, suite_name: str, request: AreaRequest) -> AgentResult:
    """The titles matching ``request.query`` inside one area, as a Locate result.

    A suite without group search answers ``area_not_supported`` rather than
    falling back to the whole map, which would undo the user's choice.
    """
    search_group = getattr(suite, "search_group", None)
    if search_group is None:
        return AgentResult(capability="locate", suite=suite_name, warnings=["area_not_supported"])
    try:
        result = locate(_GroupScoped(search_group, request.code), suite_name, request.query)
    except AttributeError:  # a wrapper whose inner suite has no group search
        return AgentResult(capability="locate", suite=suite_name, warnings=["area_not_supported"])
    if result.nodes and "ambiguous" not in result.warnings:
        # One title in the area is still the user's to confirm, never a bind.
        result = result.model_copy(update={"warnings": [*result.warnings, "ambiguous"]})
    return result
