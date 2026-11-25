"""Input models for the engine's capabilities: one Pydantic model per capability.

Every model forbids extra fields, so a caller (a UI, an MCP client, a model filling arguments)
cannot slip in a key the capability does not declare. The five deterministic request models
moved here verbatim from `api/schemas.py`, which re-exports them under the same names.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from talent_angels.skills.connect.discover import DEFAULT_DISCOVER_MAX_NODES, MAX_DISCOVER_NODES
from talent_angels.skills.connect.discover_io import MAX_ITEMS
from talent_angels.skills.connect.neighbourhood import DEFAULT_MAX_NODES, MAX_BREADTH, MAX_DEPTH
from talent_angels.skills.roadmap import DEFAULT_MAX_NODES as SUCCESSORS_MAX_NODES
from talent_angels.skills.roadmap import MAX_PER_STEP

#: The same bound the runtime applies (assistant/turn.py), so a name the API would cut is refused.
MAX_QUESTION_CHARS = 1000
MAX_TAG_WORKERS = 8

Name = Field(min_length=1, max_length=MAX_QUESTION_CHARS)
Suites = Field(default=None, description="Subset of taxonomies; omit for all.")
Depth = Field(default=2, ge=1, le=MAX_DEPTH)
Breadth = Field(default=4, ge=1, le=MAX_BREADTH)


class _Request(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class LocateRequest(_Request):
    text: str = Name
    kind: str | None = Field(default=None, max_length=64)
    suite: str | None = Field(default=None, description="One taxonomy; omit for the default.")


class NeighborsRequest(_Request):
    node_id: str = Name
    rel_types: list[str] | None = None
    suite: str | None = Field(default=None, description="One taxonomy; omit for the default.")
    limit: int = Field(default=50, ge=1, le=500)


class PathfindRequest(_Request):
    from_: str = Field(alias="from", min_length=1, max_length=MAX_QUESTION_CHARS)
    to: str = Name
    suites: list[str] | None = Suites


class RoadmapRequest(PathfindRequest):
    per_step: int = Field(default=10, ge=1, le=MAX_PER_STEP)


class SuccessorsRequest(_Request):
    skill: str = Name
    suites: list[str] | None = Suites
    depth: int = Depth
    breadth: int = Breadth
    max_nodes: int = Field(default=SUCCESSORS_MAX_NODES, ge=1, le=DEFAULT_MAX_NODES)


class MapRequest(_Request):
    start: str | None = Field(default=None, max_length=MAX_QUESTION_CHARS)
    route_from: str | None = Field(default=None, max_length=MAX_QUESTION_CHARS)
    route_to: str | None = Field(default=None, max_length=MAX_QUESTION_CHARS)
    suites: list[str] | None = Suites
    depth: int = Field(default=1, ge=1, le=MAX_DEPTH)
    breadth: int | None = Field(
        default=None, ge=1, le=MAX_BREADTH, description="Default 4 (3 on a route)."
    )
    kinds: list[str] | None = None
    max_nodes: int = Field(default=DEFAULT_MAX_NODES, ge=1, le=DEFAULT_MAX_NODES)

    @model_validator(mode="after")
    def _one_mode(self) -> MapRequest:
        route = [self.route_from, self.route_to]
        if bool(self.start) == all(route) or (any(route) and not all(route)):
            raise ValueError("give either start, or both route_from and route_to")
        return self


class DiscoverItem(_Request):
    item: str = Name
    kind: str | None = None
    related_to: list[str] = []


class DiscoverRequest(_Request):
    items: list[DiscoverItem] = Field(min_length=1, max_length=MAX_ITEMS)
    suites: list[str] | None = Suites
    depth: int = Depth
    breadth: int = Breadth
    kinds: list[str] | None = None
    max_nodes: int = Field(default=DEFAULT_DISCOVER_MAX_NODES, ge=1, le=MAX_DISCOVER_NODES)
    nodes: list[dict[str, Any]] | None = Field(
        default=None, description="A previous result's `resolved` block: skips resolution."
    )


class TagRequest(_Request):
    path: str = Field(
        min_length=1,
        max_length=4096,
        description="A project folder (with inputs/) or one input file. As a job: relative to "
        "the server's tag root (TA_TAG_ROOT).",
    )
    no_llm: bool = Field(default=True, description="Search only; no model call (the default).")
    limit: int | None = Field(default=None, ge=1, description="Tag at most N units.")
    rows: str | None = Field(
        default=None, max_length=1000, pattern=r"^[\w.,\- ]+$", description='e.g. "66-91,100".'
    )
    suites: list[str] | None = Suites
    depth: Literal["leaves", "headings", "all"] | None = None
    profile: str | None = Field(default=None, max_length=255)
    refresh: bool = Field(default=False, description="Re-run non-final verdicts.")
    fresh: bool = Field(default=False, description="Ignore the ledger entirely.")
    workers: int = Field(default=1, ge=1, le=MAX_TAG_WORKERS)

    @model_validator(mode="after")
    def _refresh_or_fresh(self) -> TagRequest:
        if self.refresh and self.fresh:
            raise ValueError("give refresh or fresh, not both")
        return self
