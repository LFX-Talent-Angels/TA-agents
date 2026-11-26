"""The capability registry: one declaration per capability, one `run()`, one manifest.

Every interface (HTTP, MCP, CLI, slash commands) reads its capability list from `CAPABILITIES`.
`run()` is the closed action set: only a registered name (or an alias) runs, only with input
that validates against that capability's model (`extra="forbid"`). There is no free-form tool,
no Cypher and no write path. No capability calls a model, except `tag` without `no_llm`.

Import rule (avoids cycles): only Pydantic models are imported at the top; every skill is
imported inside its capability function.
"""

from __future__ import annotations

import copy
import functools
import logging
import os
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import BaseModel, ValidationError

from talent_angels.engine import outputs as out
from talent_angels.engine.inputs import (
    DiscoverRequest,
    LocateRequest,
    MapRequest,
    NeighborsRequest,
    PathfindRequest,
    RoadmapRequest,
    SuccessorsRequest,
    TagRequest,
)
from talent_angels.mcp.models import NeighborsPayload, SearchPayload
from talent_angels.suites import SuiteRegistry, UnknownSuiteError
from talent_angels.tagging.models import RunSummary

__all__ = [
    "CAPABILITIES",
    "ENGINE_VERSION",
    "STATUSES",
    "Call",
    "CapabilitySpec",
    "JobCancelled",
    "Result",
    "RunContext",
    "get_capability",
    "manifest",
    "run",
]

_log = logging.getLogger(__name__)

#: Bump when a declaration or an input default changes.
ENGINE_VERSION = "1"

Json = dict[str, Any]
Cost = Literal["free", "model"]  # free = graph queries only; model = LLM calls ($)
Latency = Literal["instant", "seconds", "minutes"]
Status = Literal[
    "ok",  # at least one taxonomy answered
    "ambiguous",  # nobody answered; at least one taxonomy has a pick list (see `candidates`)
    "no_result",  # nobody answered: not found / no order data / no route / same ends
    "unavailable",  # every selected taxonomy failed: the suites are down
    "invalid_input",  # validation, or a ValueError from a skill limit
    "unknown_suite",
    "unknown_capability",
    "use_job",  # a long-running capability called synchronously
    "cancelled",  # jobs only
    "error",  # unexpected exception; `error` holds the type name only
]
STATUSES: tuple[str, ...] = Status.__args__  # type: ignore[attr-defined]


class JobCancelled(BaseException):  # noqa: N818
    """Raised from a job's progress hooks. A BaseException, so no `except Exception` in a
    capability swallows it; like Ctrl-C, it stops a tag run between ledger appends."""


@dataclass(frozen=True)
class RunContext:
    """What a long capability may use: progress lines, a step bar, and a cancel flag."""

    progress: Callable[[str], None] = lambda _line: None
    step: Callable[[str, int, int], None] | None = None  # (label, done, total)
    cancel: threading.Event | None = None


class Call(BaseModel):
    """The resolved call: rerunning it on the same graph data gives the same output bytes."""

    capability: str
    input: Json
    engine: str = ENGINE_VERSION


class Result(BaseModel):
    status: Status
    call: Call | None = None
    output: Json | None = None
    warnings: list[str] = []
    candidates: dict[str, list[Json]] = {}
    error: str | None = None
    errors: list[Json] = []


# --- status rules -----------------------------------------------------------------------
_ANSWERED = {"ok", "mapped", "via_crosswalk"}


def _suite_status(output: Json) -> str:
    states = [str(p.get("status", "")) for p in (output.get("suites") or {}).values()]
    if any(s in _ANSWERED for s in states):
        return "ok"
    if "ambiguous" in states:
        return "ambiguous"
    if states and all(s.startswith("error") for s in states):
        return "unavailable"
    return "no_result"


def _discover_status(output: Json) -> str:
    states = [str(p.get("status", "")) for p in (output.get("suites") or {}).values()]
    if states and all(s.startswith("error") for s in states):
        return "unavailable"
    if output.get("summary", {}).get("resolved_items", 0) > 0:
        return "ok"
    if any(i.get("status") == "unclear" for i in output.get("items", [])):
        return "ambiguous"
    return "no_result"


_NO_RESULT = ("not_found", "node_not_found", "no_neighbors", "empty_query", "unknown_kind:")


def _one_suite_status(output: Json) -> str:
    warnings = [str(w) for w in output.get("warnings", [])]
    if any(w.startswith("suite_unavailable:") for w in warnings):
        return "unavailable"
    if "ambiguous" in warnings:
        return "ambiguous"
    if any(w.startswith(_NO_RESULT) for w in warnings):
        return "no_result"
    return "ok"


def _candidates(output: Json) -> dict[str, list[Json]]:
    if "ambiguous" in output.get("warnings", []) and output.get("candidates"):
        return {str(output.get("suite")): list(output["candidates"])}  # locate
    return {
        name: list(part["candidates"])
        for name, part in (output.get("suites") or {}).items()
        if isinstance(part, dict) and part.get("status") == "ambiguous" and part.get("candidates")
    }


# --- the capability functions (bodies moved from api/app.py) -----------------------------
def _locate(i: LocateRequest, reg: SuiteRegistry, ctx: RunContext) -> Json:
    from talent_angels.mcp.mapping import search_payload

    manager = reg.open(i.suite)  # UnknownSuiteError propagates
    try:
        with manager as rt:
            res = rt.suite.search_nodes(i.text, kind=i.kind)
            payload = search_payload(res, suite=rt.name, query=i.text, kind=i.kind)
    except Exception as exc:  # noqa: BLE001 - the same typed warning as the MCP tool
        payload = SearchPayload(
            suite=i.suite or reg.default,
            query=i.text,
            kind=i.kind,
            warnings=[f"suite_unavailable:{type(exc).__name__}"],
        )
    return payload.model_dump(mode="json")


def _connect(i: NeighborsRequest, reg: SuiteRegistry, ctx: RunContext) -> Json:
    from talent_angels.mcp.mapping import neighbors_payload

    manager = reg.open(i.suite)  # UnknownSuiteError propagates
    try:
        with manager as rt:
            res = rt.suite.get_neighbors(i.node_id, rel_types=i.rel_types)
            payload = neighbors_payload(res, suite=rt.name, center_id=i.node_id, limit=i.limit)
    except Exception as exc:  # noqa: BLE001 - the same typed warning as the MCP tool
        payload = NeighborsPayload(
            suite=i.suite or reg.default,
            center_id=i.node_id,
            warnings=[f"suite_unavailable:{type(exc).__name__}"],
        )
    return payload.model_dump(mode="json")


def _pathfind(i: PathfindRequest, reg: SuiteRegistry, ctx: RunContext) -> Json:
    from talent_angels.session.pathfind_view import pathfind_all

    return cast(Json, pathfind_all(reg, i.from_, i.to, suites=i.suites))


def _map(i: MapRequest, reg: SuiteRegistry, ctx: RunContext) -> Json:
    from talent_angels.skills.connect.federated import map_all, map_all_route

    walk: Json = {
        "suites": i.suites,
        "depth": i.depth,
        "breadth": i.breadth or (3 if i.start is None else 4),  # the HTTP/CLI default
        "kinds": i.kinds,
        "max_nodes": i.max_nodes,
    }
    found = (
        map_all(reg, i.start, **walk)
        if i.start
        else map_all_route(reg, i.route_from or "", i.route_to or "", **walk)
    )
    return found.to_dict()


def _discover(i: DiscoverRequest, reg: SuiteRegistry, ctx: RunContext) -> Json:
    from talent_angels.skills.connect.discover import discover
    from talent_angels.skills.connect.discover_io import build_items

    items, notes = build_items(x.model_dump() for x in i.items)
    return discover(
        reg, items, suites=i.suites, depth=i.depth, breadth=i.breadth, kinds=i.kinds,
        max_nodes=i.max_nodes, recorded=i.nodes, picks=i.picks, notes=notes,
    )  # fmt: skip


def _roadmap(i: RoadmapRequest, reg: SuiteRegistry, ctx: RunContext) -> Json:
    from talent_angels.skills.roadmap import roadmap

    return roadmap(reg, i.from_, i.to, suites=i.suites, per_step=i.per_step)


def _successors(i: SuccessorsRequest, reg: SuiteRegistry, ctx: RunContext) -> Json:
    from talent_angels.skills.roadmap import successors

    return successors(
        reg, i.skill, suites=i.suites, depth=i.depth, breadth=i.breadth, max_nodes=i.max_nodes
    )


def _tag(i: TagRequest, reg: SuiteRegistry, ctx: RunContext) -> Json:
    from talent_angels.llm.factory import get_llm_client
    from talent_angels.tagging.cli import parse_rows
    from talent_angels.tagging.models import RunOptions
    from talent_angels.tagging.readers.base import ReaderError
    from talent_angels.tagging.runner import WorkbookOpenError
    from talent_angels.tagging.runner import run as tag_run

    path = Path(i.path)
    if not path.exists():
        raise ValueError(f"no such path: {i.path}")
    client = None
    if not i.no_llm:
        os.environ.setdefault("LLM_TIMEOUT_SECONDS", "60")  # the same as tagging/cli.py
        client = get_llm_client(reasoning_enabled=False)  # a failure is "error", type only
    no_llm = client is None or getattr(client, "provider", "none") == "none"
    opts = RunOptions(
        no_llm=no_llm, limit=i.limit, rows=parse_rows(i.rows) if i.rows else None,
        suites=i.suites, depth=i.depth, refresh=i.refresh, fresh=i.fresh, workers=i.workers,
        profile=i.profile,
    )  # fmt: skip
    try:
        summary = tag_run(
            path, registry=reg, opts=opts, llm_client=None if no_llm else client,
            progress=ctx.progress, step_progress=ctx.step,
        )  # fmt: skip
    except (ReaderError, WorkbookOpenError) as exc:
        raise ValueError(str(exc)) from exc  # the user can fix it
    return summary.model_dump(mode="json")


# --- the declarations -------------------------------------------------------------------
@dataclass(frozen=True)
class CapabilitySpec:
    name: str
    title: str
    question: str
    input_model: type[BaseModel]
    output_model: type[BaseModel]
    fn: Callable[[Any, SuiteRegistry, RunContext], Json]
    deterministic: bool = True  # same input + same graph data -> same bytes
    uses_model: bool = False
    cost: Cost = "free"
    latency: Latency = "seconds"
    long_running: bool = False  # True -> only through a job over HTTP and MCP
    policy: str | None = None
    examples: tuple[Json, ...] = ()
    #: keys: "cli" (subcommand), "slash" (command), "mcp" (tool), "http" (path); absent = none
    aliases: Mapping[str, str] = field(default_factory=dict)
    status_of: Callable[[Json], str] = _suite_status
    notes: str = ""


_ORDER_NOTE = "Evidence, not a syllabus: only stated_prerequisite=true is stated by a taxonomy."

CAPABILITIES: dict[str, CapabilitySpec] = {
    c.name: c
    for c in (
        CapabilitySpec(
            name="locate",
            title="Resolve text to taxonomy nodes",
            question="Which node is this job or skill, in one taxonomy?",
            input_model=LocateRequest,
            output_model=SearchPayload,
            fn=_locate,
            latency="instant",
            examples=({"text": "software developer", "kind": "occupation", "suite": "esco"},),
            aliases={"mcp": "search_nodes"},
            status_of=_one_suite_status,
            notes="`confidence` says how a match was made, not how likely it is to be right.",
        ),
        CapabilitySpec(
            name="connect",
            title="One graph hop around a node",
            question="What is directly linked to this node?",
            input_model=NeighborsRequest,
            output_model=NeighborsPayload,
            fn=_connect,
            latency="instant",
            examples=({"node_id": "esco:occupation:<uuid>", "suite": "esco", "limit": 20},),
            aliases={"mcp": "get_neighbors"},
            status_of=_one_suite_status,
            notes="IDs are suite-scoped: take node_id from locate in the same suite.",
        ),
        CapabilitySpec(
            name="pathfind",
            title="Skill gap and routes between two occupations",
            question="What separates job A from job B, and which routes connect them?",
            input_model=PathfindRequest,
            output_model=out.PathfindOutput,
            fn=_pathfind,
            policy="route-rank-v1",
            examples=({"from": "nurse", "to": "data analyst", "suites": ["esco"]},),
            aliases={"mcp": "pathfind", "http": "/v1/capabilities/pathfind"},
            notes="Graph data, not career advice. BLS has no skill gap.",
        ),
        CapabilitySpec(
            name="map",
            title="Map the neighbourhood of a skill or job",
            question="What surrounds this skill or job (or the route between two), in every "
            "taxonomy?",
            input_model=MapRequest,
            output_model=out.MapOutput,
            fn=_map,
            policy="map-rank-v1",
            examples=(
                {"start": "Python", "depth": 2, "breadth": 4, "suites": ["esco", "onet"]},
                {"route_from": "web developer", "route_to": "software developer"},
            ),
            aliases={
                "cli": "map",
                "slash": "map",
                "mcp": "map_neighbors",
                "http": "/v1/capabilities/map",
            },
            notes="Give either start, or both route_from and route_to.",
        ),
        CapabilitySpec(
            name="discover",
            title="Map how a list of roles, tasks and skills connects",
            question="How do the items on my list connect to each other?",
            input_model=DiscoverRequest,
            output_model=out.DiscoverOutput,
            fn=_discover,
            latency="minutes",
            policy="discover-v1",
            examples=(
                {
                    "items": [
                        {"item": "web developer", "kind": "role", "related_to": ["JavaScript"]},
                        {"item": "data analyst"},
                    ],
                    "depth": 2,
                },
            ),
            aliases={
                "cli": "discover",
                "slash": "discover",
                "mcp": "discover_map",
                "http": "/v1/capabilities/discover",
            },
            status_of=_discover_status,
            notes="Pass a previous result's `resolved` block as `nodes` to skip resolution.",
        ),
        CapabilitySpec(
            name="roadmap",
            title="Learning order from one occupation to another",
            question="What do I learn, in order, to go from job A to job B?",
            input_model=RoadmapRequest,
            output_model=out.RoadmapOutput,
            fn=_roadmap,
            policy="learn-order-v1",
            examples=({"from": "data analyst", "to": "data scientist", "per_step": 5},),
            aliases={
                "cli": "roadmap",
                "slash": "roadmap",
                "mcp": "roadmap",
                "http": "/v1/capabilities/roadmap",
            },
            notes=_ORDER_NOTE,
        ),
        CapabilitySpec(
            name="successors",
            title="What comes after a skill",
            question="What comes after this skill?",
            input_model=SuccessorsRequest,
            output_model=out.SuccessorsOutput,
            fn=_successors,
            policy="learn-order-v1",
            examples=({"skill": "Python (computer programming)", "depth": 2, "breadth": 4},),
            aliases={
                "cli": "successors",
                "slash": "successors",
                "mcp": "successors",
                "http": "/v1/capabilities/successors",
            },
            notes="Evidence, not a syllabus: only `stated` edges are stated by a taxonomy.",
        ),
        CapabilitySpec(
            name="tag",
            title="Batch-tag documents against the taxonomies",
            question="Which taxonomy nodes match each row of these documents?",
            input_model=TagRequest,
            output_model=RunSummary,
            fn=_tag,
            deterministic=False,  # deterministic with no_llm (the default here)
            uses_model=True,  # unless no_llm
            cost="model",
            latency="minutes",
            long_running=True,
            examples=({"path": "uca-pilot", "no_llm": True, "limit": 20},),
            aliases={"cli": "tag", "slash": "tag"},
            status_of=lambda _output: "ok",
            notes="A job over HTTP and MCP; `path` is relative to the server's TA_TAG_ROOT. "
            "`no_llm` is the default here (the CLI's default is the model): with no_llm it is "
            "deterministic and free. A model run must be allowed on the server "
            "(TA_JOBS_ALLOW_MODEL=1): about one call per unit x taxonomy plus rewrites, "
            "`model_calls` and `cost_usd` reported at the end; no pre-flight estimate and no "
            "budget cap, so try `limit` first. `workers` is capped at 8.",
        ),
    )
}

# MCP tool names are accepted as capability names; CLI, slash and HTTP aliases are not names.
_ALIAS_INDEX: dict[str, str] = {
    c.aliases["mcp"]: c.name for c in CAPABILITIES.values() if "mcp" in c.aliases
}


def get_capability(name: str) -> CapabilitySpec:
    """A canonical name or an MCP alias ("map_neighbors", "search_nodes"). Raises KeyError."""
    return CAPABILITIES[name if name in CAPABILITIES else _ALIAS_INDEX[name]]


# --- one entry point --------------------------------------------------------------------
def prepare(capability: str, payload: Mapping[str, Any] | BaseModel) -> tuple[
    CapabilitySpec, BaseModel
] | Result:  # fmt: skip
    """Look the capability up and validate its input, or the Result that refuses it."""
    try:
        cap = get_capability(capability)
    except KeyError:
        return Result(
            status="unknown_capability",
            error=f"unknown capability {capability[:64]!r}; available: {', '.join(CAPABILITIES)}",
        )
    if isinstance(payload, cap.input_model):
        return cap, payload
    try:
        return cap, cap.input_model.model_validate(payload)
    except ValidationError as exc:
        errors = exc.errors(include_url=False, include_context=False, include_input=False)
        lines = (f"{'.'.join(map(str, e['loc'])) or 'input'}: {e['msg']}" for e in errors)
        return Result(
            status="invalid_input", errors=[dict(e) for e in errors], error="; ".join(lines)
        )


def run(
    capability: str,
    payload: Mapping[str, Any] | BaseModel,
    *,
    registry: SuiteRegistry,
    ctx: RunContext | None = None,
    allow_long: bool = False,
) -> Result:
    """Run one capability. Never raises for bad input or a failing skill: the status says it."""
    prepared = prepare(capability, payload)
    if isinstance(prepared, Result):
        return prepared
    cap, inp = prepared
    if cap.long_running and not allow_long:
        return Result(status="use_job", error=f"{cap.name} is long-running: start it as a job")
    call = Call(capability=cap.name, input=inp.model_dump(mode="json", by_alias=True))
    try:
        output = cap.fn(inp, registry, ctx or RunContext())
    except JobCancelled:
        return Result(status="cancelled", call=call)
    except UnknownSuiteError as exc:  # before ValueError: it is one
        return Result(status="unknown_suite", call=call, error=str(exc))
    except ValueError as exc:
        return Result(status="invalid_input", call=call, error=str(exc))
    except Exception as exc:  # noqa: BLE001 - one line, the type only: a message may hold a key
        _log.warning("capability %s failed", cap.name, exc_info=True)
        return Result(status="error", call=call, error=type(exc).__name__)
    return Result(
        status=cap.status_of(output),  # type: ignore[arg-type]
        call=call,
        output=output,
        warnings=[str(w) for w in output.get("warnings") or []],
        candidates=_candidates(output),
    )


# --- the manifest -----------------------------------------------------------------------
def _entry(c: CapabilitySpec) -> Json:
    return {
        "name": c.name,
        "title": c.title,
        "question": c.question,
        "deterministic": c.deterministic,
        "uses_model": c.uses_model,
        "cost": c.cost,
        "latency": c.latency,
        "long_running": c.long_running,
        "policy": c.policy,
        "notes": c.notes,
        "aliases": dict(c.aliases),
        "examples": list(c.examples),
        "input_schema": c.input_model.model_json_schema(by_alias=True),
        "output_schema": c.output_model.model_json_schema(by_alias=True),
    }


@functools.cache
def _manifest() -> Json:
    return {
        "engine": "talent-angels",
        "version": ENGINE_VERSION,
        "run": {"http": "POST /v1/run", "mcp": "run", "cli": "ta-agent run <capability> '<json>'"},
        "jobs": {
            "http": "POST /v1/jobs, GET /v1/jobs/{id}, POST /v1/jobs/{id}/cancel",
            "mcp": "start_job, job_status, cancel_job",
        },
        "statuses": list(STATUSES),
        "conversational": {
            "http": [
                "POST /v1/query",
                "POST /v1/capabilities/locate",
                "POST /v1/capabilities/connect",
            ],
            "note": "Chat turns over locate/connect/pathfind; may call a model to phrase.",
        },
        "capabilities": [_entry(c) for c in CAPABILITIES.values()],
    }


def manifest(only: str | None = None) -> Json:
    """What this engine can do. `only` (a name or MCP alias) keeps one entry; unknown -> none."""
    full = copy.deepcopy(_manifest())
    if only is not None:
        name = _ALIAS_INDEX.get(only, only)
        full["capabilities"] = [c for c in full["capabilities"] if c["name"] == name]
    return full
