"""MCP server exposing the taxonomy suite contract over stdio.

Why this exists: Claude Code (or any MCP client) can walk the graph itself,
on the user's own subscription, without this project spending a token. The
client takes the role the main assistant takes inside this repo — it owns the
goal and writes the prose. We only assert graph facts.

So the tool surface is the suite contract, one for one — ``search_nodes``,
``get_neighbors``, ``enumerate_paths``, ``score_paths`` — and not the Locate /
Connect / Pathfind / Evaluate skills. A skill is a procedure *for* our
assistant; the contract is the tool surface, and a foreign client should get
the tools, not our procedures (ARCHITECTURE.md: "tools know nothing about
agents").

Everything a model needs to use these correctly lives in the tool descriptions
below — they are the only thing a client reads before deciding to call. Two
points are load-bearing and repeated deliberately: IDs are suite-scoped, and
confidence says *how* a match was made, not how likely it is to be right.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import Annotated, Any

from mcp.server import MCPServer
from mcp.server.mcpserver import Context
from pydantic import Field

from talent_angels.env import load_local_dotenv
from talent_angels.mcp.mapping import (
    neighbors_payload,
    paths_payload,
    scored_paths_payload,
    search_payload,
    suite_path_arg,
)
from talent_angels.mcp.models import (
    DiscoverPayload,
    MapPayload,
    NeighborsPayload,
    OrderPayload,
    PathfindPayload,
    PathRef,
    PathsPayload,
    PolicyRef,
    ScoredPathsPayload,
    SearchPayload,
)
from talent_angels.mcp.protocol import supports_enumerate_paths, supports_score_paths
from talent_angels.mcp.session import SuiteSession, SuiteUnavailableError
from talent_angels.skills.connect.discover import (
    DEFAULT_DISCOVER_MAX_NODES,
    MAX_DISCOVER_NODES,
    discover,
)
from talent_angels.skills.connect.discover_io import MAX_ITEMS, build_items
from talent_angels.skills.connect.federated import map_all, map_all_route
from talent_angels.skills.connect.neighbourhood import (
    DEFAULT_MAX_NODES,
    MAX_BREADTH,
    MAX_DEPTH,
    map_from_text,
    map_route,
)
from talent_angels.skills.roadmap import DEFAULT_MAX_NODES as SUCCESSORS_MAX_NODES
from talent_angels.skills.roadmap import MAX_PER_STEP, roadmap, successors
from talent_angels.suites import (
    SuiteRegistry,
    SuiteRuntime,
    UnknownSuiteError,
    default_suite_registry,
)

__all__ = ["SERVER_NAME", "SERVER_VERSION", "build_server", "main"]

SERVER_NAME = "ta-taxonomy"
SERVER_VERSION = "0.1.0"

# `Context` is injected by the SDK and stays out of the tool's input schema.
ToolContext = Context[SuiteSession, Any]

SERVER_INSTRUCTIONS = """\
Read-only access to the Talent Angels taxonomy graphs (ESCO, O*NET, SFIA, BLS
and JobTech, as far as they are registered and loaded) through the suite
contract: search_nodes, get_neighbors, enumerate_paths, score_paths, plus
map_neighbors (a multi-hop get_neighbors), discover_map (how a list of items
connects), roadmap (what to learn from A to B, in order), successors (what comes
after a skill) and pathfind (the skill gap and ranked routes between two occupations).
`capabilities` lists everything the engine can do (name, question, input and output JSON Schema,
cost, determinism); `run` runs any one of them by name; `tag` (batch-tag documents) is long, so
use `start_job`, `job_status` and `cancel_job`. Pass the
suite by name, e.g. suite=`esco`, `onet`, `sfia`, `bls` or `jobtech`.
Do not treat IDs from one suite as belonging to another.

Two rules govern everything these tools return.

1. IDs are suite-scoped. `esco:occupation:<uuid>` means something only inside
   the ESCO suite. Never equate an ID, a label or a code across suites, and
   never invent a crosswalk between them; where two suites are involved and no
   explicit crosswalk exists, the honest answer is that there is no link.

2. `confidence` reports how a match was made, not how likely it is to be
   right. It is one fixed value per match method, not a probability. Do not
   average, multiply, or compare these numbers across suites.

What these tools return is graph data and may be cited as taxonomy fact.
Anything you add on top of it is your own inference — label it as such.

`warnings` is machine-readable and never decorative. `not_found`, `ambiguous`,
`no_path`, `node_not_found` and `no_neighbors` all mean the graph did not
answer the question; report that rather than filling the gap.
"""

_CONFIDENCE_DOC = """\
`confidence` describes HOW the match was made, not the probability that the
node is the one meant. Each match method carries one fixed value — for ESCO:

  0.95  exact_pref               exact match on the preferred label
  0.90  exact_alt                exact match on an alternative label
  0.85  casefold_pref            case-insensitive preferred label, unique hit
  0.80  casefold_pref_ambiguous  case-insensitive preferred label, several hits
  0.70  contains                 substring match only

0.70 does not mean "70% sure". It means the only thing that matched was a
substring — exactly the case where a human should be asked to confirm. The
`method` field on every candidate names the rule that produced the number.\
"""

_ID_DOC = """\
IDs are suite-scoped, e.g. `esco:occupation:6a12b8f0-…`. They are valid only
inside the suite that issued them; never reuse one against another suite.\
"""


def _make_lifespan(
    registry: SuiteRegistry,
) -> Callable[[MCPServer[SuiteSession]], AbstractAsyncContextManager[SuiteSession]]:
    """One suite session per server process; suites open on first use."""

    @asynccontextmanager
    async def lifespan(server: MCPServer[SuiteSession]) -> AsyncIterator[SuiteSession]:
        # The client launches us as a bare subprocess, so nothing has sourced
        # `.env` — the Neo4j credentials come from there, as for the CLI.
        load_local_dotenv()
        with SuiteSession(registry) as session:
            yield session

    return lifespan


def _open_runtime(
    session: SuiteSession, suite: str | None
) -> tuple[SuiteRuntime | None, str, list[str]]:
    """Resolve a suite name to an open runtime, or to typed warnings.

    Failure never travels as prose: an unregistered name and an unreachable
    database are two different machine-readable warnings on an otherwise empty
    payload, so the client can tell a user mistake from an operator one.
    """
    requested = suite or session.registry.default
    try:
        return session.runtime(suite), requested, []
    except UnknownSuiteError:
        available = ",".join(session.registry.available)
        return None, requested, [f"unknown_suite:{requested}", f"available_suites:{available}"]
    except SuiteUnavailableError as exc:
        return None, requested, [f"suite_unavailable:{exc.reason}"]


def _operation_warning(exc: Exception) -> list[str]:
    """Keep a suite-call failure typed without leaking its details to the client."""
    return [f"suite_unavailable:{type(exc).__name__}"]


SuiteArg = Annotated[
    str | None,
    Field(
        description=(
            "Taxonomy suite to query. Defaults to the registry default (`esco`). "
            "Results are only ever scoped to this one suite."
        )
    ),
]


def build_server(*, registry: SuiteRegistry | None = None) -> MCPServer[SuiteSession]:
    """Wire the four contract tools and map_neighbors onto an MCP server."""
    server: MCPServer[SuiteSession] = MCPServer(
        name=SERVER_NAME,
        title="Talent Angels — taxonomy graph",
        version=SERVER_VERSION,
        instructions=SERVER_INSTRUCTIONS,
        lifespan=_make_lifespan(registry or default_suite_registry()),
    )

    @server.tool(
        name="search_nodes",
        title="Resolve text to taxonomy nodes",
        description=f"""\
Resolve free text (an occupation, a skill, a group name) to candidate nodes in
one taxonomy suite. This is the entry point: every other tool takes IDs, and
IDs come from here.

Matching is deterministic and tiered — exact preferred label, then exact
alternative label, then case-insensitive, then substring — stopping at the
first tier that hits. Nothing is invented: text with no match comes back with
`warnings: ["not_found"]` and no candidates.

{_CONFIDENCE_DOC}

{_ID_DOC}

`warnings` may contain `ambiguous` (several equally good hits — ask the user
rather than picking one), `not_found`, `empty_query`, or `unknown_kind:<kind>`.""",
    )
    def search_nodes(
        ctx: ToolContext,
        text: Annotated[
            str,
            Field(description="Free text to resolve, e.g. 'software developer' or 'Python'."),
        ],
        kind: Annotated[
            str | None,
            Field(
                description=(
                    "Optional node-kind filter. ESCO accepts occupation, skill, "
                    "isco_group, skill_group (singular or plural). An unrecognised "
                    "kind returns `unknown_kind:<kind>` rather than a guess."
                )
            ),
        ] = None,
        suite: SuiteArg = None,
    ) -> SearchPayload:
        session = ctx.request_context.lifespan_context
        runtime, name, warnings = _open_runtime(session, suite)
        if runtime is None:
            return SearchPayload(suite=name, query=text, kind=kind, warnings=warnings)
        try:
            result = runtime.suite.search_nodes(text, kind=kind)
        except Exception as exc:
            return SearchPayload(
                suite=runtime.name,
                query=text,
                kind=kind,
                warnings=_operation_warning(exc),
            )
        return search_payload(result, suite=runtime.name, query=text, kind=kind)

    @server.tool(
        name="get_neighbors",
        title="One graph hop around a node",
        description=f"""\
Return the direct graph neighbours of one resolved node, with the typed edges
connecting them. Exactly one hop — no transitive expansion, no ranking.

Use it for "what skills does this occupation require", "what is this skill's
parent group", "what sits alongside this". Call `search_nodes` first to obtain
`node_id`.

Edge `properties` carry the qualifiers that change what an answer means: ESCO
`HAS_SKILL` edges are marked `relation_type: essential` or `optional`, and that
distinction is source data — preserve it instead of flattening the list.

ESCO traversable relationship types: HAS_SKILL, BROADER_THAN,
CLASSIFIED_UNDER, RELATED_TO.

{_ID_DOC}

`warnings` may contain `node_not_found` (the ID does not exist in this suite —
do not retry it against another one), `no_neighbors`, or
`unknown_rel_types:[…]`.""",
    )
    def get_neighbors(
        ctx: ToolContext,
        node_id: Annotated[
            str,
            Field(description="Suite-scoped node ID from `search_nodes`."),
        ],
        rel_types: Annotated[
            list[str] | None,
            Field(
                description=(
                    "Optional filter of relationship types. Omit for every type the "
                    "suite marks traversable. An unknown type is rejected, not ignored."
                )
            ),
        ] = None,
        suite: SuiteArg = None,
        limit: Annotated[
            int,
            Field(
                ge=1,
                le=500,
                description=(
                    "Most neighbours to return, ranked by edge importance (O*NET) then "
                    "hot/in-demand technology. What is cut is counted in `pruning`."
                ),
            ),
        ] = 50,
    ) -> NeighborsPayload:
        session = ctx.request_context.lifespan_context
        runtime, name, warnings = _open_runtime(session, suite)
        if runtime is None:
            return NeighborsPayload(suite=name, center_id=node_id, warnings=warnings)
        try:
            result = runtime.suite.get_neighbors(node_id, rel_types=rel_types)
        except Exception as exc:
            return NeighborsPayload(
                suite=runtime.name,
                center_id=node_id,
                warnings=_operation_warning(exc),
            )
        return neighbors_payload(result, suite=runtime.name, center_id=node_id, limit=limit)

    @server.tool(
        name="enumerate_paths",
        title="Bounded routes between two nodes",
        description=f"""\
Enumerate routes between two resolved nodes in the same suite: depth-capped,
cycle-free, deterministic. Both IDs come from `search_nodes`.

The walk happens in the database, not in your context. Caps are enforced there
and what they cut is reported as counts in `pruning` (considered / returned /
pruned), never as discarded rows. A non-zero `pruned` means the answer is a
sample of the routes, not all of them; say so rather than implying coverage.

Route order is enumeration order, not quality order. `enumerate_paths` does not
rank; `score_paths` does, under a named policy.

{_ID_DOC} Both endpoints must belong to the same suite: there is no cross-suite
path.

`warnings` may contain `endpoint_not_found`, `no_path` (the graph holds no
route within the cap — a real answer, not a failure), `invalid_max_depth`, or
`invalid_max_paths`.""",
    )
    def enumerate_paths(
        ctx: ToolContext,
        from_id: Annotated[str, Field(description="Suite-scoped ID of the start node.")],
        to_id: Annotated[str, Field(description="Suite-scoped ID of the end node.")],
        max_depth: Annotated[
            int,
            Field(
                description=(
                    "Maximum hops per route. ESCO allows up to 6; a larger value comes "
                    "back as `invalid_max_depth`; values below 1 do too."
                ),
            ),
        ] = 4,
        max_paths: Annotated[
            int,
            Field(
                description=(
                    "Maximum routes returned. ESCO allows up to 100; a larger value comes "
                    "back as `invalid_max_paths`; values below 1 do too."
                ),
            ),
        ] = 20,
        suite: SuiteArg = None,
    ) -> PathsPayload:
        session = ctx.request_context.lifespan_context
        runtime, name, warnings = _open_runtime(session, suite)
        if runtime is None:
            return PathsPayload(suite=name, from_id=from_id, to_id=to_id, warnings=warnings)
        if not supports_enumerate_paths(runtime.suite):
            return PathsPayload(
                suite=runtime.name,
                from_id=from_id,
                to_id=to_id,
                warnings=[f"capability_unavailable:enumerate_paths:{runtime.name}"],
            )
        try:
            result = runtime.suite.enumerate_paths(
                from_id, to_id, max_depth=max_depth, max_paths=max_paths
            )
        except Exception as exc:
            return PathsPayload(
                suite=runtime.name,
                from_id=from_id,
                to_id=to_id,
                warnings=_operation_warning(exc),
            )
        return paths_payload(result, suite=runtime.name, from_id=from_id, to_id=to_id)

    @server.tool(
        name="score_paths",
        title="Rank routes under a named policy",
        description=f"""\
Rank routes produced by `enumerate_paths` under an explicitly named, versioned
scoring policy. Pass the `paths` array of an `enumerate_paths` result back in
unchanged, together with the policy you are asking for.

The policy name and version are not decoration. Where a source carries no edge
weights, a ranking is a Talent Angels modelling decision rather than source
data, and has to be cited that way. A suite with no honest way to score says so
instead of inventing an order.

ESCO today returns `score_paths_not_implemented`: its occupation–skill links
are binary (essential / optional), never numeric weights. Treat that warning as
the answer — do not substitute a ranking of your own and present it as ESCO's.

{_ID_DOC}""",
    )
    def score_paths(
        ctx: ToolContext,
        paths: Annotated[
            list[PathRef],
            Field(
                description="Routes to rank — the `paths` array from `enumerate_paths`, verbatim."
            ),
        ],
        policy_name: Annotated[
            str,
            Field(description="Name of the declared scoring policy being requested."),
        ],
        policy_version: Annotated[
            str,
            Field(description="Version of that policy. A ranking is only citable with both."),
        ],
        suite: SuiteArg = None,
    ) -> ScoredPathsPayload:
        session = ctx.request_context.lifespan_context
        policy = PolicyRef(name=policy_name, version=policy_version)
        runtime, name, warnings = _open_runtime(session, suite)
        if runtime is None:
            return ScoredPathsPayload(suite=name, policy=policy, warnings=warnings)
        if not supports_score_paths(runtime.suite):
            return ScoredPathsPayload(
                suite=runtime.name,
                policy=policy,
                warnings=[f"capability_unavailable:score_paths:{runtime.name}"],
            )
        try:
            result = runtime.suite.score_paths([suite_path_arg(path) for path in paths], policy)
        except Exception as exc:
            return ScoredPathsPayload(
                suite=runtime.name,
                policy=policy,
                warnings=_operation_warning(exc),
            )
        return scored_paths_payload(result, suite=runtime.name, policy=policy)

    @server.tool(
        name="map_neighbors",
        title="Map the neighbourhood of a node",
        description=f"""\
Map what surrounds a node: repeat `get_neighbors` for `depth` levels, keeping
the best `breadth` new neighbours of each node. Deterministic, no model call;
the same input gives the same result.

With no `suite`, the name is mapped in every taxonomy and merged into one cited
map: each node carries `suite` and `cite` (its source id), `suites` reports each
taxonomy's status (`mapped`, `ambiguous` with a pick list, `not_found`,
`via_crosswalk`, `error`), and `CROSSWALK` edges plus `confirmed_by` appear only
where an official code links two nodes (BLS and O*NET by SOC code, JobTech and
ESCO by their ESCO URIs). Pass `suite` for one taxonomy, or `suites` for a subset.

`start` is a node ID (full determinism) or a name. An ambiguous name is never
guessed: you get `ambiguous_start` (or that suite's `ambiguous` status) and a
`candidates` pick list; ask the user, then call again with an ID.

Per node, neighbours are ranked (map-rank-v1): relation (related skill,
narrower, level, parent, then jobs/skills needed), then essential before
optional, then O*NET importance and hot technology, then specificity (fewer
links beats a hub), then label. Groups, categories and levels are shown but
never expanded. A node reached twice appears once, with both links kept as edges.

`pruned` counts what each node cut (considered / kept). `warnings` may add
`max_nodes:N` (the cap was hit; the map is partial).

Route mode: `route_from` and `route_to` (names) map around the top-ranked
Pathfind route in each taxonomy. `route_node_ids` (one suite, a route from
`enumerate_paths`) maps every stop; stops carry `on_route` and `stop`.

The result means "linked in the graph", not a learning order: no taxonomy says
what to learn first. {_ID_DOC}""",
    )
    def map_neighbors(
        ctx: ToolContext,
        start: Annotated[
            str, Field(description="Node ID from `search_nodes`, or a name. Ignored in route mode.")
        ] = "",
        depth: Annotated[
            int, Field(ge=1, le=MAX_DEPTH, description="Levels to repeat the hop.")
        ] = 1,
        breadth: Annotated[
            int | None,
            Field(
                ge=1,
                le=MAX_BREADTH,
                description="New neighbours kept per node. Default 4, or 3 in route mode.",
            ),
        ] = None,
        kinds: Annotated[
            list[str] | None,
            Field(description="Keep only these node kinds, e.g. ['Skill']. Omit for all."),
        ] = None,
        suite: Annotated[
            str | None,
            Field(
                description=(
                    "One taxonomy only. Omit it (and `suites`) to map every taxonomy "
                    "into one cited graph."
                )
            ),
        ] = None,
        suites: Annotated[
            list[str] | None,
            Field(description="Subset of taxonomies to merge, e.g. ['esco','onet']."),
        ] = None,
        max_nodes: Annotated[
            int, Field(ge=1, le=DEFAULT_MAX_NODES, description="Hard cap on nodes in the map.")
        ] = DEFAULT_MAX_NODES,
        route_from: Annotated[
            str | None, Field(description="Route mode across taxonomies: start name.")
        ] = None,
        route_to: Annotated[
            str | None, Field(description="Route mode across taxonomies: target name.")
        ] = None,
        route_node_ids: Annotated[
            list[str] | None,
            Field(
                description=(
                    "Route mode: node IDs of a route (`enumerate_paths` `node_ids`). Maps "
                    "around every stop and merges into one graph; `start` is ignored."
                )
            ),
        ] = None,
    ) -> MapPayload:
        session = ctx.request_context.lifespan_context
        if breadth is None:  # same defaults as `ta-agent map`
            breadth = 3 if (route_from or route_node_ids) else 4
        walk = {"depth": depth, "breadth": breadth, "kinds": kinds, "max_nodes": max_nodes}
        if bool(route_from) != bool(route_to):
            return MapPayload(
                suite=suite or "all",
                depth=depth,
                breadth=breadth,
                warnings=["route_needs_route_from_and_route_to"],
            )
        by_name_route = bool(route_from and route_to)
        if not route_node_ids and (not suite or by_name_route):
            try:
                # A named route with `suite` runs in that one taxonomy only.
                every = {"suites": [suite] if suite else suites, **walk}
                found = (
                    map_all_route(session.registry, route_from, route_to, **every)  # type: ignore[arg-type]
                    if route_from and route_to
                    else map_all(session.registry, start, **every)  # type: ignore[arg-type]
                )
            except UnknownSuiteError as exc:
                return MapPayload(
                    suite="all", depth=depth, breadth=breadth, warnings=[f"unknown_suite:{exc}"]
                )
            return MapPayload.model_validate({**found.to_dict(), "suite": "all"})
        runtime, name, warnings = _open_runtime(session, suite)
        if runtime is None:
            return MapPayload(suite=name, depth=depth, breadth=breadth, warnings=warnings)
        try:
            result = (
                map_route(runtime.suite, runtime.name, route_node_ids, **walk)  # type: ignore[arg-type]
                if route_node_ids
                else map_from_text(runtime.suite, runtime.name, start, **walk)  # type: ignore[arg-type]
            )
        except Exception as exc:
            return MapPayload(
                suite=runtime.name, depth=depth, breadth=breadth, warnings=_operation_warning(exc)
            )
        return MapPayload.model_validate(result.to_dict())

    @server.tool(
        name="discover_map",
        title="Map how a list of roles, tasks and skills connects",
        description=f"""\
Discovery map for a list of up to {MAX_ITEMS} items (market roles, tasks, skills; vague is fine).
Each item is resolved in every taxonomy with the same rule as `map_neighbors` (a clear label
match only; anything else is `unclear` with up to 3 candidates, or `not_found`, never guessed),
mapped, and merged into one cited graph. Deterministic, no model call: the same resolved node
set and graph data give the same JSON, whatever the order of the items.

The result classifies in code: `links` (a direct edge between two items), `connectors` (nodes
that two or more items reach, `broad` for groups/categories/levels), `routes` (Pathfind routes,
route-rank-v1, for unconnected pairs), `orphans`, `clusters`, `gaps` (unclear / not found per
taxonomy), and a verdict on each `related_to` pair the user guessed: `confirmed` (direct link),
`indirect` (shared connector or a route of at most 3 edges), `not_in_graph`, `unresolved`.

`resolved` is the recorded node set: pass it back as `nodes` to skip resolution and get the
identical result (edit it to fix an `unclear` item by giving its `id`).

The result means "linked in the graph", not a learning order, and nothing is inferred: every
link comes from a taxonomy edge or an official crosswalk. {_ID_DOC}""",
    )
    def discover_map(
        ctx: ToolContext,
        items: Annotated[
            list[str], Field(max_length=MAX_ITEMS, description="Role, task or skill names.")
        ],
        related_to: Annotated[
            list[list[str]] | None,
            Field(description="The user's guesses as pairs: ['Python', 'machine learning']."),
        ] = None,
        suites: Annotated[
            list[str] | None, Field(description="Subset of taxonomies; omit for all.")
        ] = None,
        depth: Annotated[int, Field(ge=1, le=MAX_DEPTH, description="Levels per item.")] = 2,
        breadth: Annotated[
            int, Field(ge=1, le=MAX_BREADTH, description="New neighbours kept per node.")
        ] = 4,
        kinds: Annotated[list[str] | None, Field(description="Keep only these node kinds.")] = None,
        max_nodes: Annotated[
            int,
            Field(ge=1, le=MAX_DISCOVER_NODES, description="Hard cap on nodes in the whole map."),
        ] = DEFAULT_DISCOVER_MAX_NODES,
        nodes: Annotated[
            list[dict[str, Any]] | None,
            Field(description="A previous result's `resolved` block: skips resolution."),
        ] = None,
    ) -> DiscoverPayload:
        session = ctx.request_context.lifespan_context
        guesses: dict[str, list[str]] = {}
        stray: list[str] = []
        for pair in related_to or []:
            if len(pair) == 2 and pair[0].casefold() in {i.casefold() for i in items}:
                guesses.setdefault(pair[0].casefold(), []).append(pair[1])
            else:
                stray.append(f"related_to_ignored:{';'.join(pair)}")
        try:
            listed, notes = build_items(
                {"item": i, "related_to": guesses.get(i.casefold(), [])} for i in items
            )
            found = discover(
                session.registry,
                listed,
                suites=suites,
                depth=depth,
                breadth=breadth,
                kinds=kinds,
                max_nodes=max_nodes,
                recorded=nodes,
                notes=notes,
            )
        except (ValueError, UnknownSuiteError) as exc:
            return DiscoverPayload(warnings=[f"invalid_input:{exc}"])
        return DiscoverPayload.model_validate({**found, "warnings": [*found["warnings"], *stray]})

    @server.tool(
        name="roadmap",
        title="Learning order from one occupation to another",
        description=f"""\
What `target` needs that `start` does not, in a learning order. Deterministic, no model call:
the same input and graph data give the same JSON. Policy `learn-order-v1`.

`start` and `target` are node IDs (full determinism) or names; each taxonomy resolves them with
the rule of `map_neighbors` (a clear label match only). An unclear name is `ambiguous` with a
`candidates` pick list for that taxonomy, never guessed; ask the user, then call again with IDs.
Per taxonomy: ESCO the target's essential skills minus the start's; JobTech the target's skills
minus the start's; O*NET skills and knowledge with importance >= 3.0 whose level for the target
exceeds the start's by >= 0.5 (both levels kept); BLS a `credentials` block (entry education,
experience, training); O*NET also gives the `job_zone` of both ends. SFIA has no order data for
occupations (`no_order_data`). `suites` reports each taxonomy: `ok`, `ambiguous`, `not_found`,
`no_order_data` or `error:<Type>`.

`steps` merge the taxonomies: 1 Foundations (knowledge), 2 Core, 3 Field, 4 Job-specific (by
ESCO reuse level; O*NET skills are Core). A stated ESCO essential-for link puts the needed skill
first; inside a step the largest level jump comes first, then importance, label and ID. Each item
has its `suite`, `id`, `cite`, `reasons` and `stated_prerequisite`: only `true` means a taxonomy
states it as a prerequisite; every other place comes from the policy. At most `per_step` items
per step and taxonomy are listed, the rest is `more`.

The result is ordered by learn-order-v1 over official taxonomy data. It is evidence, not a
syllabus or advice. {_ID_DOC}""",
    )
    def roadmap_tool(
        ctx: ToolContext,
        start: Annotated[str, Field(description="Node ID or name of the current occupation.")],
        target: Annotated[str, Field(description="Node ID or name of the goal occupation.")],
        suites: Annotated[
            list[str] | None, Field(description="Subset of taxonomies; omit for all.")
        ] = None,
        per_step: Annotated[
            int, Field(ge=1, le=MAX_PER_STEP, description="Items listed per step and taxonomy.")
        ] = 10,
    ) -> OrderPayload:
        session = ctx.request_context.lifespan_context
        try:
            found = roadmap(session.registry, start, target, suites=suites, per_step=per_step)
        except (ValueError, UnknownSuiteError) as exc:
            return OrderPayload(warnings=[f"invalid_input:{exc}"])
        return OrderPayload.model_validate(found)

    @server.tool(
        name="successors",
        title="What comes after a skill",
        description=f"""\
A tree of what comes after `start`, to `depth` levels and `breadth` children per node, at most
`max_nodes` nodes per taxonomy. Deterministic, no model call. Policy `learn-order-v1`.

`start` is a node ID or a name, resolved as in `map_neighbors` (a clear label match only, else
`ambiguous` with `candidates`). Children, ESCO: `stated` are the skills ESCO says need `start`
(essential-for); `policy` are its narrower skills and the related skills of a later reuse level.
SFIA: the next level of the skill (`stated`). Stated children come first, then reuse level, label
and ID. A node appears once; groups and categories are never expanded, and SFIA levels are
followed only to the next level of the same skill. Each edge has
`kind` (`stated` | `policy`) and `reason`. O*NET, BLS and JobTech have no order data for a skill
(`no_order_data`).

The result is ordered by learn-order-v1 over official taxonomy data: only `stated` edges are
prerequisites stated by a taxonomy. It is evidence, not a syllabus. {_ID_DOC}""",
    )
    def successors_tool(
        ctx: ToolContext,
        start: Annotated[str, Field(description="Node ID or name of the skill.")],
        suites: Annotated[
            list[str] | None, Field(description="Subset of taxonomies; omit for all.")
        ] = None,
        depth: Annotated[int, Field(ge=1, le=MAX_DEPTH, description="Levels of the tree.")] = 2,
        breadth: Annotated[
            int, Field(ge=1, le=MAX_BREADTH, description="Children kept per node.")
        ] = 4,
        max_nodes: Annotated[
            int, Field(ge=1, le=DEFAULT_MAX_NODES, description="Hard cap on nodes per taxonomy.")
        ] = SUCCESSORS_MAX_NODES,
    ) -> OrderPayload:
        session = ctx.request_context.lifespan_context
        try:
            found = successors(
                session.registry,
                start,
                suites=suites,
                depth=depth,
                breadth=breadth,
                max_nodes=max_nodes,
            )
        except (ValueError, UnknownSuiteError) as exc:
            return OrderPayload(warnings=[f"invalid_input:{exc}"])
        return OrderPayload.model_validate(found)

    @server.tool(
        name="pathfind",
        title="Skill gap and routes between two occupations",
        description=f"""\
What separates occupation `start` from occupation `target`, per taxonomy: the skill gap
(`essential`, `optional` and `software` skills the target has and the start lacks) and up to
several ranked bridge `routes` through shared skills (route-rank-v1, `score` per route), plus
the compact learn-order-v1 `learning_order` when the taxonomy has one. Deterministic, no model
call, no phrasing: the same input and graph data give the same JSON.

`start` and `target` are occupation names; each taxonomy locates them itself. An unclear name
makes that taxonomy `ambiguous` with a `side` (`start` | `target`) and `candidates`; it is never
guessed, so ask the user and call again with the exact title. Per taxonomy `status` is `ok`,
`ambiguous`, `not_found`, `same_start_and_target` or `error:<Type>`; `gap_too_large` means the
gap is a near-complete retrain. BLS has no skill gap (its skill list is shared by every
occupation); its routes and gap are empty.

The result is graph data and may be cited as taxonomy fact; it is not career advice.
{_ID_DOC}""",
    )
    def pathfind_tool(
        ctx: ToolContext,
        start: Annotated[str, Field(min_length=1, description="Current occupation name.")],
        target: Annotated[str, Field(min_length=1, description="Goal occupation name.")],
        suites: Annotated[
            list[str] | None, Field(description="Subset of taxonomies; omit for all.")
        ] = None,
    ) -> PathfindPayload:
        from talent_angels.session.pathfind_view import pathfind_all  # import cycle via assistant

        session = ctx.request_context.lifespan_context
        try:
            found = pathfind_all(session.registry, start, target, suites=suites)
        except (ValueError, UnknownSuiteError) as exc:
            return PathfindPayload(warnings=[f"invalid_input:{exc}"])
        return PathfindPayload.model_validate(found)

    # The engine's front door (manifest, run, jobs). Imported here, not at module top: the
    # engine's skills import `mcp.models`, so a top-level import would be an import cycle.
    from talent_angels.mcp.engine_tools import add_engine_tools

    add_engine_tools(server)

    return server


def main() -> None:
    """Entry point for the `ta-mcp` console script (stdio transport)."""
    build_server().run(transport="stdio")
