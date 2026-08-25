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
    NeighborsPayload,
    PathRef,
    PathsPayload,
    PolicyRef,
    ScoredPathsPayload,
    SearchPayload,
)
from talent_angels.mcp.protocol import supports_enumerate_paths, supports_score_paths
from talent_angels.mcp.session import SuiteSession, SuiteUnavailableError
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
Read-only access to the Talent Angels taxonomy graphs through the suite
contract: search_nodes, get_neighbors, enumerate_paths, score_paths.

Four suites are registered. Each answers a different question, and none of
them answers another's. Pass `suite` to choose; it defaults to `esco`.

  esco   which occupations and skills exist, and whether a skill is essential
         or optional for an occupation. The only suite with occupation-specific
         skills, which is why an unqualified question lands here.
  onet   how much a generic skill matters for an occupation, surveyed, with
         sample sizes and confidence bounds. Every occupation is rated on the
         same small element set, so two distant occupations are comparable
         here even when they share nothing in ESCO.
  bls    employment, ten-year projections and wages, US only.
  sfia   how senior an IT skill is, levels 1 to 7. Structure only: the level
         names and descriptions are not redistributable.

A suite that is not loaded in the graph you are pointed at answers with
warnings rather than an error. Check rather than assume.

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
    """Wire the four contract tools onto an MCP server."""
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
        return neighbors_payload(result, suite=runtime.name, center_id=node_id)

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

    return server


def main() -> None:
    """Entry point for the `ta-mcp` console script (stdio transport)."""
    build_server().run(transport="stdio")
