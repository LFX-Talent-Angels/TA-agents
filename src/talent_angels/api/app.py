"""FastAPI edge — thin: LLM client for the app's lifetime, then every
request calls `assistant.run_turn` with the suite registry. No reasoning
lives here (ARCHITECTURE.md).
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from talent_angels.api.schemas import HealthResponse, QueryRequest, QueryResponse, UsageInfo
from talent_angels.assistant import run_turn
from talent_angels.assistant.intent import CAPABILITY_CONNECT, Capability
from talent_angels.contracts import NodeRef
from talent_angels.env import load_local_dotenv
from talent_angels.llm.factory import get_answer_mode, get_llm_client
from talent_angels.session.models import LastBinding, SessionState
from talent_angels.session.store import load_session, new_session, save_session
from talent_angels.suites import SuiteRegistry, UnknownSuiteError, default_suite_registry


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    load_local_dotenv()
    app.state.llm_client = get_llm_client()
    app.state.answer_mode = get_answer_mode()
    yield


def create_app(*, registry: SuiteRegistry | None = None) -> FastAPI:
    app = FastAPI(title="LFX Talent Angels — TA-agents", version="0.1.0", lifespan=lifespan)
    app.state.registry = registry or default_suite_registry()

    @app.get("/v1/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        reachable = False
        try:
            with app.state.registry.open() as runtime:
                reachable = runtime.is_reachable()
        except Exception:  # noqa: BLE001 — health must not raise
            reachable = False
        return HealthResponse(status="ok", neo4j_reachable=reachable)

    @app.post("/v1/query", response_model=QueryResponse)
    def query(payload: QueryRequest) -> QueryResponse:
        return _handle(app, payload)

    @app.post("/v1/capabilities/locate", response_model=QueryResponse)
    def locate_capability(payload: QueryRequest) -> QueryResponse:
        return _handle(app, payload, force_locate=True)

    @app.post("/v1/capabilities/connect", response_model=QueryResponse)
    def connect_capability(payload: QueryRequest) -> QueryResponse:
        return _handle(app, payload, force_capability=CAPABILITY_CONNECT)

    return app


def _apply_outcome_bindings(state: SessionState, outcome: object) -> None:
    """Update per-suite bindings from turn outcome, mirroring kernel._set_bind logic."""
    results = getattr(outcome, "results", ())
    for result in results:
        nodes = getattr(result, "nodes", [])
        warnings = getattr(result, "warnings", [])
        if nodes and "ambiguous" not in warnings and len(nodes) == 1:
            node: NodeRef = nodes[0]
            state.bindings[node.suite] = node
            state.binding = LastBinding(node=node)


def _handle(
    app: FastAPI,
    payload: QueryRequest,
    *,
    force_locate: bool = False,
    force_capability: Capability | None = None,
) -> QueryResponse:
    # Load or create session — gives the API the same bound-node continuity as the TUI.
    if payload.session_id:
        try:
            state = load_session(payload.session_id)
        except Exception:  # noqa: BLE001 — unknown/corrupt session starts fresh
            state = new_session()
    else:
        state = new_session()

    bound_node = state.binding.node if state.binding is not None else None
    bound_nodes = dict(state.bindings) if state.bindings else None

    try:
        outcome = run_turn(
            registry=app.state.registry,
            suite_override=payload.suite,
            llm_client=app.state.llm_client,
            question=payload.question,
            kind=payload.kind,
            answer_mode=app.state.answer_mode,
            force_locate=force_locate,
            force_capability=force_capability,
            bound_node=bound_node,
            bound_nodes=bound_nodes,
            thread_id=state.session_id,
        )
    except UnknownSuiteError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    _apply_outcome_bindings(state, outcome)
    save_session(state)

    record = outcome.record
    return QueryResponse(
        run_id=record.run_id,
        session_id=state.session_id,
        capability=outcome.capability,
        suite=outcome.result.suite,
        suites=[item.suite for item in outcome.results],
        result=outcome.result,
        results=list(outcome.results),
        answer=outcome.answer,
        usage=UsageInfo(
            gen_ai=record.gen_ai,
            cost_usd=record.cost_usd,
            graph=record.graph,
            tools=record.tools,
        ),
    )


app = create_app()
