"""FastAPI edge — thin: LLM client for the app's lifetime, then every
request calls `assistant.run_turn` with the suite registry. No reasoning
lives here (ARCHITECTURE.md).
"""

from __future__ import annotations

import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Path
from fastapi.middleware.cors import CORSMiddleware

from talent_angels.api.schemas import (
    SESSION_ID_PATTERN,
    HealthResponse,
    QueryRequest,
    QueryResponse,
    SessionCreated,
    SessionErased,
    SessionHistory,
    SuiteHealth,
    UsageInfo,
)
from talent_angels.assistant import run_turn
from talent_angels.assistant.checkpoint import forget_thread
from talent_angels.assistant.intent import CAPABILITY_CONNECT, Capability
from talent_angels.assistant.turn import TurnOutcome
from talent_angels.assistant.turn_graph import conversation_history
from talent_angels.contracts import NodeRef
from talent_angels.env import load_local_dotenv
from talent_angels.llm.factory import get_answer_mode, get_llm_client
from talent_angels.llm.stub_client import StubLLMClient
from talent_angels.memory.erase import erase_session
from talent_angels.session.models import LastBinding, SessionState
from talent_angels.session.store import (
    load_session,
    new_session,
    save_session,
    session_exists,
    sessions_dir,
)
from talent_angels.suites import SuiteRegistry, UnknownSuiteError, default_suite_registry

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    load_local_dotenv()
    try:
        app.state.llm_client = get_llm_client()
        app.state.llm_error = None
    except ValueError as exc:
        # A bad LLM config must not keep the API from starting: serve the
        # deterministic path and say so on /v1/health.
        logger.warning("LLM misconfigured, serving deterministic answers: %s", exc)
        app.state.llm_client = StubLLMClient()
        app.state.llm_error = str(exc)
    app.state.answer_mode = get_answer_mode()
    yield


def _cors_origins() -> list[str]:
    raw = os.environ.get("TA_CORS_ORIGINS", "").strip()
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


def create_app(*, registry: SuiteRegistry | None = None) -> FastAPI:
    app = FastAPI(title="LFX Talent Angels — TA-agents", version="0.1.0", lifespan=lifespan)
    app.state.registry = registry or default_suite_registry()
    origins = _cors_origins()
    if origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_methods=["GET", "POST", "DELETE"],
            allow_headers=["*"],
        )

    @app.get("/v1/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        registry: SuiteRegistry = app.state.registry
        suites: list[SuiteHealth] = []
        for name in registry.available:
            try:
                with registry.open(name) as runtime:
                    reachable = runtime.is_reachable()
            except Exception:  # noqa: BLE001 — health must not raise
                reachable = False
            suites.append(SuiteHealth(name=name, reachable=reachable))
        client = getattr(app.state, "llm_client", None)
        return HealthResponse(
            status="ok" if suites and all(s.reachable for s in suites) else "degraded",
            neo4j_reachable=any(s.reachable for s in suites),
            suites=suites,
            llm_provider=str(getattr(client, "provider", "none")),
            llm_model=str(getattr(client, "model", "")),
            llm_error=getattr(app.state, "llm_error", None),
        )

    @app.post("/v1/query", response_model=QueryResponse)
    def query(payload: QueryRequest) -> QueryResponse:
        return _handle(app, payload)

    @app.post("/v1/capabilities/locate", response_model=QueryResponse)
    def locate_capability(payload: QueryRequest) -> QueryResponse:
        return _handle(app, payload, force_locate=True)

    @app.post("/v1/capabilities/connect", response_model=QueryResponse)
    def connect_capability(payload: QueryRequest) -> QueryResponse:
        return _handle(app, payload, force_capability=CAPABILITY_CONNECT)

    @app.post("/v1/sessions", response_model=SessionCreated, status_code=201)
    def create_session() -> SessionCreated:
        state = new_session()
        save_session(state, update_last=False)
        return SessionCreated(session_id=state.session_id)

    @app.get("/v1/sessions/{session_id}/history", response_model=SessionHistory)
    def session_history(
        session_id: str = Path(pattern=SESSION_ID_PATTERN),
    ) -> SessionHistory:
        _require_session(session_id)
        return SessionHistory(session_id=session_id, turns=conversation_history(session_id))

    @app.delete("/v1/sessions/{session_id}", response_model=SessionErased)
    def erase_session_endpoint(
        session_id: str = Path(pattern=SESSION_ID_PATTERN),
    ) -> SessionErased:
        """Forget one conversation: its files and its checkpointed history."""
        _require_session(session_id)
        erased = erase_session(sessions_dir() / session_id)
        return SessionErased(
            session_id=session_id,
            session_files_deleted=erased.session_files_deleted,
            history_deleted=forget_thread(session_id),
        )

    return app


def _require_session(session_id: str) -> None:
    if not session_exists(session_id):
        raise HTTPException(status_code=404, detail=f"unknown session {session_id!r}")


def _apply_outcome_bindings(state: SessionState, outcome: TurnOutcome) -> None:
    """Update per-suite bindings from the turn, mirroring kernel._set_bind.

    A unique hit binds that suite. A fresh Locate that did not land on one
    node (ambiguous or not found) clears that suite's binding: the user moved
    on, and a stale binding would silently steer the next follow-up.
    """
    for result in outcome.results:
        suite = result.suite
        if result.nodes and "ambiguous" not in result.warnings and len(result.nodes) == 1:
            node: NodeRef = result.nodes[0]
            label = (node.pref_label or "").strip()
            if not label or label.casefold() == "none":
                continue
            state.bindings[suite] = node
            state.binding = LastBinding(node=node)
        elif result.capability == "locate" and (
            "ambiguous" in result.warnings or "not_found" in result.warnings
        ):
            state.bindings.pop(suite, None)
            if state.binding is not None and state.binding.node.suite == suite:
                state.binding = None


def _handle(
    app: FastAPI,
    payload: QueryRequest,
    *,
    force_locate: bool = False,
    force_capability: Capability | None = None,
) -> QueryResponse:
    # Load or create session — gives the API the same bound-node continuity as the TUI.
    if payload.session_id:
        _require_session(payload.session_id)
        try:
            state = load_session(payload.session_id)
        except (OSError, ValueError) as exc:
            raise HTTPException(
                status_code=409, detail=f"session {payload.session_id!r} is unreadable"
            ) from exc
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
    save_session(state, update_last=False)

    record = outcome.record
    return QueryResponse(
        run_id=record.run_id,
        session_id=state.session_id,
        capability=outcome.capability,
        plan=list(outcome.plan.capabilities),
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
