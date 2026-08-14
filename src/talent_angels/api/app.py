"""FastAPI edge — thin: one Neo4j driver + one LLM client for the app's
lifetime (lifespan), then every request calls `assistant.run_turn`. No
reasoning lives here (ARCHITECTURE.md).
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from talent_angels.api.schemas import HealthResponse, QueryRequest, QueryResponse, UsageInfo
from talent_angels.assistant import run_turn
from talent_angels.assistant.intent import CAPABILITY_CONNECT, Capability
from talent_angels.llm.factory import get_answer_mode, get_llm_client
from talent_angels.suites import SuiteRegistry, default_suite_registry


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    with app.state.registry.open() as runtime:
        app.state.runtime = runtime
        app.state.llm_client = get_llm_client()
        app.state.answer_mode = get_answer_mode()
        yield


def create_app(*, registry: SuiteRegistry | None = None) -> FastAPI:
    app = FastAPI(title="Talent Angels — TA-agents", version="0.1.0", lifespan=lifespan)
    app.state.registry = registry or default_suite_registry()

    @app.get("/v1/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        reachable = app.state.runtime.is_reachable()
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


def _handle(
    app: FastAPI,
    payload: QueryRequest,
    *,
    force_locate: bool = False,
    force_capability: Capability | None = None,
) -> QueryResponse:
    outcome = run_turn(
        suite=app.state.runtime.suite,
        suite_name=app.state.runtime.name,
        llm_client=app.state.llm_client,
        question=payload.question,
        kind=payload.kind,
        answer_mode=app.state.answer_mode,
        force_locate=force_locate,
        force_capability=force_capability,
    )
    record = outcome.record
    return QueryResponse(
        run_id=record.run_id,
        capability=outcome.capability,
        suite=outcome.result.suite,
        result=outcome.result,
        answer=outcome.answer,
        usage=UsageInfo(
            gen_ai=record.gen_ai,
            cost_usd=record.cost_usd,
            graph=record.graph,
            tools=record.tools,
        ),
    )


app = create_app()
