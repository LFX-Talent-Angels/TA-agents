"""FastAPI edge — thin: one Neo4j driver + one LLM client for the app's
lifetime (lifespan), then every request calls `assistant.run_turn`. No
reasoning lives here (ARCHITECTURE.md).
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from ta_taxonomies.suites.esco.db import neo4j_driver
from ta_taxonomies.suites.esco.tools import EscoSuite

from talent_angels.api.schemas import HealthResponse, QueryRequest, QueryResponse, UsageInfo
from talent_angels.assistant import run_turn
from talent_angels.llm import get_llm_client


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    driver_cm = neo4j_driver()
    driver, database = driver_cm.__enter__()
    app.state.driver = driver
    app.state.suite = EscoSuite(driver, database=database)
    app.state.llm_client = get_llm_client()
    app.state.answer_mode = os.environ.get("ANSWER_MODE", "structured").strip() or "structured"
    try:
        yield
    finally:
        driver_cm.__exit__(None, None, None)


def create_app() -> FastAPI:
    app = FastAPI(title="Talent Angels — TA-agents", version="0.1.0", lifespan=lifespan)

    @app.get("/v1/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        try:
            app.state.driver.verify_connectivity()
            reachable = True
        except Exception:
            reachable = False
        return HealthResponse(status="ok", neo4j_reachable=reachable)

    @app.post("/v1/query", response_model=QueryResponse)
    def query(payload: QueryRequest) -> QueryResponse:
        return _handle(app, payload, force_locate=False)

    @app.post("/v1/capabilities/locate", response_model=QueryResponse)
    def locate_capability(payload: QueryRequest) -> QueryResponse:
        return _handle(app, payload, force_locate=True)

    return app


def _handle(app: FastAPI, payload: QueryRequest, *, force_locate: bool) -> QueryResponse:
    outcome = run_turn(
        suite=app.state.suite,
        llm_client=app.state.llm_client,
        question=payload.question,
        kind=payload.kind,
        answer_mode=app.state.answer_mode,
        force_locate=force_locate,
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
