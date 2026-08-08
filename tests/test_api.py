"""API tests — live against the ESCO fixture in Neo4j; skipped if unreachable."""

from __future__ import annotations

import pytest

pytest.importorskip("ta_taxonomies")

from fastapi.testclient import TestClient  # noqa: E402
from ta_taxonomies.suites.esco.db import neo4j_driver  # noqa: E402

from talent_angels.api.app import create_app  # noqa: E402


def _neo4j_reachable() -> bool:
    try:
        with neo4j_driver() as (driver, _database):
            driver.verify_connectivity()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _neo4j_reachable(), reason="Neo4j not reachable; see TA-taxonomies NOTES.md"
)


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch, tmp_path) -> TestClient:
    monkeypatch.setenv("RUNLOG_PATH", str(tmp_path / "runlog.jsonl"))
    monkeypatch.setenv("LLM_PROVIDER", "none")
    with TestClient(create_app()) as test_client:
        yield test_client


def test_health_reports_neo4j_reachable(client: TestClient) -> None:
    response = client.get("/v1/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["neo4j_reachable"] is True


def test_capabilities_locate_finds_software_developer(client: TestClient) -> None:
    response = client.post(
        "/v1/capabilities/locate",
        json={"question": "software developer", "kind": "occupation"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["capability"] == "locate"
    assert body["result"]["confidence"] == 0.95
    assert body["result"]["nodes"][0]["pref_label"] == "software developer"
    assert body["usage"]["cost_usd"]["total"] == 0.0


def test_query_routes_through_heuristic_intent(client: TestClient) -> None:
    response = client.post("/v1/query", json={"question": "software developer"})
    assert response.status_code == 200
    body = response.json()
    assert body["capability"] == "locate"
    assert body["result"]["confidence"] == 0.95


def test_query_not_found_returns_empty_nodes_not_error(client: TestClient) -> None:
    response = client.post("/v1/query", json={"question": "xyzzy-not-a-real-occupation"})
    assert response.status_code == 200
    body = response.json()
    assert body["result"]["nodes"] == []
    assert "not_found" in body["result"]["warnings"]
