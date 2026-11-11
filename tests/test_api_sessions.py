"""The HTTP edge as a conversation backend a web UI can rely on."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from talent_angels.api.app import create_app
from talent_angels.session.store import sessions_dir
from tests.fakes.suite import fake_registry, suite_factory, unopenable_factory


@pytest.fixture(autouse=True)
def _stub_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "none")
    monkeypatch.setenv("ANSWER_MODE", "structured")


@pytest.fixture
def client() -> TestClient:
    with TestClient(create_app(registry=fake_registry())) as c:
        yield c


def test_a_session_carries_history_across_turns(client: TestClient) -> None:
    sid = client.post("/v1/sessions").json()["session_id"]
    first = client.post("/v1/query", json={"question": "software developer", "session_id": sid})
    second = client.post(
        "/v1/query", json={"question": "what skills does it need", "session_id": sid}
    )
    assert first.status_code == second.status_code == 200
    assert second.json()["capability"] == "connect"
    assert second.json()["plan"]

    history = client.get(f"/v1/sessions/{sid}/history").json()["turns"]
    assert [turn["question"] for turn in history] == [
        "software developer",
        "what skills does it need",
    ]
    assert history[0]["run_id"] == first.json()["run_id"]


def test_unknown_or_malformed_session_ids_are_rejected(client: TestClient) -> None:
    unknown = client.post("/v1/query", json={"question": "nurse", "session_id": "nope123"})
    assert unknown.status_code == 404
    for bad in ("../../escape", "a/b", ".hidden", "x" * 65):
        r = client.post("/v1/query", json={"question": "nurse", "session_id": bad})
        assert r.status_code == 422, bad
    assert client.get("/v1/sessions/nope123/history").status_code == 404


def test_erasing_a_session_forgets_its_files_and_history(client: TestClient) -> None:
    sid = client.post("/v1/sessions").json()["session_id"]
    other = client.post("/v1/sessions").json()["session_id"]
    for s in (sid, other):
        client.post("/v1/query", json={"question": "software developer", "session_id": s})

    erased = client.delete(f"/v1/sessions/{sid}")
    assert erased.status_code == 200
    assert erased.json()["history_deleted"] is True
    assert erased.json()["session_files_deleted"] > 0
    assert client.get(f"/v1/sessions/{sid}/history").status_code == 404
    assert len(client.get(f"/v1/sessions/{other}/history").json()["turns"]) == 1


def test_api_sessions_never_become_the_tui_resume_target(client: TestClient) -> None:
    client.post("/v1/query", json={"question": "software developer"})
    assert not (sessions_dir() / "_last").exists()


def test_input_is_bounded_and_an_empty_question_is_answered(client: TestClient) -> None:
    too_long = client.post("/v1/query", json={"question": "x" * 2001})
    assert too_long.status_code == 422

    empty = client.post("/v1/query", json={"question": ""})
    assert empty.status_code == 200
    assert empty.json()["result"]["warnings"] == ["no_subject"]


def test_health_reports_each_suite_and_degrades() -> None:
    from talent_angels.suites import SuiteRegistry
    from tests.fakes.suite import FakeSuite

    registry = SuiteRegistry(
        {
            "esco": suite_factory("esco", FakeSuite()),
            "onet": unopenable_factory(RuntimeError("down")),
        },
        default="esco",
    )
    with TestClient(create_app(registry=registry)) as c:
        body = c.get("/v1/health").json()
    assert body["status"] == "degraded"
    assert body["neo4j_reachable"] is True
    assert {s["name"]: s["reachable"] for s in body["suites"]} == {"esco": True, "onet": False}


def test_a_misconfigured_llm_does_not_stop_the_api(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "litellm")
    monkeypatch.setenv("LLM_MODEL", "")
    with TestClient(create_app(registry=fake_registry())) as c:
        health = c.get("/v1/health").json()
        answered = c.post("/v1/query", json={"question": "software developer"})
    assert "LLM_MODEL" in health["llm_error"]
    assert answered.status_code == 200


def test_cors_is_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    preflight = {
        "Origin": "http://localhost:8501",
        "Access-Control-Request-Method": "POST",
    }
    with TestClient(create_app(registry=fake_registry())) as c:
        assert c.options("/v1/query", headers=preflight).status_code == 405
    monkeypatch.setenv("TA_CORS_ORIGINS", "http://localhost:8501")
    with TestClient(create_app(registry=fake_registry())) as c:
        r = c.options("/v1/query", headers=preflight)
    assert r.status_code == 200
    assert r.headers["access-control-allow-origin"] == "http://localhost:8501"
