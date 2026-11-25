"""Capability endpoints: 200 shape, 422, degraded suites, byte-identical bodies."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from talent_angels.api.app import create_app
from tests.fakes.order_suite import registry
from tests.fakes.suite import fake_registry
from tests.test_pathfind_dispatch_routes import _Suite as RouteSuite
from tests.test_roadmap import A, B, _esco
from tests.test_successors import X
from tests.test_successors import _esco as _skills


@pytest.fixture(autouse=True)
def _stub_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "none")


def _client(reg: object) -> TestClient:
    return TestClient(create_app(registry=reg))  # type: ignore[arg-type]


def _twice(c: TestClient, url: str, body: dict) -> dict:
    first, second = c.post(url, json=body), c.post(url, json=body)
    assert first.status_code == second.status_code == 200, first.text
    assert first.content == second.content  # byte-identical
    return first.json()


def test_pathfind_returns_the_gap_and_ranked_routes() -> None:
    reg = registry(esco=RouteSuite())
    data = _twice(
        _client(reg), "/v1/capabilities/pathfind", {"from": "nurse", "to": "data analyst"}
    )

    part = data["suites"]["esco"]
    assert part["status"] == "ok"
    assert [n["label"] for n in part["gap"]["essential"]] == ["statistics"]
    assert part["routes"][0]["labels"] == ["nurse", "statistics", "data analyst"]
    assert part["routes"][0]["score"] is not None


def test_roadmap_and_successors_match_the_cli_json() -> None:
    c = _client(registry(esco=_esco()))
    data = _twice(c, "/v1/capabilities/roadmap", {"from": A.id, "to": B.id, "per_step": 2})
    assert data["policy"] == "learn-order-v1" and data["per_step"] == 2
    assert data["steps"][0]["items"][0]["label"] == "algebra"

    c = _client(registry(esco=_skills()))
    body = {"skill": X.id, "depth": 1, "breadth": 2}
    data = _twice(c, "/v1/capabilities/successors", body)
    assert [n["id"] for n in data["suites"]["esco"]["nodes"]] == ["esco:x", "esco:sb", "esco:sa"]


def test_map_by_start_and_by_route() -> None:
    c = _client(fake_registry())
    one = _twice(c, "/v1/capabilities/map", {"start": "software developer", "suites": ["fake"]})
    assert one["suites"]["fake"]["status"] == "mapped" and one["summary"]["nodes"] >= 1
    route = c.post(
        "/v1/capabilities/map",
        json={"route_from": "software developer", "route_to": "programmer", "suites": ["fake"]},
    )
    assert route.status_code == 200 and route.json()["query"] == "software developer -> programmer"


def test_discover_takes_items_and_a_recorded_node_set() -> None:
    c = _client(fake_registry())
    body = {
        "items": [{"item": "software developer", "kind": "role", "related_to": ["Python"]}],
        "suites": ["fake"],
    }
    data = _twice(c, "/v1/capabilities/discover", body)
    assert data["policy"] == "discover-v1" and data["summary"]["items"] == 1

    again = _twice(c, "/v1/capabilities/discover", {**body, "nodes": data["resolved"]})
    assert again["resolved"] == data["resolved"]


def test_a_suite_that_is_down_degrades_instead_of_failing() -> None:
    c = _client(fake_registry())
    for url, body in (
        ("/v1/capabilities/pathfind", {"from": "a", "to": "b", "suites": ["down"]}),
        ("/v1/capabilities/map", {"start": "x", "suites": ["down"]}),
        ("/v1/capabilities/roadmap", {"from": "a", "to": "b", "suites": ["down"]}),
    ):
        r = c.post(url, json=body)
        assert r.status_code == 200, url
        assert "error" in str(r.json()["suites"]["down"]["status"]), url


@pytest.mark.parametrize(
    ("url", "body"),
    [
        ("pathfind", {"from": "a"}),
        ("pathfind", {"from": "", "to": "b"}),
        ("pathfind", {"from": "a", "to": "b", "extra": 1}),
        ("map", {}),
        ("map", {"start": "a", "route_from": "b", "route_to": "c"}),
        ("map", {"route_from": "b"}),
        ("map", {"start": "a", "depth": 9}),
        ("map", {"start": "a", "breadth": 0}),
        ("map", {"start": "a", "max_nodes": 99999}),
        ("discover", {"items": []}),
        ("discover", {"items": [{"item": f"i{n}"} for n in range(101)]}),
        ("discover", {"items": [{"item": "a"}], "max_nodes": 0}),
        ("roadmap", {"from": "a", "to": "b", "per_step": 51}),
        ("successors", {"skill": "a", "depth": 0}),
        ("successors", {"skill": "a", "max_nodes": 501}),
    ],
)
def test_bad_requests_are_422(url: str, body: dict) -> None:
    assert _client(fake_registry()).post(f"/v1/capabilities/{url}", json=body).status_code == 422


def test_an_unknown_suite_is_a_404_like_the_query_endpoint() -> None:
    c = _client(fake_registry())
    r = c.post("/v1/capabilities/roadmap", json={"from": "a", "to": "b", "suites": ["nope"]})
    assert r.status_code == 404 and "unknown suite" in r.json()["detail"]


def test_tag_is_not_exposed_over_http() -> None:
    assert _client(fake_registry()).post("/v1/capabilities/tag", json={}).status_code == 404


# --- the engine front door: manifest, /v1/run, jobs ---------------------------------------
def test_the_manifest_lists_the_eight_capabilities_with_schemas() -> None:
    r = _client(fake_registry()).get("/v1/capabilities")
    assert r.status_code == 200
    m = r.json()
    assert [c["name"] for c in m["capabilities"]] == [
        "locate", "connect", "pathfind", "map", "discover", "roadmap", "successors", "tag",
    ]  # fmt: skip
    assert all(c["input_schema"] and c["output_schema"] for c in m["capabilities"])
    assert [c["name"] for c in m["capabilities"] if c["uses_model"]] == ["tag"]
    assert "POST /v1/run" == m["run"]["http"]


def test_run_gives_the_same_output_as_the_dedicated_route_byte_for_byte() -> None:
    reg = registry(esco=_esco())
    c = _client(reg)
    body = {"from": A.id, "to": B.id, "per_step": 2}
    dedicated = c.post("/v1/capabilities/roadmap", json=body)
    run = c.post("/v1/run", json={"capability": "roadmap", "input": body})
    assert run.status_code == 200 and run.json()["status"] == "ok"
    assert run.json()["call"]["capability"] == "roadmap"
    assert run.json()["output"] == dedicated.json()
    import json as _json

    assert _json.dumps(run.json()["output"]) == _json.dumps(dedicated.json())  # key order too


def test_run_covers_the_five_routes_and_an_mcp_alias() -> None:
    c = _client(fake_registry())
    for name, body, dedicated in (
        ("map", {"start": "software developer", "suites": ["fake"]}, "map"),
        ("discover", {"items": [{"item": "software developer"}], "suites": ["fake"]}, "discover"),
        ("pathfind", {"from": "a", "to": "b", "suites": ["fake"]}, "pathfind"),
        ("roadmap", {"from": "a", "to": "b", "suites": ["fake"]}, "roadmap"),
        ("successors", {"skill": "a", "suites": ["fake"]}, "successors"),
        ("map_neighbors", {"start": "software developer", "suites": ["fake"]}, "map"),
    ):
        run = c.post("/v1/run", json={"capability": name, "input": body}).json()
        assert run["output"] == c.post(f"/v1/capabilities/{dedicated}", json=body).json(), name


def test_run_always_answers_with_the_result_envelope() -> None:
    c = _client(fake_registry())
    cases = (
        ({"capability": "roadmap", "input": {"from": "a"}}, 422, "invalid_input"),
        ({"capability": "roadmap", "input": {"from": "a", "to": "b", "x": 1}},
         422, "invalid_input"),
        ({"capability": "nope", "input": {}}, 404, "unknown_capability"),
        ({"capability": "roadmap", "input": {"from": "a", "to": "b", "suites": ["x"]}}, 404,
         "unknown_suite"),
        ({"capability": "tag", "input": {"path": "x"}}, 409, "use_job"),
    )  # fmt: skip
    for body, code, status in cases:
        r = c.post("/v1/run", json=body)
        assert (r.status_code, r.json()["status"]) == (code, status), body
        assert r.json()["output"] is None
    assert c.post("/v1/run", json={"capability": "map", "input": {}, "extra": 1}).status_code == 422
    assert c.post("/v1/run", json={"input": {}}).status_code == 422


def test_an_unexpected_error_over_http_is_a_500_with_the_type_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import talent_angels.skills.roadmap as skill

    def boom(*_a: object, **_k: object) -> None:
        raise RuntimeError("sk-SECRET")

    monkeypatch.setattr(skill, "roadmap", boom)
    c = _client(fake_registry())
    for url, body in (
        ("/v1/run", {"capability": "roadmap", "input": {"from": "a", "to": "b"}}),
        ("/v1/capabilities/roadmap", {"from": "a", "to": "b"}),
    ):
        r = c.post(url, json=body)
        assert r.status_code == 500 and "SECRET" not in r.text and "RuntimeError" in r.text


def _tag_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    import shutil

    from tests.test_tagging_run import FIXTURE

    root = tmp_path / "root"
    (root / "proj" / "inputs").mkdir(parents=True)
    shutil.copy(FIXTURE, root / "proj" / "inputs" / "standard.json")
    monkeypatch.setenv("TA_AGENTS_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("TA_TAG_ROOT", str(root))
    monkeypatch.delenv("TA_JOBS_ALLOW_MODEL", raising=False)
    monkeypatch.setenv("TA_JOBS_ENABLED", "1")  # starting jobs over HTTP is opt-in
    return root


def test_a_tag_job_starts_polls_to_done_and_stays_in_the_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from talent_angels.engine import jobs
    from tests.test_tagging_run import _registry as tag_registry

    root = _tag_env(monkeypatch, tmp_path)
    c = _client(tag_registry())
    r = c.post("/v1/jobs", json={"capability": "tag", "input": {"path": "proj", "limit": 3}})
    assert r.status_code == 202 and r.json()["status"] == "running"
    job_id = r.json()["id"]
    jobs._join(job_id)
    done = c.get(f"/v1/jobs/{job_id}").json()
    assert done["status"] == "done" and done["result"]["output"]["units"] > 0
    assert list((root / "proj").rglob("*.xlsx"))
    assert c.post(f"/v1/jobs/{job_id}/cancel").json()["status"] == "done"  # harmless

    # the guardrails, over HTTP
    for body in (
        {"capability": "tag", "input": {"path": "../x"}},
        {"capability": "tag", "input": {"path": str(root)}},
        {"capability": "tag", "input": {"path": "proj", "no_llm": False}},  # no TA_JOBS_ALLOW_MODEL
        {"capability": "tag", "input": {"path": "proj", "extra": 1}},
    ):
        bad = c.post("/v1/jobs", json=body)
        assert bad.status_code == 422 and bad.json()["status"] == "invalid_input", body
    assert c.post("/v1/jobs", json={"capability": "shell", "input": {}}).status_code == 404


def test_starting_a_job_over_http_is_forbidden_unless_the_server_enables_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from talent_angels.engine import jobs
    from tests.test_tagging_run import _registry as tag_registry

    root = _tag_env(monkeypatch, tmp_path)
    monkeypatch.delenv("TA_JOBS_ENABLED")
    c = _client(tag_registry())
    body = {"capability": "tag", "input": {"path": "proj", "limit": 3}}
    for value in (None, "", "0", "true"):  # only "1" opens it
        if value is not None:
            monkeypatch.setenv("TA_JOBS_ENABLED", value)
        denied = c.post("/v1/jobs", json=body)
        assert denied.status_code == 403 and "TA_JOBS_ENABLED=1" in denied.json()["detail"]
    assert not list((root / "proj").rglob("*.xlsx")) and not jobs.jobs_dir().exists()

    monkeypatch.setenv("TA_JOBS_ENABLED", "1")
    started = c.post("/v1/jobs", json=body)
    assert started.status_code == 202
    jobs._join(started.json()["id"])

    monkeypatch.delenv("TA_JOBS_ENABLED")  # status and cancel of an existing job stay open
    assert c.get(f"/v1/jobs/{started.json()['id']}").status_code == 200
    assert c.post(f"/v1/jobs/{started.json()['id']}/cancel").status_code == 200


def test_job_ids_are_validated_and_unknown_ones_are_404() -> None:
    c = _client(fake_registry())
    assert c.get(f"/v1/jobs/{'0' * 32}").status_code == 404
    assert c.post(f"/v1/jobs/{'0' * 32}/cancel").status_code == 404
    assert c.get("/v1/jobs/..%2F..%2Fetc%2Fpasswd").status_code in (404, 422)
    assert c.get("/v1/jobs/not-an-id").status_code == 422
