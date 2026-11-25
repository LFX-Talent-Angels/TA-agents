"""Capability endpoints: 200 shape, 422, degraded suites, byte-identical bodies."""

from __future__ import annotations

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
