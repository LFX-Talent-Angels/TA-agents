"""The engine registry: eight declarations, one `run()`, the closed action set, byte identity.

Offline, with the fakes the capability tests already use. The MCP half of the drift guard lives
in `tests/test_mcp_server.py` (it has the running server).
"""

from __future__ import annotations

import json
import logging
from typing import Any

import pytest

from talent_angels.api.app import create_app
from talent_angels.cli import _build_parser
from talent_angels.engine import (
    CAPABILITIES,
    STATUSES,
    get_capability,
    manifest,
    run,
)
from talent_angels.session.slash import SLASH_COMMANDS
from talent_angels.skills.connect.discover import discover
from talent_angels.skills.connect.discover_io import build_items
from talent_angels.skills.connect.federated import map_all, map_all_route
from talent_angels.skills.roadmap import roadmap, successors
from talent_angels.suites import UnknownSuiteError
from tests.fakes.order_suite import registry
from tests.fakes.suite import fake_registry
from tests.test_pathfind_dispatch_routes import _Suite as RouteSuite
from tests.test_roadmap import A, B, _esco
from tests.test_successors import X
from tests.test_successors import _esco as _skills

NAMES = ["locate", "connect", "pathfind", "map", "discover", "roadmap", "successors", "tag"]
SKILL_ROUTES = ("pathfind", "map", "discover", "roadmap", "successors")


def _dump(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2)


# --- the declarations ---------------------------------------------------------------------
def test_eight_capabilities_in_the_declared_order_with_working_examples() -> None:
    assert list(CAPABILITIES) == NAMES
    for cap in CAPABILITIES.values():
        assert cap.examples and callable(cap.fn) and cap.title and cap.question
        for example in cap.examples:
            cap.input_model.model_validate(example)  # every example is a valid input


def test_the_manifest_flags_are_true_to_the_code() -> None:
    by_name = {c.name: c for c in CAPABILITIES.values()}
    # only `tag` can call a model, and only when it is not asked to search alone
    assert [n for n, c in by_name.items() if c.uses_model] == ["tag"]
    assert [n for n, c in by_name.items() if not c.deterministic] == ["tag"]
    assert [n for n, c in by_name.items() if c.long_running] == ["tag"]
    assert by_name["tag"].cost == "model" and by_name["map"].cost == "free"


def test_the_manifest_matches_the_registry_and_is_json() -> None:
    m = manifest()
    assert [c["name"] for c in m["capabilities"]] == NAMES
    for entry in m["capabilities"]:
        cap = CAPABILITIES[entry["name"]]
        assert entry["aliases"] == dict(cap.aliases)
        assert (entry["deterministic"], entry["uses_model"]) == (cap.deterministic, cap.uses_model)
        assert entry["input_schema"]["properties"] and entry["output_schema"]["properties"]
        assert entry["input_schema"].get("additionalProperties") is False  # extra="forbid"
    assert (
        "from"
        in next(c for c in m["capabilities"] if c["name"] == "roadmap")["input_schema"][
            "properties"
        ]
    )  # by alias, as the callers send it
    assert m["statuses"] == list(STATUSES)
    json.dumps(m)
    assert [c["name"] for c in manifest(only="map_neighbors")["capabilities"]] == ["map"]
    assert manifest(only="nope")["capabilities"] == []


def test_manifest_copies_are_independent() -> None:
    manifest()["capabilities"].clear()
    assert len(manifest()["capabilities"]) == 8


# --- drift guard: every alias exists on every interface -----------------------------------
def test_every_http_alias_is_a_route_and_every_capability_route_is_an_alias() -> None:
    routes = {r.path for r in create_app(registry=fake_registry()).routes}
    http = {c.aliases["http"] for c in CAPABILITIES.values() if "http" in c.aliases}
    assert http <= routes
    capability_routes = {
        r for r in routes if r.startswith("/v1/capabilities/") and r.count("/") == 3
    } - {"/v1/capabilities/locate", "/v1/capabilities/connect"}  # conversational turns
    assert capability_routes == http
    assert {"/v1/capabilities", "/v1/run", "/v1/jobs", "/v1/jobs/{job_id}"} <= routes


def test_every_cli_alias_is_a_subcommand_or_tag() -> None:
    sub = next(a for a in _build_parser()._actions if a.dest == "command")
    cli = {c.aliases["cli"] for c in CAPABILITIES.values() if "cli" in c.aliases}
    assert cli - {"tag"} <= set(sub.choices)  # type: ignore[attr-defined]
    assert {"capabilities", "run"} <= set(sub.choices)  # type: ignore[attr-defined]


def test_slash_commands_are_the_slash_aliases() -> None:
    aliases = tuple(c.aliases["slash"] for c in CAPABILITIES.values() if "slash" in c.aliases)
    assert SLASH_COMMANDS == aliases == ("map", "discover", "roadmap", "successors", "tag")


def test_an_mcp_alias_is_a_capability_name_but_a_route_is_not() -> None:
    assert get_capability("map_neighbors").name == "map"
    assert get_capability("search_nodes").name == "locate"
    assert get_capability("get_neighbors").name == "connect"
    for not_a_name in ("/v1/capabilities/map", "enumerate_paths", "nope", ""):
        with pytest.raises(KeyError):
            get_capability(not_a_name)


# --- run(): one case per status ------------------------------------------------------------
def test_ok_and_the_resolved_call() -> None:
    reg = registry(esco=_esco())
    r = run("roadmap", {"from": A.id, "to": B.id}, registry=reg)
    assert r.status == "ok" and r.output and r.call
    assert r.call.capability == "roadmap"
    assert r.call.input == {"from": A.id, "to": B.id, "suites": None, "per_step": 10}


def test_an_alias_runs_the_same_capability() -> None:
    r = run(
        "map_neighbors",
        {"start": "software developer", "suites": ["fake"]},
        registry=fake_registry(),
    )
    assert r.status == "ok" and r.call and r.call.capability == "map"


def test_ambiguous_carries_the_pick_list() -> None:
    r = run(
        "roadmap", {"from": "a", "to": B.id, "suites": ["esco"]}, registry=registry(esco=_esco())
    )
    assert r.status == "ambiguous" and r.candidates["esco"]
    assert all({"id", "label", "side"} <= set(c) for c in r.candidates["esco"])


def test_no_result_when_nothing_answers() -> None:
    r = run(
        "roadmap", {"from": "zzz", "to": B.id, "suites": ["esco"]}, registry=registry(esco=_esco())
    )
    assert r.status == "no_result" and r.output


def test_unavailable_when_every_selected_suite_is_down() -> None:
    reg = fake_registry()
    for name, payload in (
        ("map", {"start": "x", "suites": ["down"]}),
        ("roadmap", {"from": "a", "to": "b", "suites": ["down"]}),
        ("pathfind", {"from": "a", "to": "b", "suites": ["down"]}),
    ):
        r = run(name, payload, registry=reg)
        assert r.status == "unavailable" and r.output, name
    assert run("locate", {"text": "x", "suite": "down"}, registry=reg).status == "unavailable"


@pytest.mark.parametrize(
    ("capability", "payload"),
    [
        ("map", {"start": "a", "depth": 9}),
        ("map", {}),
        ("map", {"start": "a", "extra": 1}),
        ("roadmap", {"from": "a", "to": "b", "per_step": 0}),
        ("roadmap", {"from": "", "to": "b"}),
        ("successors", {"skill": "a" * 1001}),
        ("tag", {"path": "x", "refresh": True, "fresh": True}),
        ("tag", {"path": "x", "workers": 99}),
    ],
)
def test_bad_input_is_refused_and_never_runs(capability: str, payload: dict[str, Any]) -> None:
    r = run(capability, payload, registry=fake_registry(), allow_long=True)
    assert r.status == "invalid_input" and r.errors and r.error
    assert r.output is None and r.call is None


def test_an_unknown_name_is_refused_and_never_runs() -> None:
    for name in ("shell", "cypher", "MATCH (n) RETURN n", "map; drop", "/v1/capabilities/map"):
        r = run(name, {}, registry=fake_registry())
        assert r.status == "unknown_capability" and r.output is None and r.call is None
    long = run("x" * 5000, {}, registry=fake_registry())
    assert len(long.error or "") < 400  # a long junk name is not echoed in full


def test_unknown_suite_and_use_job() -> None:
    reg = fake_registry()
    assert run("roadmap", {"from": "a", "to": "b", "suites": ["nope"]}, registry=reg).status == (
        "unknown_suite"
    )
    assert run("locate", {"text": "x", "suite": "nope"}, registry=reg).status == "unknown_suite"
    r = run("tag", {"path": "x"}, registry=reg)
    assert r.status == "use_job" and r.output is None


def test_an_unexpected_error_shows_its_type_only(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    import talent_angels.skills.roadmap as skill

    def boom(*_a: object, **_k: object) -> None:
        raise RuntimeError("sk-or-v1-SECRET provider said no")

    monkeypatch.setattr(skill, "roadmap", boom)
    with caplog.at_level(logging.WARNING):
        r = run("roadmap", {"from": "a", "to": "b"}, registry=fake_registry())
    assert r.status == "error" and r.error == "RuntimeError"
    assert "SECRET" not in r.model_dump_json()
    assert "RuntimeError" in caplog.text  # the details go to the log


# --- byte identity: run() returns the skill's own dict ------------------------------------
def test_pathfind_is_the_skill_output_byte_for_byte() -> None:
    from talent_angels.session.pathfind_view import pathfind_all

    reg = registry(esco=RouteSuite())
    r = run("pathfind", {"from": "nurse", "to": "data analyst"}, registry=reg)
    assert _dump(r.output) == _dump(pathfind_all(reg, "nurse", "data analyst", suites=None))


def test_map_is_the_skill_output_byte_for_byte() -> None:
    reg = fake_registry()
    walk: dict[str, Any] = {"suites": ["fake"], "depth": 1, "kinds": None, "max_nodes": 500}
    r = run("map", {"start": "software developer", **walk}, registry=reg)
    direct = map_all(reg, "software developer", breadth=4, **walk).to_dict()
    assert _dump(r.output) == _dump(direct)
    ends = {"route_from": "software developer", "route_to": "programmer"}
    route = run("map", {**ends, **walk}, registry=reg)
    direct = map_all_route(reg, "software developer", "programmer", breadth=3, **walk).to_dict()
    assert _dump(route.output) == _dump(direct)


def test_discover_is_the_skill_output_byte_for_byte() -> None:
    reg = fake_registry()
    item = {"item": "software developer", "kind": "role", "related_to": ["Python"]}
    r = run("discover", {"items": [item], "suites": ["fake"]}, registry=reg)
    items, notes = build_items([{**item}])
    direct = discover(reg, items, suites=["fake"], depth=2, breadth=4, kinds=None,
                      max_nodes=2000, recorded=None, notes=notes)  # fmt: skip
    assert _dump(r.output) == _dump(direct)


def test_roadmap_and_successors_are_the_skill_output_byte_for_byte() -> None:
    reg = registry(esco=_esco())
    r = run("roadmap", {"from": A.id, "to": B.id, "per_step": 2}, registry=reg)
    assert _dump(r.output) == _dump(roadmap(reg, A.id, B.id, suites=None, per_step=2))
    reg = registry(esco=_skills())
    r = run("successors", {"skill": X.id, "depth": 1, "breadth": 2}, registry=reg)
    direct = successors(reg, X.id, suites=None, depth=1, breadth=2, max_nodes=200)
    assert _dump(r.output) == _dump(direct)


def test_the_resolved_call_reruns_to_the_same_bytes() -> None:
    reg = fake_registry()
    for name, payload in (
        ("map", {"start": "software developer", "suites": ["fake"]}),
        ("map", {"route_from": "software developer", "route_to": "programmer", "suites": ["fake"]}),
        ("discover", {"items": [{"item": "software developer"}], "suites": ["fake"]}),
    ):
        first = run(name, payload, registry=reg)
        again = run(first.call.capability, first.call.input, registry=reg)  # type: ignore[union-attr]
        assert _dump(first.output) == _dump(again.output), name
        assert first.model_dump_json() == run(name, payload, registry=reg).model_dump_json()


def test_locate_and_connect_are_the_suite_calls_with_typed_warnings() -> None:
    reg = fake_registry()
    found = run("locate", {"text": "software developer"}, registry=reg)
    assert found.status == "ok" and found.output and found.output["candidates"]
    miss = run("locate", {"text": "stonemason"}, registry=reg)
    assert miss.status == "no_result" and "not_found" in miss.warnings
    node_id = found.output["candidates"][0]["node"]["id"]
    near = run("connect", {"node_id": node_id, "limit": 2}, registry=reg)
    assert near.status == "ok" and near.output and near.output["center_id"] == node_id
    gone = run("connect", {"node_id": "fake:occupation:nope"}, registry=reg)
    assert gone.status == "no_result"


def test_unknown_suite_error_type_is_what_run_catches() -> None:
    assert issubclass(UnknownSuiteError, ValueError)  # run() tests it before ValueError
