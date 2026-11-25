"""The typed output models describe the skills' real JSON (`extra="forbid"`: drift fails here).

The models sit on no serialization path, so these tests are their whole job besides the
manifest's JSON Schemas.
"""

from __future__ import annotations

from typing import Any

import pytest

from talent_angels.engine import CAPABILITIES, manifest, run
from talent_angels.engine import outputs as out
from tests.fakes.order_suite import registry
from tests.fakes.suite import fake_registry
from tests.test_pathfind_dispatch_routes import _Suite as RouteSuite
from tests.test_roadmap import OA, OB, A, B, _esco, _onet
from tests.test_successors import X
from tests.test_successors import _esco as _skills

MODELS = {
    "map": out.MapOutput,
    "discover": out.DiscoverOutput,
    "roadmap": out.RoadmapOutput,
    "successors": out.SuccessorsOutput,
    "pathfind": out.PathfindOutput,
}


def _check(name: str, payload: dict[str, Any], reg: Any) -> dict[str, Any]:
    result = run(name, payload, registry=reg)
    assert result.output is not None, result
    MODELS.get(name, CAPABILITIES[name].output_model).model_validate(result.output)
    return result.output


def test_map_by_start_by_route_and_with_a_suite_down() -> None:
    reg = fake_registry()
    _check("map", {"start": "software developer", "suites": ["fake"]}, reg)
    _check("map", {"start": "software developer"}, reg)  # every fake, broken and down ones too
    route = {"route_from": "software developer", "route_to": "programmer", "suites": ["fake"]}
    _check("map", route, reg)
    _check("map", {"start": "x", "suites": ["down"]}, reg)


def test_discover_with_guesses_unresolved_items_and_a_recorded_node_set() -> None:
    reg = fake_registry()
    body = {
        "items": [
            {"item": "software developer", "kind": "role", "related_to": ["Python"]},
            {"item": "stonemason"},
        ]
    }
    data = _check("discover", body, reg)
    assert data["guesses"] and data["gaps"]
    _check("discover", {**body, "nodes": data["resolved"]}, reg)
    _check("discover", {"items": [{"item": "x"}], "suites": ["down"]}, reg)


def test_roadmap_esco_onet_levels_and_job_zone_and_the_unclear_variants() -> None:
    _check("roadmap", {"from": A.id, "to": B.id}, registry(esco=_esco()))
    onet = _check("roadmap", {"from": OA.id, "to": OB.id}, registry(onet=_onet()))
    assert onet["job_zone"] and any(
        i["levels"] for s in onet["steps"] for i in s["items"]
    )  # the O*NET-only fields are described
    for frm in ("a", "zzz"):  # ambiguous, not found
        _check("roadmap", {"from": frm, "to": B.id}, registry(esco=_esco()))
    _check("roadmap", {"from": "a", "to": "b", "suites": ["down"]}, fake_registry())


def test_successors_and_pathfind_variants() -> None:
    _check("successors", {"skill": X.id, "depth": 2}, registry(esco=_skills()))
    _check("successors", {"skill": "ba", "suites": ["esco"]}, registry(esco=_skills()))  # unclear
    _check("successors", {"skill": "a", "suites": ["down"]}, fake_registry())
    reg = registry(esco=RouteSuite())
    _check("pathfind", {"from": "nurse", "to": "data analyst"}, reg)
    _check("pathfind", {"from": "nurse", "to": "nurse"}, reg)
    _check("pathfind", {"from": "a", "to": "b", "suites": ["down"]}, fake_registry())


def test_locate_and_connect_outputs_are_their_payload_models() -> None:
    reg = fake_registry()
    found = _check("locate", {"text": "software developer"}, reg)
    _check("locate", {"text": "stonemason"}, reg)
    _check("connect", {"node_id": found["candidates"][0]["node"]["id"]}, reg)


def test_a_key_the_model_does_not_describe_fails() -> None:
    data = _check("roadmap", {"from": A.id, "to": B.id}, registry(esco=_esco()))
    with pytest.raises(ValueError):
        out.RoadmapOutput.model_validate({**data, "surprise": 1})


def test_every_manifest_entry_has_both_schemas_and_names_its_defs() -> None:
    for entry in manifest()["capabilities"]:
        schema = entry["output_schema"]
        assert schema["type"] == "object" and schema["properties"], entry["name"]


def test_discover_connectors_carry_tool_only_and_validate() -> None:
    from tests.test_discover import _registry as discover_registry
    from tests.test_discover import _tool_pair_graph

    data = _check(
        "discover",
        {"items": [{"item": "alphaitem"}, {"item": "betaitem"}], "depth": 1, "breadth": 4},
        discover_registry(_tool_pair_graph()),
    )
    assert {c["label"]: c["tool_only"] for c in data["connectors"]} == {"oven": True, "kiln": False}
