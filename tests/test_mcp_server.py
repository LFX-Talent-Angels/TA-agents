"""Speak MCP to a real server process over stdio — fake suite, no database.

The tool bodies and their generated schemas only exist inside a running
server, so this drives one exactly as Claude Code does: launch the process,
initialize, call tools, read the structured payloads back.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import anyio
import pytest
from mcp import ClientSession, StdioServerParameters, stdio_client

from talent_angels.mcp import SERVER_NAME
from tests.fakes.suite import DEV, PYTHON

REPO_ROOT = Path(__file__).resolve().parents[1]


def _server_parameters() -> StdioServerParameters:
    env = dict(os.environ)
    # `-m tests.fakes.mcp_stdio` needs both the repo root and the src layout,
    # whether or not the package happens to be installed in this environment.
    env["PYTHONPATH"] = os.pathsep.join([str(REPO_ROOT), str(REPO_ROOT / "src")])
    return StdioServerParameters(
        command=sys.executable,
        args=["-m", "tests.fakes.mcp_stdio"],
        cwd=str(REPO_ROOT),
        env=env,
    )


async def _drive() -> dict[str, Any]:
    """One server process, every call this module asserts on."""
    async with stdio_client(_server_parameters()) as (read, write):
        async with ClientSession(read, write) as session:
            init = await session.initialize()
            listed = await session.list_tools()
            collected: dict[str, Any] = {
                "server_name": init.server_info.name,
                "instructions": init.instructions or "",
                "tools": {tool.name: tool for tool in listed.tools},
            }

            async def call(key: str, tool: str, args: dict[str, Any]) -> None:
                result = await session.call_tool(tool, args)
                collected[key] = result.structured_content

            await call("search", "search_nodes", {"text": "software developer"})
            await call("map", "map_neighbors", {"start": "software developer", "suite": "fake"})
            await call("map_all", "map_neighbors", {"start": "software developer"})
            await call(
                "map_subset",
                "map_neighbors",
                {"start": "software developer", "suites": ["fake", "bogus"]},
            )
            await call("map_route", "map_neighbors", {"route_node_ids": [DEV.id], "suite": "fake"})
            await call("map_bad", "map_neighbors", {"start": "x", "suite": "down"})
            await call(
                "discover",
                "discover_map",
                {
                    "items": ["software developer", "stonemason"],
                    "related_to": [["software developer", "stonemason"], ["Rust", "stonemason"]],
                },
            )
            await call("discover_big", "discover_map", {"items": [f"i{n}" for n in range(101)]})
            await call("roadmap", "roadmap", {"start": "software developer", "target": "x"})
            bogus = {"suites": ["bogus"]}
            await call("roadmap_suite", "roadmap", {"start": "a", "target": "b", **bogus})
            await call("roadmap_limit", "roadmap", {"start": "a", "target": "b", "per_step": 0})
            await call("successors", "successors", {"start": "Python"})
            await call("successors_suite", "successors", {"start": "a", **bogus})
            await call("successors_limit", "successors", {"start": "a", "depth": 9})
            await call(
                "pathfind",
                "pathfind",
                {"start": "software developer", "target": "programmer", "suites": ["fake"]},
            )
            await call("pathfind_suite", "pathfind", {"start": "a", "target": "b", "suites": ["x"]})
            await call("pathfind_blank", "pathfind", {"start": "", "target": "b"})
            await call("miss", "search_nodes", {"text": "stonemason"})
            await call("neighbors", "get_neighbors", {"node_id": DEV.id})
            await call("neighbors_missing", "get_neighbors", {"node_id": "fake:occupation:nope"})
            await call("paths", "enumerate_paths", {"from_id": DEV.id, "to_id": PYTHON.id})
            await call(
                "invalid_depth",
                "enumerate_paths",
                {"from_id": DEV.id, "to_id": PYTHON.id, "max_depth": 0},
            )
            await call("unknown_suite", "search_nodes", {"text": "x", "suite": "bls"})
            await call("suite_down", "search_nodes", {"text": "x", "suite": "down"})
            await call("suite_broken", "search_nodes", {"text": "x", "suite": "broken"})
            await call(
                "no_pathfind",
                "enumerate_paths",
                {"from_id": DEV.id, "to_id": PYTHON.id, "suite": "locate_only"},
            )
            await call(
                "scored",
                "score_paths",
                {
                    "paths": collected["paths"]["paths"],
                    "policy_name": "essential-first",
                    "policy_version": "0.1.0",
                },
            )
            return collected


@pytest.fixture(scope="module")
def served() -> dict[str, Any]:
    return anyio.run(_drive)


def test_the_server_exposes_the_contract_tools_and_map_neighbors(served: dict[str, Any]) -> None:
    assert served["server_name"] == SERVER_NAME
    assert set(served["tools"]) == {
        "search_nodes",
        "get_neighbors",
        "enumerate_paths",
        "score_paths",
        "map_neighbors",
        "discover_map",
        "roadmap",
        "successors",
        "pathfind",
    }


def test_pathfind_returns_the_path_card_data_per_taxonomy(served: dict[str, Any]) -> None:
    found = served["pathfind"]
    assert found["suites"]["fake"]["status"] == "same_start_and_target"  # both names are one node
    assert found["start"] == "software developer"
    assert served["pathfind_suite"]["warnings"][0].startswith("invalid_input:unknown suite")
    assert served["pathfind_blank"] is None  # refused up front, like the other limits


def test_roadmap_and_successors_return_the_learn_order_json_as_structured_results(
    served: dict[str, Any],
) -> None:
    for key, tool in (("roadmap", "roadmap"), ("successors", "successors")):
        payload = served[key]
        assert payload["policy"] == "learn-order-v1"
        assert "only items marked stated_prerequisite" in payload["meaning"]
        # none of the fake suites carries order data, and none is guessed
        assert {s["status"] for s in payload["suites"].values()} <= {
            "no_order_data",
            "error:ConnectionRefusedError",
            "error:RuntimeError",
        }, tool
    assert served["roadmap"]["steps"] == []
    # a bad suite name is a structured warning; a limit past its bound is refused up front
    for key in ("roadmap_suite", "successors_suite"):
        assert served[key]["warnings"][0].startswith("invalid_input:unknown suite")
    assert served["roadmap_limit"] is None and served["successors_limit"] is None


def test_discover_map_returns_the_cited_discovery_json(served: dict[str, Any]) -> None:
    found = served["discover"]
    assert found["policy"] == "discover-v1"
    assert [(i["item"], i["status"]) for i in found["items"]] == [
        ("software developer", "resolved"),
        ("stonemason", "not_found"),
    ]
    assert found["suites"]["broken"]["status"] == "error:RuntimeError"
    assert [(g["item"], g["verdict"]) for g in found["guesses"]] == [
        ("software developer", "unresolved")
    ]
    assert found["warnings"] == ["related_to_ignored:Rust;stonemason"]
    assert all(n["cite"] for n in found["nodes"])


def test_discover_map_refuses_more_than_a_hundred_items(served: dict[str, Any]) -> None:
    # The tool's own input limit rejects the call before any suite is opened.
    assert served["discover_big"] is None


def test_instructions_state_the_two_rules_a_client_can_get_wrong(
    served: dict[str, Any],
) -> None:
    instructions = served["instructions"]

    assert "suite-scoped" in instructions
    assert "not a probability" in instructions


def test_search_describes_confidence_as_a_match_method(served: dict[str, Any]) -> None:
    description = served["tools"]["search_nodes"].description or ""

    assert "HOW the match was made" in description
    # The number a client is most likely to read as "70% sure".
    assert "0.70 does not mean" in description


def test_every_tool_declares_its_inputs_and_a_structured_output(
    served: dict[str, Any],
) -> None:
    expected = {
        "search_nodes": {"text", "kind", "suite"},
        "get_neighbors": {"node_id", "rel_types", "suite", "limit"},
        "enumerate_paths": {"from_id", "to_id", "max_depth", "max_paths", "suite"},
        "score_paths": {"paths", "policy_name", "policy_version", "suite"},
        "roadmap": {"start", "target", "suites", "per_step"},
        "successors": {"start", "suites", "depth", "breadth", "max_nodes"},
        "pathfind": {"start", "target", "suites"},
        "map_neighbors": {
            "start",
            "depth",
            "breadth",
            "kinds",
            "suite",
            "suites",
            "max_nodes",
            "route_node_ids",
            "route_from",
            "route_to",
        },
    }
    for name, arguments in expected.items():
        tool = served["tools"][name]
        assert set(tool.input_schema["properties"]) == arguments, name
        assert tool.output_schema is not None, name
        # The injected context must never leak into the client-facing schema.
        assert "ctx" not in tool.input_schema["properties"], name


def test_search_returns_typed_candidates_with_the_match_method(
    served: dict[str, Any],
) -> None:
    candidate = served["search"]["candidates"][0]

    assert candidate["node"]["id"] == DEV.id
    assert candidate["node"]["pref_label"] == "software developer"
    assert (candidate["confidence"], candidate["method"]) == (0.95, "exact_pref")
    assert served["search"]["warnings"] == []


def test_neighbors_return_typed_edges_with_their_relation_type(
    served: dict[str, Any],
) -> None:
    edge = served["neighbors"]["edges"][0]

    assert edge["type"] == "HAS_SKILL"
    assert edge["target_node_id"] == PYTHON.id
    assert edge["properties"] == {"relation_type": "essential"}


def test_map_neighbors_returns_a_typed_map(served: dict[str, Any]) -> None:
    payload = served["map"]

    assert payload["start"]["id"] == DEV.id
    assert [n["level"] for n in payload["nodes"]][:2] == [0, 1]
    assert payload["edges"][0]["relation"] == "needs"
    assert payload["warnings"] == []
    assert "on_route" not in payload["nodes"][0] and "suite" not in payload["nodes"][0]
    assert served["map_route"]["route"] == [DEV.id]
    assert served["map_route"]["nodes"][0]["on_route"] is True
    assert served["map_bad"]["warnings"] == ["suite_unavailable:ConnectionRefusedError"]


def test_map_neighbors_without_a_suite_maps_every_taxonomy(served: dict[str, Any]) -> None:
    payload = served["map_all"]

    assert payload["suite"] == "all"
    assert {k: v["status"] for k, v in payload["suites"].items()} == {
        "fake": "mapped",
        "locate_only": "not_found",
        "broken": "error",
        "down": "error",
    }
    assert {n["suite"] for n in payload["nodes"]} == {"fake"}
    assert payload["nodes"][0]["cite"] == DEV.source_id
    assert payload["summary"]["suites_mapped"] == 1
    assert served["map_subset"]["warnings"][0].startswith("unknown_suite:")


def test_paths_carry_routes_and_pruning_counts(served: dict[str, Any]) -> None:
    assert served["paths"]["paths"][0]["node_ids"] == [DEV.id, PYTHON.id]
    assert served["paths"]["pruning"] == {"considered": 3, "returned": 1, "pruned": 2}


def test_scored_paths_pass_the_suite_refusal_through(served: dict[str, Any]) -> None:
    assert served["scored"]["policy"] == {"name": "essential-first", "version": "0.1.0"}
    assert "score_paths_not_implemented" in served["scored"]["warnings"]
    assert served["scored"]["scored_paths"] == []


@pytest.mark.parametrize(
    ("key", "warning"),
    [
        ("miss", "not_found"),
        ("neighbors_missing", "node_not_found"),
        ("unknown_suite", "unknown_suite:bls"),
        ("suite_down", "suite_unavailable:ConnectionRefusedError"),
        ("suite_broken", "suite_unavailable:RuntimeError"),
        ("no_pathfind", "capability_unavailable:enumerate_paths:locate_only"),
        ("invalid_depth", "invalid_max_depth"),
    ],
)
def test_every_failure_is_a_machine_readable_warning(
    served: dict[str, Any], key: str, warning: str
) -> None:
    payload = served[key]

    assert warning in payload["warnings"]
    assert payload.get("candidates", []) == []
    assert payload.get("nodes", []) == []


def test_an_unavailable_suite_never_leaks_the_driver_message(
    served: dict[str, Any],
) -> None:
    # Driver errors quote the connection URI, and this text lands in a chat log.
    assert served["suite_down"]["warnings"] == ["suite_unavailable:ConnectionRefusedError"]


def test_a_suite_call_failure_never_leaks_its_error_message(served: dict[str, Any]) -> None:
    assert served["suite_broken"]["warnings"] == ["suite_unavailable:RuntimeError"]


def test_neighbours_are_bounded_and_the_cut_is_counted() -> None:
    """Unbounded, one O*NET hop was ~1.4 MB of tool output."""
    from talent_angels.mcp.mapping import neighbors_payload
    from tests.fakes.taxonomy import FakeEdge, FakeNode, FakeToolResult

    center = FakeNode(id="c", kind="Occupation", label="dev", source="onet", source_id="c")
    skills = [
        FakeNode(id=f"s{i}", kind="Skill", label=f"skill {i}", source="onet", source_id=f"s{i}")
        for i in range(10)
    ]
    edges = [
        FakeEdge(type="HAS_SKILL", from_id="c", to_id=f"s{i}", properties={"importance": float(i)})
        for i in range(10)
    ]
    payload = neighbors_payload(
        FakeToolResult(nodes=[center, *skills], edges=edges), suite="onet", center_id="c", limit=3
    )

    assert payload.pruning is not None
    assert (payload.pruning.considered, payload.pruning.returned, payload.pruning.pruned) == (
        10,
        3,
        7,
    )
    assert [n.id for n in payload.nodes] == ["c", "s9", "s8", "s7"]
    assert "truncated:7" in payload.warnings
