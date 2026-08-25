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
            await call("miss", "search_nodes", {"text": "stonemason"})
            await call("neighbors", "get_neighbors", {"node_id": DEV.id})
            await call("neighbors_missing", "get_neighbors", {"node_id": "fake:occupation:nope"})
            await call("paths", "enumerate_paths", {"from_id": DEV.id, "to_id": PYTHON.id})
            await call(
                "invalid_depth",
                "enumerate_paths",
                {"from_id": DEV.id, "to_id": PYTHON.id, "max_depth": 0},
            )
            await call("unknown_suite", "search_nodes", {"text": "x", "suite": "onet"})
            await call("suite_down", "search_nodes", {"text": "x", "suite": "down"})
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


def test_the_server_exposes_the_four_contract_tools(served: dict[str, Any]) -> None:
    assert served["server_name"] == SERVER_NAME
    assert set(served["tools"]) == {
        "search_nodes",
        "get_neighbors",
        "enumerate_paths",
        "score_paths",
    }


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
        "get_neighbors": {"node_id", "rel_types", "suite"},
        "enumerate_paths": {"from_id", "to_id", "max_depth", "max_paths", "suite"},
        "score_paths": {"paths", "policy_name", "policy_version", "suite"},
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
        ("unknown_suite", "unknown_suite:onet"),
        ("suite_down", "suite_unavailable:ConnectionRefusedError"),
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
