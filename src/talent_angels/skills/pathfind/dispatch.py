"""Pathfind dispatch: locate + connect both ends, then the gap and routes.

Everything Pathfind needs lives here, in code, called the same way Locate and
Connect are called (ADR-0008 decision 6) — never phrased or decided by a
prompt. Mirrors ``turn_graph.compare_one``'s shape: locate both titles,
connect each, and report which side needs a pick rather than guessing one.
"""

from __future__ import annotations

from talent_angels.contracts import AgentResult, EdgeRef, NodeRef
from talent_angels.runlog import ToolCall
from talent_angels.skills.connect import ConnectRequest, connect
from talent_angels.skills.locate import locate
from talent_angels.skills.locate.rank import group_and_sort_locate
from talent_angels.skills.pathfind.gap import compute_gap
from talent_angels.skills.pathfind.routes import find_routes, supports_enumerate_paths
from talent_angels.suites.measured import MeasuredSuite
from talent_angels.suites.protocol import SuiteTools
from talent_angels.suites.schema import SuiteSchema

CAPABILITY_PATHFIND = "pathfind"


def _locate_and_connect(
    measured: MeasuredSuite, suite_name: str, schema: SuiteSchema, subject: str, *, side: str
) -> tuple[NodeRef | None, AgentResult]:
    found = locate(measured, suite_name, subject, kind="occupation")
    located = group_and_sort_locate(
        measured,
        found,
        subject,
        suite_name=suite_name,
        group_rel_type=schema.group_rel_type,
        group_node_kinds=schema.group_node_kinds,
    )
    if not located.nodes or "ambiguous" in located.warnings:
        # Which side needs a pick — a pathfind never guesses one for the user.
        marked = located.model_copy(
            update={"warnings": [*located.warnings, f"pathfind_side:{side}"]}
        )
        return None, marked
    result = connect(
        measured,
        suite_name,
        located.nodes[0],
        request=ConnectRequest(subject=subject, rel_types=schema.skill_rel_types),
        confidence=located.confidence,
        locate_evidence=located.evidence,
        optional_rel_values=schema.optional_rel_values,
    )
    return located.nodes[0], result


def pathfind_one(
    suite: SuiteTools, suite_name: str, start_subject: str, target_subject: str
) -> tuple[AgentResult, list[ToolCall]]:
    """Locate+Connect both ends, then the skill gap and (ADR-0008) bridge routes.

    The skill gap is always computed. Bridge routes (``enumerate_paths``) are
    only attempted when the suite supports them — not every registered suite
    does (``MeasuredSuite`` docstring) — and a large gap is carried as the
    ``gap_too_large`` warning, never a refusal (ADR-0008 decision 5):
    Pathfind still answers with the real gap either way.
    """
    measured = MeasuredSuite(suite)
    schema = suite.suite_schema

    start_node, start_result = _locate_and_connect(
        measured, suite_name, schema, start_subject, side="start"
    )
    if start_node is None:
        miss = start_result.model_copy(update={"capability": CAPABILITY_PATHFIND})
        return miss, measured.tool_calls

    target_node, target_result = _locate_and_connect(
        measured, suite_name, schema, target_subject, side="target"
    )
    if target_node is None:
        miss = target_result.model_copy(update={"capability": CAPABILITY_PATHFIND})
        return miss, measured.tool_calls

    gap = compute_gap(start_result, target_result, optional_rel_values=schema.optional_rel_values)

    missing_ids = {node.id for node in (*gap.essential_missing, *gap.optional_missing)}
    gap_edges = [
        edge
        for edge in target_result.edges
        if edge.source_node_id in missing_ids or edge.target_node_id in missing_ids
    ]

    nodes: list[NodeRef] = [start_node, target_node, *gap.essential_missing, *gap.optional_missing]
    edges: list[EdgeRef] = list(gap_edges)
    seen_ids = {node.id for node in nodes}
    warnings = list(dict.fromkeys([*start_result.warnings, *target_result.warnings]))
    if gap.gap_too_large:
        warnings.append("gap_too_large")

    if supports_enumerate_paths(measured):
        routes = find_routes(measured, suite_name, start_node.id, target_node.id)
        for route in routes.routes:
            for node in route.nodes:
                if node.id not in seen_ids:
                    seen_ids.add(node.id)
                    nodes.append(node)
            edges.extend(route.edges)
        for warning in routes.warnings:
            if warning not in warnings:
                warnings.append(warning)

    evidence = [*start_result.evidence, *target_result.evidence]
    result = AgentResult(
        capability=CAPABILITY_PATHFIND,
        suite=suite_name,
        nodes=nodes,
        edges=edges,
        evidence=evidence,
        warnings=warnings,
    )
    return result, measured.tool_calls
