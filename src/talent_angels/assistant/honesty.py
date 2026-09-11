"""Plain checks on merged results. Not a judge and not a subagent."""

from __future__ import annotations

from collections.abc import Sequence

from talent_angels.contracts import AgentResult


def honesty_warnings(results: Sequence[AgentResult]) -> list[str]:
    """Return extra warnings; never rewrite graph facts."""
    found: list[str] = []
    for result in results:
        for node in result.nodes:
            if node.suite and result.suite and node.suite != result.suite:
                found.append(f"honesty:node_suite:{node.id}")
            expected = f"{result.suite}:"
            if result.suite and node.id and not node.id.startswith(expected):
                found.append(f"honesty:id_prefix:{node.id}")
    return found
