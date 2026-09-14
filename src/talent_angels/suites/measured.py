"""Measure concrete taxonomy tool calls without coupling skills to telemetry."""

from __future__ import annotations

import time

from talent_angels.runlog import ToolCall
from talent_angels.skills.connect.reveal import NeighborResult
from talent_angels.skills.locate.resolve import SearchResult
from talent_angels.suites.protocol import SuiteTools


class MeasuredSuite:
    """Transparent current-suite wrapper that records actual call durations."""

    def __init__(self, suite: SuiteTools) -> None:
        self._suite = suite
        self.tool_calls: list[ToolCall] = []

    def search_nodes(self, text: str, kind: str | None = None) -> SearchResult:
        start = time.perf_counter()
        ok = False
        try:
            result = self._suite.search_nodes(text, kind=kind)
            ok = True
            return result
        finally:
            self.tool_calls.append(
                ToolCall(
                    name="search_nodes",
                    ms=(time.perf_counter() - start) * 1000,
                    ok=ok,
                    args={"text": text, "kind": kind},
                )
            )

    def get_neighbors(self, node_id: str, rel_types: list[str] | None = None) -> NeighborResult:
        start = time.perf_counter()
        ok = False
        try:
            result = self._suite.get_neighbors(node_id, rel_types=rel_types)
            ok = True
            return result
        finally:
            self.tool_calls.append(
                ToolCall(
                    name="get_neighbors",
                    ms=(time.perf_counter() - start) * 1000,
                    ok=ok,
                    args={"node_id": node_id, "rel_types": rel_types},
                )
            )

    def enumerate_paths(
        self,
        from_id: str,
        to_id: str,
        *,
        max_depth: int = 4,
        max_paths: int = 20,
    ) -> object:
        inner = getattr(self._suite, "enumerate_paths", None)
        if inner is None:
            raise AttributeError("enumerate_paths")
        start = time.perf_counter()
        ok = False
        try:
            result = inner(from_id, to_id, max_depth=max_depth, max_paths=max_paths)
            ok = True
            return result
        finally:
            self.tool_calls.append(
                ToolCall(
                    name="enumerate_paths",
                    ms=(time.perf_counter() - start) * 1000,
                    ok=ok,
                    args={"from_id": from_id, "to_id": to_id, "max_depth": max_depth},
                )
            )

    def score_paths(self, paths: object, policy: object) -> object:
        inner = getattr(self._suite, "score_paths", None)
        if inner is None:
            raise AttributeError("score_paths")
        start = time.perf_counter()
        ok = False
        try:
            result = inner(paths, policy)
            ok = True
            return result
        finally:
            self.tool_calls.append(
                ToolCall(
                    name="score_paths",
                    ms=(time.perf_counter() - start) * 1000,
                    ok=ok,
                    args={"policy": getattr(policy, "name", str(policy))},
                )
            )
