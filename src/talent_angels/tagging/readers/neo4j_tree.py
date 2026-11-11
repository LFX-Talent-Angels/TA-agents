"""Generic Neo4j `{n, r, m}` tree export reader (PILOT-PLAN.md "Input format").

`n` is the child, `m` the parent; labels (Document/Section/Element/Block/Item)
describe shape only — `select.py` decides a row's tagging kind from heading
*text*, never from these. Rows whose relationship isn't `rel_type` are ignored
rather than treated as an error: a tree export can carry other relationship
kinds alongside the one that defines parentage. Their endpoints are not
registered as tree nodes — a cross-reference to a node outside the tree would
otherwise show up as a phantom second root.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from talent_angels.tagging.models import Document, SourceRow
from talent_angels.tagging.readers.base import ReaderError

_SLUG_RE = re.compile(r"[^a-z0-9]+")
_REQUIRED_KEYS = {"n", "r", "m"}


def _slug(text: str, *, max_len: int = 60) -> str:
    slug = _SLUG_RE.sub("-", text.strip().lower()).strip("-")
    return slug[:max_len].rstrip("-")


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _check_no_cycles(parent_of: dict[int, int], path: Path) -> None:
    """Walk every parent chain once; a revisited in-progress node is a cycle."""
    status: dict[int, int] = {}  # 0 = visiting, 1 = resolved

    def visit(node: int, stack: list[int]) -> None:
        if status.get(node) == 1:
            return
        if status.get(node) == 0:
            start = stack.index(node)
            cycle = [*stack[start:], node]
            raise ReaderError(f"{path}: cycle in parent chain: {cycle}")
        status[node] = 0
        stack.append(node)
        parent = parent_of.get(node)
        if parent is not None:
            visit(parent, stack)
        stack.pop()
        status[node] = 1

    for node in parent_of:
        visit(node, [])


class Neo4jTreeReader:
    """Reads a Neo4j `IS_PART_OF` tree export (CONOCER's export shape)."""

    name = "neo4j-tree-json"

    def __init__(self, rel_type: str = "IS_PART_OF") -> None:
        self.rel_type = rel_type

    def can_read(self, path: Path) -> bool:
        if path.suffix.lower() != ".json":
            return False
        try:
            data = _load_json(path)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return False
        if not isinstance(data, list) or not data:
            return False
        first = data[0]
        return isinstance(first, dict) and _REQUIRED_KEYS <= first.keys()

    def read(self, path: Path) -> Document:
        try:
            data = _load_json(path)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ReaderError(f"{path}: not readable JSON ({exc})") from exc
        if not isinstance(data, list) or not data:
            raise ReaderError(f"{path}: expected a non-empty JSON list of {{n, r, m}} rows")

        nodes: dict[int, dict[str, Any]] = {}
        order: dict[int, int] = {}
        parent_of: dict[int, int] = {}
        next_order = 1  # 0 is reserved for the root

        for i, row in enumerate(data):
            if not isinstance(row, dict) or not _REQUIRED_KEYS <= row.keys():
                raise ReaderError(f"{path}: row {i} is not a valid {{n, r, m}} object")
            n, r, m = row["n"], row.get("r") or {}, row["m"]
            if r.get("type") != self.rel_type:
                # A different relationship kind: not parentage, not an error —
                # and its endpoints are not part of this tree, so they are not
                # registered (`implicit` would make an outside node a 2nd root).
                continue
            try:
                for node in (n, m):
                    nodes.setdefault(node["identity"], node)
                child_id = n["identity"]
            except (KeyError, TypeError) as exc:
                raise ReaderError(f"{path}: row {i} has a node without an identity") from exc
            if child_id not in order:
                order[child_id] = next_order
                next_order += 1

            # `r.start`/`r.end` are the edge's own endpoints — the authoritative
            # parentage, even if a corrupted export's embedded `n`/`m` payload
            # disagrees with them.
            try:
                edge_child, edge_parent = int(r["start"]), int(r["end"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ReaderError(f"{path}: row {i} has a relationship without start/end") from exc
            if edge_child in parent_of and parent_of[edge_child] != edge_parent:
                raise ReaderError(
                    f"{path}: node {edge_child} has more than one parent "
                    f"({parent_of[edge_child]} and {edge_parent})"
                )
            parent_of[edge_child] = edge_parent

        all_ids = set(nodes)
        roots = all_ids - set(parent_of)
        if len(roots) != 1:
            raise ReaderError(f"{path}: expected exactly one root node, found {sorted(roots)}")
        root_id = next(iter(roots))
        order[root_id] = 0

        for child_id, parent_id in parent_of.items():
            if parent_id not in nodes:
                raise ReaderError(f"{path}: node {child_id}'s parent {parent_id} does not exist")

        _check_no_cycles(parent_of, path)

        rows: list[SourceRow] = []
        for identity in sorted(all_ids, key=lambda nid: order[nid]):
            node = nodes[identity]
            labels = node.get("labels") or []
            text = str(node.get("properties", {}).get("text", "")).strip()
            rows.append(
                SourceRow(
                    row_id=str(identity),
                    parent_id=None if identity == root_id else str(parent_of[identity]),
                    label=labels[0] if labels else "",
                    text=text,
                    order=order[identity],
                )
            )

        root_text = str(nodes[root_id].get("properties", {}).get("text", "")).strip()
        key = _slug(root_text) or _slug(path.stem) or "doc"
        return Document(key=key, path=str(path), source_format="neo4j-tree-json", rows=rows)
