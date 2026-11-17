"""Graph export reader: a standard exported as a meaning graph, not a document tree.

Two shapes, both seen from CONOCER's knowledge-graph work:

- Neo4j `{n, r, m}` records with no `IS_PART_OF` tree: `n -[r]-> m` where `m`
  is the child (`EC1691 -REQUIRES_PERFORMANCE-> D1691.1`). Text is the node's
  `text`, `name` or `title`; its type is a `type` property or its first label.
- An edge table (JSON list or CSV): origin id/type/name, relation, destination
  id/type/name, with English or Spanish column names (`Tipo_Origen`,
  `ID_Origen`, `Nombre_Origen`, `Relacion`, `Tipo_Destino` ...). A row with no
  origin hangs off the document itself.

Either way the graph becomes a tree, so `select.py` and everything after it
work unchanged. Each row keeps the relationship from its parent
(`SourceRow.relation`); the profile's `relations:` map turns that into a kind,
because a graph has no "PERFORMANCES" heading to read it from.

Graph to tree: the root is the one node nothing points to (or a synthetic
root named after the file when there are several). Each node's parent is the
one that reaches it first, breadth-first from the root, in file order: the
shallowest parent. A node with more parents (a knowledge block that the
standard organizes and three phases require) is tagged once, under that one.
"""

from __future__ import annotations

import csv
import json
import re
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from talent_angels.tagging.models import Document, SourceRow
from talent_angels.tagging.readers.base import ReaderError

_SLUG_RE = re.compile(r"[^a-z0-9]+")
_TREE_REL = "IS_PART_OF"
_NRM_KEYS = {"n", "r", "m"}
_TEXT_KEYS = ("text", "name", "title")

# Edge-table columns, matched case- and accent-insensitively after removing
# "_", "-" and spaces. Origin = the parent end of the edge.
_ORIGIN_ID = {"idorigen", "origenid", "sourceid", "fromid", "startid", "parentid"}
_ORIGIN_NAME = {"nombreorigen", "origenname", "sourcename", "fromname", "startname", "parentname"}
_ORIGIN_TYPE = {"tipoorigen", "origentype", "sourcetype", "fromtype", "starttype", "parenttype"}
_RELATION = {"relacion", "relation", "relationship", "reltype", "rel"}
_DEST_ID = {"iddestino", "destinoid", "targetid", "toid", "endid", "childid"}
_DEST_NAME = {"nombredestino", "destinoname", "targetname", "toname", "endname", "childname"}
_DEST_TYPE = {"tipodestino", "destinotype", "targettype", "totype", "endtype", "childtype"}
_ACCENTS = str.maketrans("áéíóúñ", "aeioun")
_MISSING = {"null", "none"}


def _slug(text: str, *, max_len: int = 60) -> str:
    slug = _SLUG_RE.sub("-", text.strip().lower()).strip("-")
    return slug[:max_len].rstrip("-")


def _col(name: str) -> str:
    return re.sub(r"[\s_\-]+", "", str(name).casefold().translate(_ACCENTS))


@dataclass(frozen=True)
class _Node:
    key: str
    text: str
    type: str


@dataclass(frozen=True)
class _Edge:
    parent: str | None  # None: the edge hangs off the document itself
    child: str
    relation: str


class _Graph:
    def __init__(self) -> None:
        self.nodes: dict[str, _Node] = {}
        self.edges: list[_Edge] = []

    def add_node(self, key: str, text: str, node_type: str) -> None:
        """Register a node; a later row may fill in a name or type an earlier one lacked."""
        known = self.nodes.get(key) or _Node(key, "", "")
        self.nodes[key] = _Node(key, known.text or text, known.type or node_type)


def _columns(keys: Any) -> dict[str, str] | None:
    """Edge-table column name per role, or None when the keys aren't an edge table."""
    by_norm = {_col(k): str(k) for k in keys}
    found: dict[str, str] = {}
    for role, aliases in (
        ("origin_id", _ORIGIN_ID),
        ("origin_name", _ORIGIN_NAME),
        ("origin_type", _ORIGIN_TYPE),
        ("relation", _RELATION),
        ("dest_id", _DEST_ID),
        ("dest_name", _DEST_NAME),
        ("dest_type", _DEST_TYPE),
    ):
        match = next((by_norm[a] for a in aliases if a in by_norm), None)
        if match is not None:
            found[role] = match
    if {"origin_id", "relation", "dest_id"} <= found.keys() and (
        "dest_name" in found or "origin_name" in found
    ):
        return found
    return None


def _cell(record: dict[str, Any], column: str | None) -> str:
    if column is None:
        return ""
    value = record.get(column)
    text = "" if value is None else str(value).strip()
    # Neo4j Browser writes a missing value as "null" in its CSV download.
    return "" if text.casefold() in _MISSING else text


def _load_records(path: Path) -> list[Any] | None:
    suffix = path.suffix.lower()
    try:
        if suffix == ".json":
            data = json.loads(path.read_text(encoding="utf-8"))
            return data if isinstance(data, list) else None
        if suffix == ".csv":
            with path.open(encoding="utf-8-sig", newline="") as handle:
                return list(csv.DictReader(handle))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, csv.Error):
        return None
    return None


def _is_graph_nrm(records: list[Any]) -> bool:
    """`{n, r, m}` records with no `IS_PART_OF` edge (those are tree exports)."""
    if not records or not all(isinstance(r, dict) and _NRM_KEYS <= r.keys() for r in records):
        return False
    return not any((r.get("r") or {}).get("type") == _TREE_REL for r in records)


def _edge_table_columns(records: list[Any]) -> dict[str, str] | None:
    if not records or not isinstance(records[0], dict):
        return None
    return _columns(records[0].keys())


def _nrm_node(node: Any, path: Path, index: int) -> _Node:
    if not isinstance(node, dict) or "identity" not in node:
        raise ReaderError(f"{path}: row {index} has a node without an identity")
    props = node.get("properties") or {}
    text = next((str(props[k]).strip() for k in _TEXT_KEYS if props.get(k)), "")
    labels = node.get("labels") or []
    node_type = str(props.get("type") or (labels[0] if labels else ""))
    return _Node(str(node["identity"]), text, node_type)


def _graph_from_nrm(records: list[Any], path: Path) -> _Graph:
    graph = _Graph()
    for i, record in enumerate(records):
        n, m = _nrm_node(record["n"], path, i), _nrm_node(record["m"], path, i)
        r = record.get("r") or {}
        for node in (n, m):
            graph.add_node(node.key, node.text, node.type)
        # The edge's own endpoints decide direction: start is the parent.
        try:
            start, end = str(r["start"]), str(r["end"])
        except (KeyError, TypeError) as exc:
            raise ReaderError(f"{path}: row {i} has a relationship without start/end") from exc
        if {start, end} != {n.key, m.key}:
            raise ReaderError(f"{path}: row {i}'s relationship does not join its n and m")
        graph.edges.append(_Edge(start, end, str(r.get("type") or "")))
    return graph


def _graph_from_table(records: list[Any], columns: dict[str, str], path: Path) -> _Graph:
    graph = _Graph()
    for i, record in enumerate(records):
        if not isinstance(record, dict):
            raise ReaderError(f"{path}: row {i} is not an object")
        child = _cell(record, columns["dest_id"])
        if not child:
            raise ReaderError(f"{path}: row {i} has no destination id")
        graph.add_node(
            child, _cell(record, columns.get("dest_name")), _cell(record, columns.get("dest_type"))
        )
        parent = _cell(record, columns["origin_id"]) or None
        if parent is not None:
            graph.add_node(
                parent,
                _cell(record, columns.get("origin_name")),
                _cell(record, columns.get("origin_type")),
            )
        graph.edges.append(_Edge(parent, child, _cell(record, columns["relation"])))
    return graph


_SYNTHETIC_ROOT = "root"


def _to_document(graph: _Graph, path: Path, source_format: str) -> Document:
    if not graph.edges:
        raise ReaderError(f"{path}: no relationships to build a document from")
    has_parent = {e.child for e in graph.edges if e.parent is not None and e.parent != e.child}
    roots = [key for key in graph.nodes if key not in has_parent]
    # Edges with no origin hang off the document: the single root if there is
    # one, else a synthetic root.
    orphans = {e.child for e in graph.edges if e.parent is None}
    real_roots = [key for key in roots if key not in orphans]
    outgoing: dict[str, list[_Edge]] = {}
    if len(real_roots) == 1:
        root = real_roots[0]
        root_text = graph.nodes[root].text
    else:
        root = _SYNTHETIC_ROOT
        while root in graph.nodes:
            root = f"_{root}"
        root_text = path.stem
        outgoing[root] = [_Edge(root, key, "") for key in real_roots]
    for edge in graph.edges:
        parent = edge.parent if edge.parent is not None else root
        outgoing.setdefault(parent, []).append(_Edge(parent, edge.child, edge.relation))

    # Breadth-first: each node's parent is the shallowest one, first in file order.
    parent_of: dict[str, tuple[str, str]] = {}
    seen = {root}
    queue = deque([root])
    while queue:
        current = queue.popleft()
        for edge in outgoing.get(current, []):
            if edge.child in seen:
                continue
            seen.add(edge.child)
            parent_of[edge.child] = (current, edge.relation)
            queue.append(edge.child)
    unreached = sorted(set(graph.nodes) - seen)
    if unreached:
        raise ReaderError(f"{path}: nodes not reachable from the root (a cycle?): {unreached}")

    children: dict[str, list[str]] = {}
    for child, (parent, _) in parent_of.items():
        children.setdefault(parent, []).append(child)

    rows: list[SourceRow] = []
    stack = [root]
    while stack:  # depth-first, so each line sits under its parent in the sheet
        key = stack.pop()
        parent_id, relation = parent_of.get(key, ("", ""))
        node = graph.nodes.get(key)
        rows.append(
            SourceRow(
                row_id=key,
                parent_id=parent_id or None,
                label=node.type if node else "Document",
                text=node.text if node else root_text,
                order=len(rows),
                relation=relation,
            )
        )
        stack.extend(reversed(children.get(key, [])))

    doc_key = _slug(root_text) or _slug(path.stem) or "doc"
    return Document(key=doc_key, path=str(path), source_format=source_format, rows=rows)


class GraphReader:
    """Reads a standard exported as a graph: `{n, r, m}` records or an edge table."""

    name = "graph"

    def can_read(self, path: Path) -> bool:
        records = _load_records(path)
        if records is None:
            return False
        if path.suffix.lower() == ".json" and _is_graph_nrm(records):
            return True
        return _edge_table_columns(records) is not None

    def read(self, path: Path) -> Document:
        records = _load_records(path)
        if not records:
            raise ReaderError(f"{path}: expected a non-empty list of graph rows")
        if path.suffix.lower() == ".json" and _is_graph_nrm(records):
            return _to_document(_graph_from_nrm(records, path), path, "neo4j-graph-json")
        columns = _edge_table_columns(records)
        if columns is None:
            raise ReaderError(
                f"{path}: not a {{n, r, m}} graph or an origin/relation/destination table"
            )
        return _to_document(_graph_from_table(records, columns, path), path, "edge-table")
