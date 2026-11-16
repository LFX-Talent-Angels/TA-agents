"""`GraphReader`: standards exported as a meaning graph, not an `IS_PART_OF` tree.

Shapes match CONOCER's knowledge-graph exports: `{n, r, m}` records with
`Entity` nodes and the kind in the relationship, and an origin/relation/
destination edge table (Spanish columns; JSON, or CSV with "null" cells).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from talent_angels.tagging.cli import _parse_rows
from talent_angels.tagging.profile import load_profile
from talent_angels.tagging.readers import read_document
from talent_angels.tagging.readers.base import ReaderError
from talent_angels.tagging.readers.graph import GraphReader
from talent_angels.tagging.readers.neo4j_tree import Neo4jTreeReader
from talent_angels.tagging.select import select_units


def _node(identity: int, name: str, node_type: str | None = None) -> dict:
    props = {"name": name} | ({"type": node_type} if node_type else {})
    return {"identity": identity, "labels": ["Entity"], "properties": props}


def _edge(n: dict, rel: str, m: dict) -> dict:
    return {"n": n, "r": {"start": n["identity"], "end": m["identity"], "type": rel}, "m": m}


def _pilot_graph() -> list[dict]:
    body = _node(1, "CONOCER")
    std = _node(2, "EC1691: Development of educational materials using GenAI tools")
    occ = _node(3, "SINCO 2711: Teaching and pedagogical technicians", "Occupation")
    element = _node(4, "E5289: Design learning strategies through prompts")
    perf = _node(5, "D1691.1: Applies GenAI tools to design a learning strategy")
    know = _node(6, "K1691.1: Ethical considerations in GenAI", "CognitiveKnowledge")
    prod = _node(7, "P1691.1: Designed learning strategy document", "WorkProduct")
    att = _node(8, "A1691.1: Perseverance (iterating until the target is met)")
    return [
        _edge(body, "REGULATES_STANDARD", std),
        _edge(std, "MAPS_TO_OCCUPATION", occ),
        _edge(std, "CONTAINS_ELEMENT", element),
        _edge(element, "REQUIRES_PERFORMANCE", perf),
        _edge(element, "REQUIRES_KNOWLEDGE", know),
        _edge(element, "PRODUCES_PRODUCT", prod),
        _edge(element, "EXHIBITS_ATTITUDE", att),
    ]


_COLUMNS = [
    "Tipo_Origen",
    "ID_Origen",
    "Nombre_Origen",
    "Relacion",
    "Tipo_Destino",
    "ID_Destino",
    "Nombre_Destino",
]


def _table() -> list[dict]:
    rows = [
        (
            "Standard",
            "EC-1",
            "Consultor en Gobernanza",
            "ORGANIZES_KNOWLEDGE",
            "KnowledgeBlock",
            "B1",
            "Fundamentos",
        ),
        (
            "GovernancePhase",
            "FASE_A",
            "Fundamentación",
            "CONTAINS_PERFORMANCE",
            "Performance",
            "D01",
            "Delimitar el alcance",
        ),
        (
            "GovernancePhase",
            "FASE_A",
            "Fundamentación",
            "REQUIRES_BLOCK",
            "KnowledgeBlock",
            "B1",
            "Fundamentos",
        ),
        (
            "KnowledgeBlock",
            "B1",
            "Fundamentos",
            "COVERS_KNOWLEDGE",
            "CognitiveKnowledge",
            "K01",
            "Gobernanza de IA",
        ),
        (
            "KnowledgeBlock",
            "B1",
            "Fundamentos",
            "COVERS_KNOWLEDGE",
            "CognitiveKnowledge",
            "K02",
            "Marcos de referencia",
        ),
        (
            "Performance",
            "D01",
            "Delimitar el alcance",
            "DELIVERS_PRODUCT",
            "ProductEvidence",
            "EVD01",
            "Documento de alcance",
        ),
        (None, None, None, "HAS_PHASE", "GovernancePhase", "FASE_A", "Fundamentación"),
    ]
    return [dict(zip(_COLUMNS, row, strict=True)) for row in rows]


def _write_json(tmp_path: Path, data: list[dict], name: str = "export.json") -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _write_csv(tmp_path: Path, data: list[dict]) -> Path:
    path = tmp_path / "export.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_COLUMNS)
        writer.writeheader()
        for row in data:  # Neo4j Browser writes a missing value as "null"
            writer.writerow({k: "null" if v is None else v for k, v in row.items()})
    return path


def _units(path: Path) -> dict[str, tuple[str, str, str]]:
    units, _ = select_units(read_document(path), load_profile(None))
    return {u.row_id: (u.kind, u.role, u.clean_text) for u in units}


def test_tree_reader_leaves_a_graph_to_the_graph_reader(tmp_path: Path) -> None:
    path = _write_json(tmp_path, _pilot_graph())
    assert Neo4jTreeReader().can_read(path) is False
    assert GraphReader().can_read(path) is True
    assert read_document(path).source_format == "neo4j-graph-json"


def test_graph_kinds_come_from_relationships_and_codes_are_cleaned(tmp_path: Path) -> None:
    units = _units(_write_json(tmp_path, _pilot_graph()))
    assert units == {
        "3": ("occupation", "leaf", "Teaching and pedagogical technicians"),
        "5": ("performance", "leaf", "Applies GenAI tools to design a learning strategy"),
        "6": ("knowledge", "leaf", "Ethical considerations in GenAI"),
        "7": ("product", "leaf", "Designed learning strategy document"),
        "8": ("attitude", "leaf", "Perseverance (iterating until the target is met)"),
    }  # the standard and its element are context, not lines to tag


def test_graph_lines_keep_their_element_as_context(tmp_path: Path) -> None:
    doc = read_document(_write_json(tmp_path, _pilot_graph()))
    units, _ = select_units(doc, load_profile(None))
    perf = next(u for u in units if u.row_id == "5")
    assert perf.context_path == [
        "EC1691: Development of educational materials using GenAI tools",
        "E5289: Design learning strategies through prompts",
    ]


def test_edge_table_shallowest_parent_and_products_are_not_steps(tmp_path: Path) -> None:
    doc = read_document(_write_json(tmp_path, _table()))
    assert doc.source_format == "edge-table"
    assert doc.key == "consultor-en-gobernanza"
    parent = {row.row_id: row.parent_id for row in doc.rows}
    # B1 is organized by the standard and required by a phase: kept once, under the standard.
    assert parent["B1"] == "EC-1"
    # A row with no origin hangs off the document (the single root).
    assert parent["FASE_A"] == "EC-1"
    units = _units(_write_json(tmp_path, _table()))
    assert units["B1"][:2] == ("knowledge", "group")
    assert units["B1"][2] == "Fundamentos: Gobernanza de IA; Marcos de referencia"
    # D01's product child is a different kind: D01 stays a leaf, not a group.
    assert units["D01"] == ("performance", "leaf", "Delimitar el alcance")
    assert units["EVD01"] == ("product", "leaf", "Documento de alcance")
    assert "FASE_A" not in units and "EC-1" not in units


def test_edge_table_csv_with_null_cells_matches_json(tmp_path: Path) -> None:
    from_json = select_units(read_document(_write_json(tmp_path, _table())), load_profile(None))[0]
    from_csv = select_units(read_document(_write_csv(tmp_path, _table())), load_profile(None))[0]
    assert [u.unit_key for u in from_csv] == [u.unit_key for u in from_json]


def test_several_roots_get_a_synthetic_document_root(tmp_path: Path) -> None:
    a, b = _node(1, "Standard A"), _node(2, "Standard B")
    pa = _node(3, "Writes reports")
    pb = _node(4, "Reviews code")
    path = _write_json(
        tmp_path,
        [_edge(a, "REQUIRES_PERFORMANCE", pa), _edge(b, "REQUIRES_PERFORMANCE", pb)],
        name="two-standards.json",
    )
    doc = read_document(path)
    roots = [row for row in doc.rows if row.parent_id is None]
    assert len(roots) == 1 and roots[0].text == "two-standards"
    assert {row.row_id: row.parent_id for row in doc.rows}["1"] == roots[0].row_id
    assert set(_units(path)) == {"3", "4"}


def test_a_cycle_without_a_root_is_a_reader_error(tmp_path: Path) -> None:
    a, b = _node(1, "A"), _node(2, "B")
    path = _write_json(tmp_path, [_edge(a, "REQUIRES_PERFORMANCE", b), _edge(b, "LINKS", a)])
    with pytest.raises(ReaderError, match="not reachable"):
        read_document(path)


def test_tree_exports_never_take_a_kind_from_node_labels() -> None:
    profile = load_profile(None)
    assert profile.kind_of_relation("", "Performance") is None
    assert profile.kind_of_relation("HAS_PHASE", "ProductEvidence") == "product"
    assert profile.kind_of_relation("REQUIERE_DESEMPEÑO", "Entity") == "performance"


def test_code_cleaner_keeps_prose_before_a_colon() -> None:
    profile = load_profile(None)
    assert profile.clean("D-PR.1: Formulates prompts") == "Formulates prompts"
    assert profile.clean("Precision: Formulating clear instructions") == (
        "Precision: Formulating clear instructions"
    )


def test_rows_option_accepts_ids_with_dashes() -> None:
    assert _parse_rows("66-68,D-PR.1,EC-CGDIA-2025-V1U") == {
        "66",
        "67",
        "68",
        "D-PR.1",
        "EC-CGDIA-2025-V1U",
    }


def test_a_graph_job_title_is_searched_as_written(tmp_path: Path) -> None:
    from talent_angels.tagging.runner import _needs_rewrite
    from talent_angels.tagging.search import queries_for

    units, _ = select_units(
        read_document(_write_json(tmp_path, _pilot_graph())), load_profile(None)
    )
    title = next(u for u in units if u.kind == "occupation")
    perf = next(u for u in units if u.kind == "performance")
    assert queries_for(title) == ["Teaching and pedagogical technicians"]
    assert _needs_rewrite(title) is False
    assert _needs_rewrite(perf) is True and perf.seed_phrases == []
