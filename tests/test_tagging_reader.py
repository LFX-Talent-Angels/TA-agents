"""`Neo4jTreeReader`: the generic `{n, r, m}` tree export -> `Document` step."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from talent_angels.tagging.readers import READERS, read_document
from talent_angels.tagging.readers.base import ReaderError
from talent_angels.tagging.readers.neo4j_tree import Neo4jTreeReader

FIXTURE = Path(__file__).parent / "fixtures" / "tagging" / "standard_small.json"


def _node(identity: int, text: str, labels: tuple[str, ...] = ("Item",)) -> dict:
    return {
        "identity": identity,
        "labels": list(labels),
        "properties": {"text": text},
        "elementId": f"x:{identity}",
    }


def _rel(identity: int, start: int, end: int, rel_type: str = "IS_PART_OF") -> dict:
    return {
        "identity": 10_000 + identity,
        "start": start,
        "end": end,
        "type": rel_type,
        "properties": {},
        "elementId": f"r:{identity}",
    }


def _write(tmp_path: Path, rows: list[dict], name: str = "export.json") -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(rows))
    return path


def test_reads_the_synthetic_fixture() -> None:
    doc = read_document(FIXTURE)

    assert doc.source_format == "neo4j-tree-json"
    assert doc.path == str(FIXTURE)
    # One SourceRow per node, including the root; order starts at 0 (root).
    root_rows = [row for row in doc.rows if row.parent_id is None]
    assert len(root_rows) == 1
    assert root_rows[0].order == 0
    orders = [row.order for row in doc.rows]
    assert orders == sorted(orders)
    assert orders == list(range(len(doc.rows)))


def test_document_key_slugs_the_root_text() -> None:
    doc = read_document(FIXTURE)
    assert doc.key == "bakery-hygiene-and-food-safety-standard-json"


def test_document_key_falls_back_to_file_stem(tmp_path: Path) -> None:
    rows = [
        {"n": _node(1, "child"), "r": _rel(1, 1, 0), "m": _node(0, "   ", labels=("Document",))}
    ]
    path = _write(tmp_path, rows, name="My Export.json")
    doc = read_document(path)
    assert doc.key == "my-export"


def test_can_read_true_for_fixture() -> None:
    reader = READERS[0]
    assert reader.can_read(FIXTURE) is True


@pytest.mark.parametrize(
    "content",
    [
        "not json at all",
        "[]",
        "{}",
        json.dumps([{"foo": "bar"}]),
        json.dumps([1, 2, 3]),
    ],
)
def test_can_read_false_for_non_matching_json(tmp_path: Path, content: str) -> None:
    path = tmp_path / "other.json"
    path.write_text(content)
    assert Neo4jTreeReader().can_read(path) is False


def test_can_read_false_for_non_json_suffix(tmp_path: Path) -> None:
    path = tmp_path / "export.csv"
    path.write_text("a,b,c\n1,2,3\n")
    assert Neo4jTreeReader().can_read(path) is False


def test_ignores_rows_with_a_different_relationship_type() -> None:
    # The fixture's last row is a `REFERS_TO` edge between two existing nodes;
    # it must not be treated as parentage.
    doc = read_document(FIXTURE)
    by_id = doc.by_id()
    row_19 = by_id["19"]
    assert row_19.parent_id == "16"  # unchanged by the REFERS_TO noise row


def test_two_roots_is_a_reader_error(tmp_path: Path) -> None:
    rows = [
        {"n": _node(1, "A"), "r": _rel(1, 1, 0), "m": _node(0, "Root")},
        {"n": _node(2, "B"), "r": _rel(2, 2, 0, "OTHER"), "m": _node(0, "Root")},
    ]
    path = _write(tmp_path, rows)
    with pytest.raises(ReaderError):
        Neo4jTreeReader().read(path)


def test_missing_parent_is_a_reader_error(tmp_path: Path) -> None:
    # The edge's own `end` (99) disagrees with the embedded `m` node's identity
    # (5) — a corrupted export where the declared parent never exists.
    rows = [{"n": _node(1, "A"), "r": _rel(1, 1, 99), "m": _node(5, "Other")}]
    path = _write(tmp_path, rows)
    with pytest.raises(ReaderError, match="parent 99 does not exist"):
        Neo4jTreeReader().read(path)


def test_two_parents_is_a_reader_error(tmp_path: Path) -> None:
    rows = [
        {"n": _node(1, "A"), "r": _rel(1, 1, 0), "m": _node(0, "Root")},
        {"n": _node(1, "A"), "r": _rel(2, 1, 5), "m": _node(5, "Other")},
    ]
    path = _write(tmp_path, rows)
    with pytest.raises(ReaderError, match="more than one parent"):
        Neo4jTreeReader().read(path)


def test_cycle_is_a_reader_error(tmp_path: Path) -> None:
    rows = [
        {"n": _node(3, "C"), "r": _rel(0, 3, 0), "m": _node(0, "Root")},
        {"n": _node(1, "A"), "r": _rel(1, 1, 2), "m": _node(2, "B")},
        {"n": _node(2, "B"), "r": _rel(2, 2, 1), "m": _node(1, "A")},
    ]
    path = _write(tmp_path, rows)
    with pytest.raises(ReaderError, match="cycle"):
        Neo4jTreeReader().read(path)


def test_no_reader_for_unreadable_file(tmp_path: Path) -> None:
    path = tmp_path / "data.txt"
    path.write_text("nothing useful here")
    with pytest.raises(ReaderError, match="no reader for"):
        read_document(path)


def test_a_row_without_identity_is_a_reader_error(tmp_path) -> None:
    import json

    import pytest

    from talent_angels.tagging.readers.base import ReaderError
    from talent_angels.tagging.readers.neo4j_tree import Neo4jTreeReader

    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps([{"n": {"labels": []}, "r": {"type": "IS_PART_OF"}, "m": {}}]))
    with pytest.raises(ReaderError, match="without an identity"):
        Neo4jTreeReader().read(bad)
