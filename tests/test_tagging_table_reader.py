"""`TableReader`: the flat CSV/XLSX id/parent/text/type table -> `Document` step."""

from __future__ import annotations

import csv
from pathlib import Path

import openpyxl
import pytest

from talent_angels.tagging.readers import READERS, read_document
from talent_angels.tagging.readers.base import ReaderError
from talent_angels.tagging.readers.table import TableReader


def _write_csv(path: Path, rows: list[list[str]], *, delimiter: str = ",") -> Path:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter=delimiter)
        writer.writerows(rows)
    return path


def _write_xlsx(path: Path, rows: list[list[str]]) -> Path:
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    assert sheet is not None
    for row in rows:
        sheet.append(row)
    workbook.save(path)
    return path


def test_registered_after_neo4j_tree_reader() -> None:
    names = [reader.name for reader in READERS]
    assert names.index("neo4j-tree-json") < names.index("table")


# --- comma/semicolon CSV ----------------------------------------------------


def test_reads_comma_delimited_csv(tmp_path: Path) -> None:
    path = _write_csv(
        tmp_path / "course.csv",
        [
            ["id", "parent", "text", "type"],
            ["1", "", "Front Office Operations", "course"],
            ["2", "1", "Guest check-in", "unit"],
        ],
    )
    doc = TableReader().read(path)
    assert doc.source_format == "table"
    by_id = doc.by_id()
    assert by_id["1"].text == "Front Office Operations"
    assert by_id["1"].parent_id == "root"
    assert by_id["2"].parent_id == "1"
    assert by_id["2"].label == "unit"


def test_reads_semicolon_delimited_csv(tmp_path: Path) -> None:
    path = _write_csv(
        tmp_path / "course.csv",
        [["id", "parent", "text"], ["1", "", "A course"]],
        delimiter=";",
    )
    doc = TableReader().read(path)
    assert [row.text for row in doc.rows if row.parent_id is not None] == ["A course"]


def test_reads_tab_delimited_csv(tmp_path: Path) -> None:
    path = _write_csv(
        tmp_path / "course.csv",
        [["id", "parent", "text"], ["1", "", "A course"]],
        delimiter="\t",
    )
    doc = TableReader().read(path)
    assert [row.text for row in doc.rows if row.parent_id is not None] == ["A course"]


# --- xlsx --------------------------------------------------------------


def test_reads_xlsx(tmp_path: Path) -> None:
    path = _write_xlsx(
        tmp_path / "course.xlsx",
        [["id", "parent", "text"], ["1", "", "A course"], ["2", "1", "A unit"]],
    )
    doc = TableReader().read(path)
    by_id = doc.by_id()
    assert by_id["1"].text == "A course"
    assert by_id["2"].parent_id == "1"


def test_xlsx_sheet_override(tmp_path: Path) -> None:
    path = tmp_path / "project" / "inputs" / "course.xlsx"
    path.parent.mkdir(parents=True)
    workbook = openpyxl.Workbook()
    default_sheet = workbook.active
    assert default_sheet is not None
    default_sheet.title = "Cover"
    default_sheet.append(["note"])
    data_sheet = workbook.create_sheet("Data")
    data_sheet.append(["id", "parent", "text"])
    data_sheet.append(["1", "", "A course"])
    workbook.save(path)
    (path.parent.parent / "tag.yaml").write_text("table:\n  sheet: Data\n")

    doc = TableReader().read(path)
    assert [row.text for row in doc.rows if row.parent_id is not None] == ["A course"]


# --- header aliases ------------------------------------------------------


def test_spanish_header_aliases(tmp_path: Path) -> None:
    path = _write_csv(
        tmp_path / "curso.csv",
        [
            ["código", "padre", "descripción", "tipo"],
            ["1", "", "Operaciones de front office", "curso"],
        ],
    )
    doc = TableReader().read(path)
    rows = [row for row in doc.rows if row.parent_id is not None]
    assert rows[0].text == "Operaciones de front office"
    assert rows[0].row_id == "1"
    assert rows[0].label == "curso"


def test_missing_text_column_can_read_false(tmp_path: Path) -> None:
    path = _write_csv(tmp_path / "data.csv", [["id", "parent"], ["1", ""]])
    assert TableReader().can_read(path) is False


def test_conocer_shaped_n_r_m_csv_is_not_readable(tmp_path: Path) -> None:
    path = _write_csv(tmp_path / "export.csv", [["n", "r", "m"], ["{}", "{}", "{}"]])
    assert TableReader().can_read(path) is False


# --- synthetic root / id, bad parent ----------------------------------------


def test_no_id_or_parent_columns_uses_synthetic_root(tmp_path: Path) -> None:
    path = _write_csv(tmp_path / "flat list.csv", [["text"], ["Do a thing"], ["Do another"]])
    doc = TableReader().read(path)
    roots = [row for row in doc.rows if row.parent_id is None]
    assert len(roots) == 1
    assert roots[0].row_id == "root"
    assert roots[0].text == "flat list"
    data_rows = [row for row in doc.rows if row.parent_id is not None]
    assert [row.row_id for row in data_rows] == ["1", "2"]
    assert all(row.parent_id == "root" for row in data_rows)


def test_bad_parent_is_a_reader_error(tmp_path: Path) -> None:
    path = _write_csv(
        tmp_path / "course.csv", [["id", "parent", "text"], ["2", "99", "orphan row"]]
    )
    with pytest.raises(ReaderError, match="unknown parent"):
        TableReader().read(path)


def test_blank_text_rows_are_skipped(tmp_path: Path) -> None:
    path = _write_csv(
        tmp_path / "course.csv",
        [["id", "parent", "text"], ["1", "", ""], ["2", "", "kept row"]],
    )
    doc = TableReader().read(path)
    data_rows = [row for row in doc.rows if row.parent_id is not None]
    assert [row.text for row in data_rows] == ["kept row"]


# --- tag.yaml column overrides -------------------------------------------


def test_tag_yaml_column_overrides(tmp_path: Path) -> None:
    project = tmp_path / "project"
    (project / "inputs").mkdir(parents=True)
    path = _write_csv(
        project / "inputs" / "export.csv",
        [["Identifier", "Container", "Body"], ["1", "", "custom column names"]],
    )
    (project / "tag.yaml").write_text(
        "table:\n  id: Identifier\n  parent: Container\n  text: Body\n"
    )
    doc = TableReader().read(path)
    data_rows = [row for row in doc.rows if row.parent_id is not None]
    assert data_rows[0].text == "custom column names"
    assert data_rows[0].row_id == "1"


def test_document_key_is_slug_of_file_stem(tmp_path: Path) -> None:
    path = _write_csv(tmp_path / "My Export File.csv", [["text"], ["a row"]])
    doc = TableReader().read(path)
    assert doc.key == "my-export-file"


def test_read_document_dispatches_to_table_reader(tmp_path: Path) -> None:
    path = _write_csv(tmp_path / "plain.csv", [["text"], ["a row"]])
    doc = read_document(path)
    assert doc.source_format == "table"
