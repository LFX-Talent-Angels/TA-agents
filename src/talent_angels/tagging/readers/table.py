"""Flat-table reader: a client's CSV/XLSX export of id/parent/text/type columns.

Any business whose input is a spreadsheet rather than a Neo4j tree export can
use this reader directly — e.g. with `--profile generic` (every row is an
"item") — or keep its own kind-heading profile, since `select.py` keys a
row's kind on its own *text*, never on which reader produced it. Column names
are matched by alias, case-insensitively (English and Spanish), so most
exports need no `tag.yaml` at all; `<project>/tag.yaml`'s `table:` key
overrides the column names (and, for `.xlsx`, the sheet) explicitly.
"""

from __future__ import annotations

import csv
import io
import re
from pathlib import Path

import yaml
from openpyxl import load_workbook

from talent_angels.tagging.models import Document, SourceRow
from talent_angels.tagging.readers.base import ReaderError

_ID_ALIASES = {"id", "row_id", "row id", "code", "#", "no", "número", "numero"}
_PARENT_ALIASES = {"parent", "parent_id", "parent id", "padre"}
_TEXT_ALIASES = {
    "text",
    "description",
    "descripción",
    "descripcion",
    "content",
    "contenido",
    "statement",
}
_TYPE_ALIASES = {"type", "kind", "tipo", "label"}

_ROOT_ID = "root"
_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _slug(text: str, *, max_len: int = 60) -> str:
    slug = _SLUG_RE.sub("-", text.strip().lower()).strip("-")
    return slug[:max_len].rstrip("-")


def _norm(value: str) -> str:
    return " ".join(str(value).strip().casefold().split())


def _table_overrides(path: Path, *, strict: bool = False) -> dict[str, str]:
    """`<project>/tag.yaml`'s `table:` key — checked at both the `inputs/` layout's
    project root (`path.parent.parent`) and the bare-folder layout's (`path.parent`).

    `can_read` must never raise on a malformed file (Reader protocol), so it
    reads with `strict=False`; `read()` passes `strict=True` and reports a
    broken `tag.yaml` instead of silently ignoring the column overrides.
    """
    for candidate in (path.parent / "tag.yaml", path.parent.parent / "tag.yaml"):
        if not candidate.is_file():
            continue
        try:
            data = yaml.safe_load(candidate.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            if strict:
                raise ReaderError(f"{candidate}: invalid YAML ({exc})") from exc
            return {}
        if data is None:
            return {}
        if not isinstance(data, dict):
            if strict:
                raise ReaderError(f"{candidate}: expected a mapping at the top level")
            return {}
        table = data.get("table")
        if table is None:
            return {}
        if not isinstance(table, dict):
            if strict:
                raise ReaderError(f"{candidate}: 'table' must be a mapping")
            return {}
        return {str(k): str(v) for k, v in table.items()}
    return {}


def _check_no_cycles(parent_of: dict[str, str], path: Path) -> None:
    """Walk every parent chain once; a revisited row inside a chain is a cycle.

    `select.py` walks the same chains, so a cyclic table would hang the whole
    run (or blow the recursion limit) instead of failing with a message.
    """
    resolved: set[str] = set()
    for start in parent_of:
        chain: list[str] = []
        node: str | None = start
        while node is not None and node not in resolved:
            if node in chain:
                cycle = [*chain[chain.index(node) :], node]
                raise ReaderError(f"{path}: cycle in parent chain: {' -> '.join(cycle)}")
            chain.append(node)
            node = parent_of.get(node)
        resolved.update(chain)


def _find_column(
    headers: list[str], role: str, aliases: set[str], overrides: dict[str, str]
) -> int | None:
    override = overrides.get(role)
    if override is not None:
        target = _norm(override)
        return next((i for i, header in enumerate(headers) if _norm(header) == target), None)
    return next((i for i, header in enumerate(headers) if _norm(header) in aliases), None)


def _cell(row: list[str], col: int | None) -> str:
    if col is None or col >= len(row):
        return ""
    value = row[col]
    return value.strip() if value else ""


def _sniff_delimiter(sample: str) -> str:
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t").delimiter
    except csv.Error:
        return ","


def _read_csv_rows(path: Path) -> list[list[str]]:
    text = path.read_text(encoding="utf-8-sig")
    if not text.strip():
        return []
    delimiter = _sniff_delimiter(text.splitlines()[0])
    reader = csv.reader(io.StringIO(text), delimiter=delimiter)
    return [row for row in reader if any(cell.strip() for cell in row)]


def _read_xlsx_rows(path: Path, overrides: dict[str, str]) -> list[list[str]]:
    workbook = load_workbook(path, read_only=True, data_only=True)
    try:
        sheet_name = overrides.get("sheet")
        sheet = (
            workbook[sheet_name]
            if sheet_name and sheet_name in workbook.sheetnames
            else workbook.worksheets[0]
        )
        rows: list[list[str]] = []
        for row in sheet.iter_rows(values_only=True):
            cells = ["" if value is None else str(value) for value in row]
            if any(cell.strip() for cell in cells):
                rows.append(cells)
        return rows
    finally:
        workbook.close()


def _rows_for(path: Path, overrides: dict[str, str]) -> list[list[str]] | None:
    suffix = path.suffix.lower()
    try:
        if suffix == ".csv":
            return _read_csv_rows(path)
        if suffix == ".xlsx":
            return _read_xlsx_rows(path, overrides)
    except (OSError, UnicodeDecodeError, csv.Error, KeyError, ValueError):
        return None
    return None


class TableReader:
    """Reads a flat id/parent/text/type table (CSV or XLSX) into a `Document`.

    No id column: ids are "1", "2", ... in row order. No parent column (or a
    blank parent cell): the row's parent is a synthetic root (`row_id="root"`,
    text = the file's stem) — so a flat list with no hierarchy at all still
    reads as a one-level tree, same shape `select.py` already expects.
    """

    name = "table"

    def can_read(self, path: Path) -> bool:
        if path.suffix.lower() not in (".csv", ".xlsx"):
            return False
        overrides = _table_overrides(path)
        rows = _rows_for(path, overrides)
        if not rows:
            return False
        return _find_column(rows[0], "text", _TEXT_ALIASES, overrides) is not None

    def read(self, path: Path) -> Document:
        overrides = _table_overrides(path, strict=True)
        rows = _rows_for(path, overrides)
        if not rows:
            raise ReaderError(f"{path}: empty or unreadable table")
        header, *data_rows = rows

        col_id = _find_column(header, "id", _ID_ALIASES, overrides)
        col_parent = _find_column(header, "parent", _PARENT_ALIASES, overrides)
        col_text = _find_column(header, "text", _TEXT_ALIASES, overrides)
        col_type = _find_column(header, "type", _TYPE_ALIASES, overrides)
        if col_text is None:
            raise ReaderError(f"{path}: no text/description column found")

        row_ids: set[str] = {_ROOT_ID}
        pending: list[tuple[str, str | None, str, str]] = []  # row_id, parent_id, text, label

        for index, row in enumerate(data_rows, start=1):
            text = _cell(row, col_text)
            if not text:
                continue
            row_id = _cell(row, col_id) or str(index)
            parent_id = _cell(row, col_parent) or None
            label = _cell(row, col_type)
            if row_id in row_ids:
                raise ReaderError(f"{path}: duplicate row id {row_id!r}")
            row_ids.add(row_id)
            pending.append((row_id, parent_id, text, label))

        for row_id, parent_id, _text, _label in pending:
            if parent_id is not None and parent_id not in row_ids:
                raise ReaderError(f"{path}: row {row_id} has an unknown parent {parent_id!r}")

        _check_no_cycles(
            {row_id: parent_id for row_id, parent_id, _t, _l in pending if parent_id is not None},
            path,
        )

        source_rows: list[SourceRow] = [
            SourceRow(row_id=_ROOT_ID, parent_id=None, label="root", text=path.stem, order=0)
        ]
        for order, (row_id, parent_id, text, label) in enumerate(pending, start=1):
            source_rows.append(
                SourceRow(
                    row_id=row_id,
                    parent_id=parent_id or _ROOT_ID,
                    label=label,
                    text=text,
                    order=order,
                )
            )

        key = _slug(path.stem) or "doc"
        return Document(key=key, path=str(path), source_format="table", rows=source_rows)
