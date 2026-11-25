"""Discovery map input: a messy list of roles, tasks and skills read from .txt/.csv/.xlsx/.json.

Canonical on the way in: items are de-duplicated case-insensitively and sorted, so the
order of the rows in the file never reaches the output.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MAX_ITEMS = 100
ITEM_KINDS = ("role", "occupation", "task", "skill")  # recorded, never used to filter


@dataclass(frozen=True)
class Item:
    text: str
    kind: str | None = None  # the user's own label; suites name kinds differently
    related_to: tuple[str, ...] = ()  # the user's guesses: "this goes with those"

    @property
    def key(self) -> str:
        return self.text.casefold()


def _split(value: object) -> list[str]:
    parts = value if isinstance(value, list | tuple) else str(value or "").split(";")
    return [p for part in parts if (p := str(part).strip())]


def build_items(rows: Iterable[Mapping[str, Any]]) -> tuple[list[Item], list[str]]:
    """Rows of {item, kind?, related_to?} -> sorted unique items, and notes on drops."""
    notes: list[str] = []
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        text = str(row.get("item") or "").strip()
        if not text:
            notes.append("blank item skipped")
            continue
        groups.setdefault(text.casefold(), []).append({**row, "item": text})
    items: list[Item] = []
    for key, group in groups.items():
        group.sort(key=lambda r: r["item"])  # the spelling kept must not depend on row order
        if len(group) > 1:
            notes.append(f"duplicate dropped: {', '.join(r['item'] for r in group[1:])} = {key}")
        kinds = [k for r in group if (k := str(r.get("kind") or "").strip().casefold())]
        if kinds and kinds[0] not in ITEM_KINDS:
            notes.append(f"unknown kind {kinds[0]!r} for {group[0]['item']!r} ignored")
            kinds = []
        guesses = {
            g.casefold(): g
            for r in group
            for g in _split(r.get("related_to"))
            if g.casefold() != key
        }
        items.append(Item(group[0]["item"], kinds[0] if kinds else None, tuple(guesses.values())))
    if len(items) > MAX_ITEMS:
        raise ValueError(f"at most {MAX_ITEMS} items, got {len(items)}")
    items.sort(key=lambda i: i.key)
    return [
        Item(i.text, i.kind, tuple(sorted(i.related_to, key=lambda g: (g.casefold(), g))))
        for i in items
    ], sorted(notes)


def _table_rows(table: Sequence[Sequence[object]]) -> list[dict[str, Any]]:
    """`item` column by header, else the first column (and then every row is data)."""
    cells = [["" if c is None else str(c).strip() for c in row] for row in table]
    if not cells:
        return []
    head = [c.casefold() for c in cells[0]]
    if "item" not in head:
        return [{"item": row[0]} for row in cells if row]
    col = {name: head.index(name) for name in ("item", "kind", "related_to") if name in head}
    return [{name: row[i] if i < len(row) else "" for name, i in col.items()} for row in cells[1:]]


def read_items(path: str | Path) -> tuple[list[Item], list[str]]:
    path = Path(path)
    suffix = path.suffix.casefold()
    rows: list[dict[str, Any]]
    if suffix == ".txt":
        rows = [{"item": line} for line in path.read_text(encoding="utf-8-sig").splitlines()]
    elif suffix == ".csv":
        with path.open(newline="", encoding="utf-8-sig") as handle:
            rows = _table_rows(list(csv.reader(handle)))
    elif suffix in (".xlsx", ".xlsm"):
        from openpyxl import load_workbook

        book = load_workbook(path, read_only=True, data_only=True)
        try:
            rows = _table_rows(list(book.worksheets[0].iter_rows(values_only=True)))
        finally:
            book.close()
    elif suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        if not isinstance(data, list):
            raise ValueError("a .json input must be a list of strings or of objects")
        if not all(isinstance(d, str | dict) for d in data):
            raise ValueError("a .json input must be a list of strings or of objects")
        rows = [{"item": d} if isinstance(d, str) else d for d in data]
    else:
        raise ValueError(f"unsupported input {path.suffix!r}; use .txt, .csv, .xlsx or .json")
    return build_items(rows)


__all__ = ["ITEM_KINDS", "MAX_ITEMS", "Item", "build_items", "read_items"]
