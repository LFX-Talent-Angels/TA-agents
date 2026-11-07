"""Say what a skipped CSV "twin" of a JSON input holds, instead of asking for trust.

Neo4j's browser exports one query result twice, as JSON (with node ids) and as
CSV (text only). Only the JSON can rebuild the tree, so the CSV is not tagged —
but whether it carries anything the JSON lacks is checked and reported, so a
future export with extra columns or rows is never dropped silently.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

#: A Cypher-literal cell, `(:Item {text: ...})`, as Neo4j's CSV export writes nodes.
_NODE_TEXT_RE = re.compile(r"\{text: (.*)\}\)$", re.DOTALL)


def _json_triples(json_path: Path) -> list[tuple[str, str, str]] | None:
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
        return [
            (
                str(row["n"]["properties"].get("text", "")).strip(),
                str(row["r"].get("type", "")),
                str(row["m"]["properties"].get("text", "")).strip(),
            )
            for row in data
        ]
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _csv_triples(rows: list[list[str]]) -> list[tuple[str, str, str]] | None:
    triples: list[tuple[str, str, str]] = []
    for row in rows:
        if len(row) < 3:
            return None
        child = _NODE_TEXT_RE.search(row[0].strip())
        parent = _NODE_TEXT_RE.search(row[2].strip())
        if child is None or parent is None:
            return None
        relation = row[1].strip().strip("[]:")
        triples.append((child.group(1).strip(), relation, parent.group(1).strip()))
    return triples


def describe_csv_twin(csv_path: Path, json_path: Path) -> str:
    """One line for the run summary: is anything in the CSV missing from the JSON?"""
    name, twin = csv_path.name, json_path.name
    try:
        with csv_path.open(encoding="utf-8-sig", newline="") as handle:
            table = list(csv.reader(handle))
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        return f"{name}: not read ({type(exc).__name__}); using {twin}"
    if not table:
        return f"{name}: empty; using {twin}"

    header, body = [cell.strip().casefold() for cell in table[0]], table[1:]
    if header[:3] == ["n", "r", "m"]:
        csv_rows, json_rows = _csv_triples(body), _json_triples(json_path)
        if csv_rows is not None and json_rows is not None:
            extra_columns = len(header) - 3
            only_csv = len(set(csv_rows) - set(json_rows))
            if not only_csv and not extra_columns:
                return f"{name}: same {len(csv_rows)} rows as {twin}, nothing lost (using the JSON)"
            parts = []
            if only_csv:
                parts.append(f"{only_csv} rows")
            if extra_columns:
                parts.append(f"{extra_columns} extra columns ({', '.join(table[0][3:])})")
            return f"{name}: has {' and '.join(parts)} that {twin} lacks; NOT used yet"
    return (
        f"{name}: a table with columns {', '.join(table[0])} next to {twin}; "
        "NOT used yet (only the JSON is tagged)"
    )
