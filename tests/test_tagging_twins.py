"""A CSV twin of the JSON input is checked and described, never dropped silently."""

from __future__ import annotations

import json
from pathlib import Path

from talent_angels.tagging.twins import describe_csv_twin


def _json(path: Path, rows: list[tuple[str, str]]) -> Path:
    data = [
        {
            "n": {"identity": i + 1, "labels": ["Item"], "properties": {"text": child}},
            "r": {"type": "IS_PART_OF", "start": i + 1, "end": 0},
            "m": {"identity": 0, "labels": ["Document"], "properties": {"text": parent}},
        }
        for i, (child, parent) in enumerate(rows)
    ]
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _csv(path: Path, rows: list[tuple[str, str]], extra: str = "") -> Path:
    header = "n,r,m" + (f",{extra}" if extra else "")
    lines = [header] + [
        f'"(:Item {{text: {child}}})",[:IS_PART_OF],"(:Document {{text: {parent}}})"'
        + (",x" if extra else "")
        for child, parent in rows
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


ROWS = [("Cleans the oven, and", "Doc"), ("Weighs flour", "Doc")]


def test_an_identical_twin_is_reported_as_nothing_lost(tmp_path: Path) -> None:
    message = describe_csv_twin(_csv(tmp_path / "a.csv", ROWS), _json(tmp_path / "a.json", ROWS))
    assert message == "a.csv: same 2 rows as a.json, nothing lost (using the JSON)"


def test_rows_only_in_the_csv_are_called_out(tmp_path: Path) -> None:
    csv_path = _csv(tmp_path / "a.csv", [*ROWS, ("Proofs dough", "Doc")])
    message = describe_csv_twin(csv_path, _json(tmp_path / "a.json", ROWS))
    assert "has 1 rows that a.json lacks; NOT used yet" in message


def test_extra_columns_are_called_out(tmp_path: Path) -> None:
    csv_path = _csv(tmp_path / "a.csv", ROWS, extra="level")
    message = describe_csv_twin(csv_path, _json(tmp_path / "a.json", ROWS))
    assert "1 extra columns (level)" in message and "NOT used yet" in message


def test_a_different_table_next_to_the_json_is_named(tmp_path: Path) -> None:
    (tmp_path / "a.csv").write_text("code,description\n1,Weighs flour\n", encoding="utf-8")
    message = describe_csv_twin(tmp_path / "a.csv", _json(tmp_path / "a.json", ROWS))
    assert "a table with columns code, description" in message and "NOT used yet" in message
