"""Discovery map input readers and the `ta-agent discover` command, offline."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from talent_angels.cli import main
from talent_angels.skills.connect.discover_io import read_items
from tests.test_discover import _registry

ROWS = "item,kind,related_to\nPython,skill,machine learning; SQL\nweb developer,Role,Python\n"


def _write(tmp_path: Path, name: str, text: str) -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def test_txt_is_one_item_per_line_blank_and_duplicates_dropped_with_notes(tmp_path: Path) -> None:
    items, notes = read_items(_write(tmp_path, "a.txt", "Python\n\nSQL\npython\n  \nsql\n"))
    assert [i.text for i in items] == ["Python", "SQL"]  # the kept spelling is the first sorted
    assert notes == [
        "blank item skipped",
        "blank item skipped",
        "duplicate dropped: python = python",
        "duplicate dropped: sql = sql",
    ]


def test_csv_reads_the_item_kind_and_related_to_columns(tmp_path: Path) -> None:
    items, notes = read_items(_write(tmp_path, "a.csv", ROWS))
    assert [(i.text, i.kind, i.related_to) for i in items] == [
        ("Python", "skill", ("machine learning", "SQL")),
        ("web developer", "role", ("Python",)),
    ]
    assert notes == []


def test_csv_without_an_item_column_uses_the_first_column_and_every_row(tmp_path: Path) -> None:
    items, _ = read_items(_write(tmp_path, "a.csv", "Skills,Notes\nPython,x\nSQL,y\n"))
    assert [i.text for i in items] == ["Python", "Skills", "SQL"]


def test_xlsx_json_and_unknown_kinds(tmp_path: Path) -> None:
    from openpyxl import Workbook

    book = Workbook()
    sheet = book.active
    sheet.append(["Notes", "ITEM", "Kind"])
    sheet.append(["n", "Python", "skill"])
    sheet.append(["n", "Linux", "gadget"])
    book.save(tmp_path / "a.xlsx")
    items, notes = read_items(tmp_path / "a.xlsx")
    assert [(i.text, i.kind) for i in items] == [("Linux", None), ("Python", "skill")]
    assert notes == ["unknown kind 'gadget' for 'Linux' ignored"]

    data = ["SQL", {"item": "Python", "kind": "skill", "related_to": ["SQL", "Linux"]}]
    items, _ = read_items(_write(tmp_path, "a.json", json.dumps(data)))
    assert [(i.text, i.related_to) for i in items] == [("Python", ("Linux", "SQL")), ("SQL", ())]


def test_bad_inputs_are_errors(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unsupported"):
        read_items(_write(tmp_path, "a.doc", "x"))
    with pytest.raises(ValueError, match="list"):
        read_items(_write(tmp_path, "a.json", '{"item": "x"}'))
    with pytest.raises(ValueError, match="at most 100"):
        read_items(_write(tmp_path, "big.txt", "\n".join(f"item {n}" for n in range(101))))


def _cli(tmp_path: Path, *extra: str) -> int:
    source = _write(
        tmp_path, "list.csv", "item,related_to\nweb developer,JavaScript\nJavaScript,\nteamwork,\n"
    )
    return main(["discover", str(source), "--depth", "1", *extra], registry=_registry())


def test_discover_writes_three_files_and_prints_a_summary(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _cli(tmp_path, "--out", str(tmp_path / "out")) == 0
    out = capsys.readouterr().out
    assert "3 items: 3 found" in out and "1 confirmed" in out
    assert sorted(p.name for p in (tmp_path / "out").iterdir()) == [
        "discovery.html", "discovery.json", "discovery.xlsx",
    ]  # fmt: skip
    data = json.loads((tmp_path / "out" / "discovery.json").read_text(encoding="utf-8"))
    assert data["policy"] == "discover-v1" and data["orphans"][0]["item"] == "teamwork"


def test_the_default_output_folder_is_next_to_the_input(tmp_path: Path) -> None:
    assert _cli(tmp_path) == 0
    assert (tmp_path / "discovery.json").exists()


def test_nodes_rerun_is_byte_identical_and_survives_a_hand_edit(tmp_path: Path) -> None:
    assert _cli(tmp_path, "--out", str(tmp_path / "one")) == 0
    first = (tmp_path / "one" / "discovery.json").read_bytes()
    nodes = tmp_path / "one" / "discovery.json"
    assert _cli(tmp_path, "--out", str(tmp_path / "two"), "--nodes", str(nodes)) == 0
    assert (tmp_path / "two" / "discovery.json").read_bytes() == first
    # A bare list of entries works too, and a non-list is refused.
    bare = _write(tmp_path, "bare.json", json.dumps(json.loads(first)["resolved"]))
    assert _cli(tmp_path, "--out", str(tmp_path / "three"), "--nodes", str(bare)) == 0
    assert (tmp_path / "three" / "discovery.json").read_bytes() == first
    assert _cli(tmp_path, "--nodes", str(_write(tmp_path, "bad.json", '{"x": 1}'))) == 2


def test_input_errors_exit_two_with_a_message(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(["discover", str(tmp_path / "missing.csv")], registry=_registry())
    assert code == 2 and "discover:" in capsys.readouterr().err
    assert _cli(tmp_path, "--suites", "nope") == 2


def test_ta_agent_routes_discover_to_the_cli(tmp_path: Path) -> None:
    from talent_angels.tui.app import main as tui_main

    source = _write(tmp_path, "list.txt", "web developer\nJavaScript\n")
    assert tui_main(["discover", str(source), "--depth", "1"], registry=_registry()) == 0
    assert (tmp_path / "discovery.json").exists()
