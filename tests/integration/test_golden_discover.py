"""Golden discovery map on the live taxonomies: structure, never-guessed items, determinism.

Skipped when Neo4j isn't reachable. Values: tests/evals/golden_discover.json.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

pytest.importorskip(
    "ta_taxonomies",
    reason="TA-taxonomies is not installed; install the sibling package for integration tests",
)

from talent_angels.cli import main  # noqa: E402
from tests.integration.support import neo4j_reachable  # noqa: E402

ROOT = Path(__file__).parents[2]
GOLDEN = json.loads((ROOT / "tests" / "evals" / "golden_discover.json").read_text(encoding="utf-8"))

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not neo4j_reachable(), reason="Neo4j is not reachable"),
]


def _run(out: Path, *extra: str) -> Path:
    assert main(["discover", str(ROOT / GOLDEN["input"]), "--out", str(out), *extra]) == 0
    return out


@pytest.fixture(scope="module")
def runs(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    base = tmp_path_factory.mktemp("discover")
    first = _run(base / "one")
    return {
        "first": first,
        "second": _run(base / "two"),
        # The `resolved` block of the first run replaces resolution on the third.
        "from_nodes": _run(base / "three", "--nodes", str(first / "discovery.json")),
    }


def _data(runs: dict[str, Path]) -> dict:
    return json.loads((runs["first"] / "discovery.json").read_text(encoding="utf-8"))


def test_two_runs_and_a_rerun_from_the_recorded_nodes_are_byte_identical(
    runs: dict[str, Path],
) -> None:
    files = [(folder / "discovery.json").read_bytes() for folder in runs.values()]
    assert files[0] == files[1] == files[2]
    htmls = [(folder / "discovery.html").read_bytes() for folder in runs.values()]
    assert htmls[0] == htmls[1] == htmls[2]


def test_the_three_files_exist_and_the_html_is_static(runs: dict[str, Path]) -> None:
    folder = runs["first"]
    assert {p.name for p in folder.iterdir()} == {
        "discovery.json",
        "discovery.xlsx",
        "discovery.html",
    }
    html = (folder / "discovery.html").read_text(encoding="utf-8")
    assert "<script" not in html and "src=" not in html
    assert all(h.startswith("http") for h in re.findall(r'href="([^"]*)"', html))  # cites only


def test_structure_and_never_guessed_items(runs: dict[str, Path]) -> None:
    data = _data(runs)
    assert data["policy"] == "discover-v1" and data["meaning"].endswith("not a learning order")
    items = {i["item"]: i for i in data["items"]}
    assert len(items) == 25 and list(items) == sorted(items, key=str.casefold)
    for name, suites in GOLDEN["expect_resolved_in"].items():
        assert set(suites) <= set(items[name]["resolved_in"]), name
    for name in GOLDEN["expect_never_resolved"]:
        assert items[name]["resolved_in"] == [], name
    for name in GOLDEN["expect_not_in_any_taxonomy"]:
        assert items[name]["not_in_any_taxonomy"], name
    for key, allowed in GOLDEN["expect_guess"].items():
        item, other = key.split(" -> ")
        verdict = next(g for g in data["guesses"] if (g["item"], g["related_to"]) == (item, other))
        assert verdict["verdict"] in allowed, key
    for key, minimum in GOLDEN["expect_min"].items():
        assert data["summary"][key] >= minimum, key


def test_every_node_is_cited_and_every_edge_and_link_stays_inside_the_map(
    runs: dict[str, Path],
) -> None:
    data = _data(runs)
    ids = {n["id"] for n in data["nodes"]}
    assert all(n["suite"] and n["cite"] for n in data["nodes"])
    assert all(e["from_id"] in ids and e["to_id"] in ids for e in data["edges"])
    assert all(x["a_id"] in ids and x["b_id"] in ids for x in data["links"])
    assert all(c["id"] in ids and c["item_count"] >= 2 for c in data["connectors"])
    order = [(c["tool_only"], c["broad"], -c["item_count"]) for c in data["connectors"]]
    assert order == sorted(order)  # specific, then broad, then tool-only; most items first
    flags = {c["label"]: c["tool_only"] for c in data["connectors"]}
    assert all(flags[label] for label in GOLDEN["expect_tool_only"] if label in flags)
    assert not any(flags[label] for label in GOLDEN["expect_not_tool_only"] if label in flags)
    assert all(g["verdict"] != "confirmed" or data["links"] for g in data["guesses"])
    assert all(r["edges"] >= 1 for r in data["routes"])
    assert {s["status"] for s in data["suites"].values()} == {"ok"}
