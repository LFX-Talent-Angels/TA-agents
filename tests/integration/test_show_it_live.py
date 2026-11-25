"""The HTML pages from the real graph: one per page type, rendered twice, byte-identical."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytest.importorskip(
    "ta_taxonomies",
    reason="TA-taxonomies is not installed; install the sibling package for integration tests",
)

from talent_angels.cli import main  # noqa: E402
from tests.integration.support import neo4j_reachable  # noqa: E402

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not neo4j_reachable(), reason="Neo4j is not reachable"),
]

ROOT = Path(__file__).parents[2]
SAMPLE = ROOT / "tests" / "fixtures" / "discover_uca_sample.csv"

PAGES = {
    "map": ["map", "use object-oriented programming", "--suites", "esco", "--depth", "2",
            "--breadth", "4"],
    "route": ["map", "--route", "web developer", "software developer"],
    "successors": ["successors", "marketing principles"],
    "roadmap-ids": ["roadmap", "onet:occupation:15-2051.01", "onet:occupation:15-2051.00"],
    "roadmap-names": ["roadmap", "data analyst", "data scientist"],
}  # fmt: skip
MARKERS = {"map": "Graph map", "route": "Graph map", "successors": "What comes after",
           "roadmap-ids": "Foundations", "roadmap-names": "Foundations"}  # fmt: skip


def _page(argv: list[str], out: Path, capsys: pytest.CaptureFixture[str]) -> str:
    main([*argv, "--html", str(out)])
    capsys.readouterr()
    return out.read_text(encoding="utf-8")


@pytest.mark.parametrize("name", PAGES)
def test_each_page_type_renders_from_the_real_graph_and_is_identical_on_a_rerun(
    name: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    first = _page(PAGES[name], tmp_path / "one.html", capsys)
    second = _page(PAGES[name], tmp_path / "two.html", capsys)
    assert first == second and MARKERS[name] in first
    assert "prefers-color-scheme:dark" in first and not re.search(r"<script|<link|src=", first)
    if name == "route":
        assert '<path class="e-spine"' in first
    if name == "successors":
        assert '<path class="e-stated"' in first and '<path class="e-policy"' in first


def test_the_discovery_page_labels_connectors_and_greys_tool_only_ones(
    tmp_path: Path,
) -> None:
    assert main(["discover", str(SAMPLE), "--out", str(tmp_path / "a")]) == 0
    assert main(["discover", str(SAMPLE), "--out", str(tmp_path / "b")]) == 0
    html = (tmp_path / "a" / "discovery.html").read_text(encoding="utf-8")
    assert html == (tmp_path / "b" / "discovery.html").read_text(encoding="utf-8")
    assert "prefers-color-scheme:dark" in html and "Top connectors" in html
    assert "<tr class=dim><td>Aerospace Engineers</td>" in html
