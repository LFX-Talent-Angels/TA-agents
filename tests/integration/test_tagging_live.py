"""Live `--no-llm` run against the real suites (PILOT-PLAN.md "Review loop").

Skipped (not failed) when Neo4j isn't reachable, same pattern as every other
integration test — see `tests/integration/support.py`. Uses the same
synthetic fixture as the offline e2e (`tests/test_tagging_run.py`), copied
into `tmp_path` so nothing here touches `data/local/sprint7/`.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import openpyxl
import pytest

pytest.importorskip(
    "ta_taxonomies",
    reason="TA-taxonomies is not installed; install the sibling package for integration tests",
)

from talent_angels.suites import default_suite_registry  # noqa: E402
from talent_angels.tagging.models import RunOptions  # noqa: E402
from talent_angels.tagging.runner import run  # noqa: E402
from tests.integration.support import neo4j_reachable  # noqa: E402

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not neo4j_reachable(), reason="Neo4j is not reachable"),
]

FIXTURE = Path(__file__).parents[1] / "fixtures" / "tagging" / "standard_small.json"


def test_no_llm_run_produces_all_four_sheets_with_real_suites(tmp_path: Path) -> None:
    project = tmp_path / "project"
    (project / "inputs").mkdir(parents=True)
    shutil.copy(FIXTURE, project / "inputs" / "standard.json")

    registry = default_suite_registry(neighbor_cache=False)
    summary = run(project, registry=registry, opts=RunOptions(no_llm=True))

    assert summary.units > 0
    [output] = summary.outputs
    workbook = openpyxl.load_workbook(output)
    assert workbook.sheetnames == ["Summary", "Review", "Standard", "Gaps"]

    # At least one unit got >= 1 candidate in every suite this registry has
    # attached: a `--no-llm` run only ever verdicts `review` on a hit (`gap`
    # means zero candidates), so some `review` rules out a false "n/a" or
    # "unavailable" for the whole run.
    for suite_name, counts in summary.counts.items():
        assert counts.get("review", 0) > 0, (
            f"{suite_name}: no candidates found for any unit in a --no-llm run"
        )
