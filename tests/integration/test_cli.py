"""CLI tests — live against the ESCO fixture in Neo4j; skipped if unreachable."""

from __future__ import annotations

import json

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


@pytest.fixture(autouse=True)
def _isolate_runlog(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setenv("RUNLOG_PATH", str(tmp_path / "runlog.jsonl"))
    monkeypatch.setenv("LLM_PROVIDER", "none")


def test_cli_locate_command_prints_json_result(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["locate", "software developer", "--kind", "occupation"])

    assert exit_code == 0
    output = json.loads(capsys.readouterr().out)
    assert output["capability"] == "locate"
    assert output["confidence"] == 0.95
    assert output["cost_usd"] == 0.0


def test_cli_query_command_routes_via_heuristic_intent(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["query", "software developer"])

    assert exit_code == 0
    output = json.loads(capsys.readouterr().out)
    assert output["capability"] == "locate"
    assert output["confidence"] == 0.95
