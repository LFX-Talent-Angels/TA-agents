"""CLI tests — live against the ESCO fixture in Neo4j; skipped if unreachable."""

from __future__ import annotations

import json

import pytest
from ta_taxonomies.suites.esco.db import neo4j_driver

from talent_angels.cli import main


def _neo4j_reachable() -> bool:
    try:
        with neo4j_driver() as (driver, _database):
            driver.verify_connectivity()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _neo4j_reachable(), reason="Neo4j not reachable; see TA-taxonomies NOTES.md"
)


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
