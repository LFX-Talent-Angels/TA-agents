"""Offline CLI/API tests through an injected suite registry."""

from __future__ import annotations

import json
from contextlib import contextmanager

import pytest
from fastapi.testclient import TestClient

from talent_angels import cli
from talent_angels.api.app import create_app
from talent_angels.cli import main
from talent_angels.suites import SuiteRegistry, SuiteRuntime
from tests.fakes.taxonomy import FakeCandidate, FakeNode, FakeToolResult


class FakeSuite:
    def search_nodes(self, text: str, kind: str | None = None) -> FakeToolResult:
        node = FakeNode(
            id="test:occupation:1",
            kind="Occupation",
            label=text,
            source="test",
            source_id="test-source-1",
            properties={},
        )
        return FakeToolResult(
            nodes=[node],
            candidates=[FakeCandidate(node=node, confidence=0.9, method="exact")],
            evidence=["test:search:exact"],
        )


def _registry(*, reachable: bool = True) -> SuiteRegistry:
    @contextmanager
    def factory():
        yield SuiteRuntime(name="test", suite=FakeSuite(), health_check=lambda: reachable)

    return SuiteRegistry({"test": factory}, default="test")


@pytest.fixture(autouse=True)
def _local_runtime(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setenv("RUNLOG_PATH", str(tmp_path / "runlog.jsonl"))
    monkeypatch.setenv("LLM_PROVIDER", "none")


def test_cli_uses_injected_registry(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["locate", "accountant"], registry=_registry())

    assert exit_code == 0
    output = json.loads(capsys.readouterr().out)
    assert output["suite"] == "test"
    assert output["confidence"] == 0.9


def test_api_uses_injected_registry_and_adapter_health() -> None:
    with TestClient(create_app(registry=_registry(reachable=False))) as client:
        health = client.get("/v1/health")
        response = client.post("/v1/capabilities/locate", json={"question": "accountant"})

    assert health.json()["neo4j_reachable"] is False
    assert response.status_code == 200
    assert response.json()["suite"] == "test"
    assert response.json()["result"]["nodes"][0]["pref_label"] == "accountant"


def test_cli_bench_serializes_separate_locate_metrics(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    golden_path = tmp_path / "golden.json"
    golden_path.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "question": "accountant",
                        "kind": "occupation",
                        "expected_top_id": "test:occupation:1",
                    },
                    {
                        "question": "ambiguous accountant",
                        "kind": "occupation",
                        "metric": "candidate_recall",
                        "expected_top_id": "test:occupation:1",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(cli, "GOLDEN_LOCATE_PATH", golden_path)

    assert main(["bench"], registry=_registry()) == 0

    report = json.loads(capsys.readouterr().out)
    baseline = report["baseline"]
    assert baseline["hit_at_1_accuracy"] == 1.0
    assert baseline["hit_at_1_questions"] == 1
    assert baseline["candidate_recall"] == 1.0
    assert baseline["candidate_recall_questions"] == 1
