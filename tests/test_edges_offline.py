"""Offline CLI/API tests through an injected suite registry."""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from talent_angels import cli
from talent_angels.api.app import create_app
from talent_angels.cli import main
from talent_angels.llm.protocol import LLMResult, LLMUsage, Message
from talent_angels.suites import SuiteRegistry, SuiteRuntime
from tests.fakes.taxonomy import FakeCandidate, FakeEdge, FakeNode, FakeToolResult


class RecordingLLMClient:
    provider = "test"
    model = "recorder"

    def __init__(self) -> None:
        self.calls: list[list[Message]] = []

    def complete(self, messages: list[Message]) -> LLMResult:
        self.calls.append(messages)
        return LLMResult(
            text="phrased",
            provider=self.provider,
            model=self.model,
            usage=LLMUsage(input_tokens=3, output_tokens=2),
        )


class FakeSuite:
    def __init__(self) -> None:
        self.last_node: FakeNode | None = None

    def search_nodes(self, text: str, kind: str | None = None) -> FakeToolResult:
        node = FakeNode(
            id="test:occupation:1",
            kind="Occupation",
            label=text,
            source="test",
            source_id="test-source-1",
            properties={},
        )
        self.last_node = node
        return FakeToolResult(
            nodes=[node],
            candidates=[FakeCandidate(node=node, confidence=0.9, method="exact")],
            evidence=["test:search:exact"],
        )

    def get_neighbors(self, node_id: str, rel_types: list[str] | None = None) -> FakeToolResult:
        assert self.last_node is not None
        skill = FakeNode(
            id="test:skill:1",
            kind="Skill",
            label="analyse software requirements",
            source="test",
            source_id="test-skill-1",
            properties={},
        )
        return FakeToolResult(
            nodes=[self.last_node, skill],
            edges=[
                FakeEdge(
                    type="HAS_SKILL",
                    from_id=node_id,
                    to_id=skill.id,
                    properties={"relation_type": "essential"},
                )
            ],
            evidence=[f"test:neighbors:{node_id}"],
        )


def _registry(*, reachable: bool = True) -> SuiteRegistry:
    @contextmanager
    def factory():
        yield SuiteRuntime(name="test", suite=FakeSuite(), health_check=lambda: reachable)

    return SuiteRegistry({"test": factory}, default="test")


@pytest.fixture(autouse=True)
def _local_runtime(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setenv("RUNLOG_PATH", str(tmp_path / "runlog.jsonl"))
    monkeypatch.setenv("QUERY_DETAILS_DIR", str(tmp_path / "query-details"))
    monkeypatch.setenv("LLM_PROVIDER", "none")
    monkeypatch.setenv("LLM_MODEL", "stub")
    monkeypatch.setenv("ANSWER_MODE", "structured")


def test_cli_uses_injected_registry(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["locate", "accountant"], registry=_registry())

    assert exit_code == 0
    output = json.loads(capsys.readouterr().out)
    assert next(iter(output)) == "answer"
    assert "nodes" not in output
    assert output["node_count"] == 1
    assert output["suite"] == "test"
    assert output["confidence"] == 0.9
    assert output["plan"] == ["locate"]
    assert output["tokens"]["calls"] == 0
    assert output["cost_usd"]["known"] is True
    assert output["cost_usd"]["total"] == 0.0
    details = Path(output["details"])
    assert details.is_file()
    body = details.read_text(encoding="utf-8")
    assert "accountant" in body
    assert "## Answer" in body


def test_api_uses_injected_registry_and_adapter_health() -> None:
    with TestClient(create_app(registry=_registry(reachable=False))) as client:
        health = client.get("/v1/health")
        response = client.post("/v1/capabilities/locate", json={"question": "accountant"})

    assert health.json()["neo4j_reachable"] is False
    assert response.status_code == 200
    assert response.json()["suite"] == "test"
    assert response.json()["result"]["nodes"][0]["pref_label"] == "accountant"


def test_cli_pathfind_question_is_honest(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = main(
        ["query", "What is the skill path from data analyst to data scientist?"],
        registry=_registry(),
    )

    assert exit_code == 0
    output = json.loads(capsys.readouterr().out)
    assert output["capability"] == "pathfind"
    assert output["plan"] == ["locate", "connect", "pathfind"]
    assert "capability_not_implemented:pathfind" in output["warnings"]
    assert "Pathfind is not in this MVP" in output["answer"]
    assert output["tools"] == []


def test_api_pathfind_question_is_honest() -> None:
    with TestClient(create_app(registry=_registry())) as client:
        response = client.post(
            "/v1/query",
            json={"question": "skill path from data analyst to data scientist"},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["capability"] == "pathfind"
    assert "capability_not_implemented:pathfind" in body["result"]["warnings"]
    assert "Pathfind is not in this MVP" in body["answer"]
    assert body["usage"]["tools"] == []


def test_cli_connect_uses_the_same_assistant_flow(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = main(
        ["connect", "What essential skills does a software developer need?"],
        registry=_registry(),
    )

    assert exit_code == 0
    output = json.loads(capsys.readouterr().out)
    assert output["capability"] == "connect"
    assert "analyse software requirements" in output["answer"]


def test_api_connect_reports_both_taxonomy_tool_calls() -> None:
    with TestClient(create_app(registry=_registry())) as client:
        response = client.post(
            "/v1/capabilities/connect",
            json={"question": "What essential skills does a software developer need?"},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["capability"] == "connect"
    assert [tool["name"] for tool in body["usage"]["tools"]] == [
        "search_nodes",
        "get_neighbors",
    ]
    assert body["usage"]["graph"]["queries"] == 2
    assert body["usage"]["graph"]["total_ms"] == pytest.approx(
        sum(tool["ms"] for tool in body["usage"]["tools"])
    )


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


def test_cli_locate_uses_natural_answer_mode(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    recorder = RecordingLLMClient()
    monkeypatch.setenv("ANSWER_MODE", "natural")
    monkeypatch.setattr(cli, "get_llm_client", lambda: recorder)

    assert main(["locate", "accountant"], registry=_registry()) == 0

    output = json.loads(capsys.readouterr().out)
    assert output["answer"] == "phrased"
    assert len(recorder.calls) == 1


def test_cli_bench_stays_structured_when_natural_mode_is_set(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    recorder = RecordingLLMClient()
    monkeypatch.setenv("ANSWER_MODE", "natural")
    monkeypatch.setattr(cli, "get_llm_client", lambda: recorder)
    golden_path = tmp_path / "golden.json"
    golden_path.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "question": "accountant",
                        "kind": "occupation",
                        "expected_top_id": "test:occupation:1",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(cli, "GOLDEN_LOCATE_PATH", golden_path)

    assert main(["bench"], registry=_registry()) == 0
    json.loads(capsys.readouterr().out)
    assert recorder.calls == []
