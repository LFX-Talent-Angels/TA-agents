"""The engine front door against the real graph: outputs match their models, and the new
routes give the old routes' bytes. Skipped without Neo4j."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip(
    "ta_taxonomies",
    reason="TA-taxonomies is not installed; install the sibling package for integration tests",
)

from fastapi.testclient import TestClient  # noqa: E402

from talent_angels.api.app import create_app  # noqa: E402
from talent_angels.engine import CAPABILITIES, jobs, run  # noqa: E402
from talent_angels.suites import SuiteRegistry, default_suite_registry  # noqa: E402
from tests.integration.support import neo4j_reachable  # noqa: E402

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not neo4j_reachable(), reason="Neo4j is not reachable"),
]

TAG_FIXTURE = Path(__file__).parents[1] / "fixtures" / "tagging" / "standard_small.json"
SKILL = "esco:skill:de03f4fd-c147-4477-a048-7109e5ba2d6f"  # marketing principles (golden)
CASES: list[tuple[str, dict[str, Any]]] = [
    ("map", {"start": "Python", "depth": 2, "suites": ["esco", "onet"]}),
    ("map", {"route_from": "web developer", "route_to": "software developer"}),
    ("pathfind", {"from": "software developer", "to": "data scientist"}),
    ("roadmap", {"from": "software developer", "to": "data scientist"}),
    ("roadmap", {"from": "15-2051.01", "to": "15-2051.00", "suites": ["onet"]}),
    ("successors", {"skill": SKILL}),
    ("discover", {"items": [{"item": "web developer", "related_to": ["JavaScript"]},
                            {"item": "data analyst"}], "depth": 2}),
    ("locate", {"text": "software developer", "kind": "occupation", "suite": "esco"}),
]  # fmt: skip


@pytest.fixture(scope="module")
def live() -> SuiteRegistry:
    return default_suite_registry(neighbor_cache=False)


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch, live: SuiteRegistry) -> Any:
    monkeypatch.setenv("LLM_PROVIDER", "none")
    with TestClient(create_app(registry=live)) as c:
        yield c


@pytest.mark.parametrize(("name", "payload"), CASES, ids=[c[0] for c in CASES])
def test_a_live_output_validates_against_its_model_and_reruns_to_the_same_bytes(
    live: SuiteRegistry, name: str, payload: dict[str, Any]
) -> None:
    result = run(name, payload, registry=live)
    assert result.output is not None and result.status in {"ok", "ambiguous", "no_result"}
    CAPABILITIES[name].output_model.model_validate(result.output)  # extra="forbid"
    assert result.call is not None
    again = run(result.call.capability, result.call.input, registry=live)
    assert json.dumps(again.output) == json.dumps(result.output)


@pytest.mark.parametrize(
    ("name", "payload"),
    [c for c in CASES if c[0] != "locate"],
    ids=lambda x: x if isinstance(x, str) else "",
)
def test_run_is_the_dedicated_route_byte_for_byte(
    client: TestClient, name: str, payload: dict[str, Any]
) -> None:
    dedicated = client.post(f"/v1/capabilities/{name}", json=payload)
    via_run = client.post("/v1/run", json={"capability": name, "input": payload})
    assert dedicated.status_code == via_run.status_code == 200
    assert json.dumps(via_run.json()["output"]) == json.dumps(dedicated.json())


def test_connect_follows_locate(live: SuiteRegistry) -> None:
    found = run("locate", CASES[-1][1], registry=live).output
    assert found is not None
    node_id = found["candidates"][0]["node"]["id"]
    near = run("connect", {"node_id": node_id, "suite": "esco", "limit": 20}, registry=live)
    assert near.status == "ok" and near.output
    CAPABILITIES["connect"].output_model.model_validate(near.output)


def test_a_tag_job_without_a_model_over_http(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / "root"
    (root / "proj" / "inputs").mkdir(parents=True)
    shutil.copy(TAG_FIXTURE, root / "proj" / "inputs" / "standard.json")
    monkeypatch.setenv("TA_AGENTS_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("TA_TAG_ROOT", str(root))
    monkeypatch.delenv("TA_JOBS_ALLOW_MODEL", raising=False)
    monkeypatch.setenv("TA_JOBS_ENABLED", "1")
    started = client.post(
        "/v1/jobs",
        json={"capability": "tag", "input": {"path": "proj", "limit": 2, "suites": ["esco"]}},
    )
    assert started.status_code == 202
    jobs._join(started.json()["id"], timeout=120)
    done = client.get(f"/v1/jobs/{started.json()['id']}").json()
    assert done["status"] == "done" and done["result"]["output"]["units"] > 0
    assert list((root / "proj").rglob("*.xlsx"))
