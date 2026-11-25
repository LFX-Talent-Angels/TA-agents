"""Every new endpoint and slash command once, against the real graph; skipped without Neo4j."""

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
from talent_angels.memory.paths import outputs_dir  # noqa: E402
from talent_angels.session.kernel import handle_line  # noqa: E402
from talent_angels.session.store import new_session  # noqa: E402
from talent_angels.suites import SuiteRegistry, default_suite_registry  # noqa: E402
from tests.integration.support import neo4j_reachable  # noqa: E402

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not neo4j_reachable(), reason="Neo4j is not reachable"),
]

ROOT = Path(__file__).parents[2]
SAMPLE = ROOT / "tests" / "fixtures" / "discover_uca_sample.csv"
TAG_FIXTURE = ROOT / "tests" / "fixtures" / "tagging" / "standard_small.json"
SKILL = "esco:skill:de03f4fd-c147-4477-a048-7109e5ba2d6f"  # marketing principles (golden)


@pytest.fixture(scope="module")
def live() -> SuiteRegistry:
    return default_suite_registry(neighbor_cache=False)


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch, live: SuiteRegistry) -> Any:
    monkeypatch.setenv("LLM_PROVIDER", "none")
    with TestClient(create_app(registry=live)) as c:
        yield c


def _say(line: str, live: SuiteRegistry) -> str:
    def no_turn(*_a: object, **_k: object) -> None:
        raise AssertionError("a slash command is not a chat turn")

    return handle_line(new_session(), line, runner=no_turn, registry=live).text  # type: ignore[arg-type]


def test_api_pathfind_roadmap_and_successors(client: TestClient) -> None:
    pf = client.post(
        "/v1/capabilities/pathfind",
        json={"from": "software developer", "to": "data scientist", "suites": ["esco"]},
    )
    assert pf.status_code == 200
    assert pf.json()["suites"]["esco"]["status"] in {"ok", "ambiguous", "not_found"}

    rm = client.post(
        "/v1/capabilities/roadmap",
        json={"from": "software developer", "to": "data scientist", "suites": ["esco"]},
    )
    assert rm.status_code == 200 and rm.json()["policy"] == "learn-order-v1"

    sc = client.post("/v1/capabilities/successors", json={"skill": SKILL, "suites": ["esco"]})
    assert sc.status_code == 200 and sc.json()["suites"]["esco"]["status"] == "ok"
    assert (
        sc.content
        == client.post(
            "/v1/capabilities/successors", json={"skill": SKILL, "suites": ["esco"]}
        ).content
    )


def test_api_map_and_discover(client: TestClient) -> None:
    mp = client.post("/v1/capabilities/map", json={"start": "Python", "suites": ["esco"]})
    assert mp.status_code == 200 and mp.json()["suites"]["esco"]["status"] in {
        "mapped",
        "ambiguous",
    }
    rows = [{"item": "software developer", "related_to": ["Python"]}, {"item": "data analyst"}]
    dc = client.post("/v1/capabilities/discover", json={"items": rows, "suites": ["esco"]})
    assert dc.status_code == 200 and dc.json()["summary"]["items"] == 2


def test_chat_commands(live: SuiteRegistry, tmp_path: Path) -> None:
    assert "learn-order-v1" in _say("/roadmap software developer | data scientist", live)
    assert "What comes after" in _say(f"/successors {SKILL} --suites esco", live)
    assert "Map for" in _say("/map Python --suites esco", live)
    assert "items:" in _say(f"/discover {SAMPLE} --suites esco --depth 1", live)
    assert {p.suffix for p in outputs_dir().glob("*/*")} >= {".json", ".xlsx", ".html"}

    project = tmp_path / "project"
    (project / "inputs").mkdir(parents=True)
    shutil.copy(TAG_FIXTURE, project / "inputs" / "standard.json")
    assert "units=" in _say(f"/tag {project} --no-llm --limit 2 --suites esco", live)
    assert list(project.rglob("*.xlsx"))


def test_chat_command_output_is_byte_identical(live: SuiteRegistry) -> None:
    _say(f"/successors {SKILL} --suites esco", live)
    [first] = list(outputs_dir().glob("successors-*/successors.json"))
    before = first.read_bytes()
    _say(f"/successors {SKILL} --suites esco", live)
    assert first.read_bytes() == before and json.loads(before)["x"] == SKILL
