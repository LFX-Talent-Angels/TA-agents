"""Live ESCO checks for the generic Locate skill and assistant loop."""

from __future__ import annotations

import pytest

pytest.importorskip(
    "ta_taxonomies",
    reason="TA-taxonomies is not installed; install the sibling package for integration tests",
)

from talent_angels.assistant import build_graph  # noqa: E402
from talent_angels.llm.stub_client import StubLLMClient  # noqa: E402
from talent_angels.skills.locate import ESCO_SUITE_NAME, locate  # noqa: E402
from talent_angels.suites import default_suite_registry  # noqa: E402
from tests.integration.support import neo4j_reachable  # noqa: E402

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not neo4j_reachable(), reason="Neo4j is not reachable"),
]


def test_locate_against_live_esco_fixture_finds_software_developer() -> None:
    with default_suite_registry().open() as runtime:
        outcome = locate(runtime.suite, runtime.name, "software developer", kind="occupation")

    assert outcome.confidence == 0.95
    assert outcome.nodes
    assert outcome.nodes[0].kind == "Occupation"
    assert outcome.nodes[0].pref_label.lower() == "software developer"


def test_graph_end_to_end_against_live_esco_fixture() -> None:
    with default_suite_registry().open() as runtime:
        graph = build_graph(
            suite=runtime.suite,
            suite_name=runtime.name,
            llm_client=StubLLMClient(),
            answer_mode="structured",
        )
        final_state = graph.invoke({"question": "software developer", "kind": "occupation"})

    assert final_state["result"].confidence == 0.95
    assert "software developer" in final_state["answer"].lower()
    assert final_state["result"].suite == ESCO_SUITE_NAME
