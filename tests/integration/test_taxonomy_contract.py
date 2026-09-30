"""Compatibility check against the real, merged TA-taxonomies contract."""

from __future__ import annotations

import pytest

pytest.importorskip(
    "ta_taxonomies",
    reason="TA-taxonomies is not installed; CI installs the merged contract for this check",
)

from ta_taxonomies.contract import Candidate, Edge, Node, ToolResult  # noqa: E402

from talent_angels.assistant.connect_request import extract_connect_request  # noqa: E402
from talent_angels.contracts import EvidencePointer, NodeRef  # noqa: E402
from talent_angels.skills.connect import connect  # noqa: E402
from talent_angels.skills.locate import locate  # noqa: E402

pytestmark = pytest.mark.integration


class ContractSuite:
    """Small adapter whose result uses the actual cross-repo contract models."""

    def search_nodes(self, text: str, kind: str | None = None) -> ToolResult:
        node = Node(
            id="esco:occupation:contract-check",
            kind="Occupation",
            label=text,
            source="esco",
            source_id="https://example.invalid/esco/contract-check",
        )
        return ToolResult(
            nodes=[node],
            candidates=[Candidate(node=node, confidence=0.95, method="exact_pref")],
            evidence=["esco:search:exact_pref:contract-check"],
        )

    def get_neighbors(self, node_id: str, rel_types: list[str] | None = None) -> ToolResult:
        skill = Node(
            id="esco:skill:contract-check",
            kind="Skill",
            label="analyse software requirements",
            source="esco",
            source_id="https://example.invalid/esco/skill/contract-check",
        )
        return ToolResult(
            nodes=[skill],
            edges=[
                Edge(
                    type="HAS_SKILL",
                    from_id=node_id,
                    to_id=skill.id,
                    properties={"relation_type": "essential"},
                )
            ],
            evidence=[f"esco:neighbors:{node_id}"],
        )


def test_locate_accepts_real_taxonomy_contract_models() -> None:
    outcome = locate(ContractSuite(), "esco", "software developer", kind="occupation")

    assert outcome.nodes[0].id == "esco:occupation:contract-check"
    assert outcome.confidence == 0.95
    assert outcome.evidence[0].pointer == "esco:search:exact_pref:contract-check"


def test_connect_accepts_real_taxonomy_contract_models() -> None:
    center = NodeRef(
        id="esco:occupation:contract-check",
        suite="esco",
        source="esco",
        source_id="https://example.invalid/esco/contract-check",
        kind="Occupation",
        pref_label="software developer",
    )

    outcome = connect(
        ContractSuite(),
        "esco",
        center,
        request=extract_connect_request("What essential skills does a software developer need?"),
        confidence=0.95,
        locate_evidence=[EvidencePointer(suite="esco", pointer="esco:search:exact_pref")],
    )

    assert [node.pref_label for node in outcome.nodes] == [
        "software developer",
        "analyse software requirements",
    ]
    assert outcome.edges[0].properties == {"relation_type": "essential"}
    assert [evidence.pointer for evidence in outcome.evidence] == [
        "esco:search:exact_pref",
        "esco:neighbors:esco:occupation:contract-check",
    ]


def test_the_runtime_imports_every_symbol_it_needs_from_taxonomies() -> None:
    """The concrete adapters and the schema must exist in the pinned package.

    The contract check above only covered the result models, so TA-agents could
    depend on `OnetSuite` and `SuiteSchema` from an unmerged TA-taxonomies
    branch and CI would still pass.
    """
    from ta_taxonomies.contract.schema import SuiteSchema
    from ta_taxonomies.suites.esco.tools import EscoSuite
    from ta_taxonomies.suites.onet.tools import OnetSuite

    import talent_angels.suites.esco  # noqa: F401
    import talent_angels.suites.onet  # noqa: F401
    from talent_angels.suites.schema import SuiteSchema as Reexported

    assert Reexported is SuiteSchema
    for suite in (EscoSuite, OnetSuite):
        assert hasattr(suite, "suite_schema"), suite
