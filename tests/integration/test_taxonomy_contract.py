"""Compatibility check against the real, merged TA-taxonomies contract."""

from __future__ import annotations

import pytest

pytest.importorskip(
    "ta_taxonomies",
    reason="TA-taxonomies is not installed; CI installs the merged contract for this check",
)

from ta_taxonomies.contract import Candidate, Node, ToolResult  # noqa: E402

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


def test_locate_accepts_real_taxonomy_contract_models() -> None:
    outcome = locate(ContractSuite(), "esco", "software developer", kind="occupation")

    assert outcome.nodes[0].id == "esco:occupation:contract-check"
    assert outcome.confidence == 0.95
    assert outcome.evidence[0].pointer == "esco:search:exact_pref:contract-check"
