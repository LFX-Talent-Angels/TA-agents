"""Merged answers keep suite identity; they never blend node ids."""

from __future__ import annotations

from talent_angels.assistant.merge import merge_answers, suite_heading
from talent_angels.assistant.synthesize import synthesize
from talent_angels.contracts import AgentResult, NodeRef
from talent_angels.llm.protocol import LLMResult, LLMUsage, Message


def _node(*, suite: str, node_id: str, label: str) -> NodeRef:
    return NodeRef(
        id=node_id,
        suite=suite,
        source=suite,
        source_id=node_id,
        kind="Occupation",
        pref_label=label,
    )


def test_merge_labels_each_suite_and_keeps_both_ids() -> None:
    esco = AgentResult(
        capability="locate",
        suite="esco",
        nodes=[_node(suite="esco", node_id="esco:occupation:dev", label="software developer")],
        confidence=0.95,
    )
    onet = AgentResult(
        capability="locate",
        suite="onet",
        nodes=[
            _node(
                suite="onet",
                node_id="onet:occupation:15-1252.00",
                label="Software Developers",
            )
        ],
        confidence=0.95,
    )

    answer = merge_answers((esco, onet))

    assert "software developer" in answer
    assert "Software Developers" in answer
    assert "Sources used: ESCO · O*NET" in answer
    assert "separate official records" in answer
    assert "esco:occupation:dev" not in answer or "not one shared id" in answer


def test_unreachable_suites_surface_as_warnings() -> None:
    answer = merge_answers((), extra_warnings=("suite_unavailable:onet",))
    assert "No attached taxonomy was reachable" in answer
    assert "suite_unavailable:onet" in answer


def test_suite_heading_for_known_and_future_suites() -> None:
    assert suite_heading("esco") == "ESCO"
    assert suite_heading("onet") == "O*NET"
    assert suite_heading("sfia") == "SFIA"


class _CodeBlockClient:
    provider = "litellm"
    model = "test"

    def complete(self, messages: list[Message], **_kwargs: object) -> LLMResult:
        text = "Both maps list a chef.\n\n```\nSources used:\n- ESCO: chef\n```"
        return LLMResult(text=text, provider=self.provider, model=self.model, usage=LLMUsage())


def test_synthesize_drops_a_code_block_and_prints_sources_once() -> None:
    esco = AgentResult(
        capability="locate",
        suite="esco",
        nodes=[_node(suite="esco", node_id="esco:occupation:chef", label="chef")],
    )
    answer = synthesize((esco,), question="chef", llm_client=_CodeBlockClient())
    assert "```" not in answer
    assert answer.count("Sources used") == 1
