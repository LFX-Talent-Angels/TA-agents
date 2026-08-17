"""User-facing answer packaging tests."""

from talent_angels.assistant.answer import build_answer
from talent_angels.contracts import AgentResult, EdgeRef, NodeRef
from talent_angels.llm.stub_client import StubLLMClient


def _node(index: int) -> NodeRef:
    return NodeRef(
        id=f"esco:occupation:{index}",
        suite="esco",
        source="esco",
        source_id=f"source-{index}",
        kind="Occupation",
        pref_label=f"candidate {index}",
    )


def test_structured_answer_surfaces_ambiguous_choices() -> None:
    result = AgentResult(
        capability="locate",
        suite="esco",
        nodes=[_node(i) for i in range(1, 6)],
        confidence=0.7,
        warnings=["ambiguous"],
    )

    answer, usage = build_answer(result, llm_client=StubLLMClient(), mode="structured")

    assert answer.startswith("Ambiguous locate result")
    assert "candidate 1" in answer
    assert "candidate 3" in answer
    assert "candidate 4" not in answer
    assert "2 more candidate(s)" in answer
    assert "clarify" in answer.lower()
    assert usage is None


def test_structured_answer_refuses_unimplemented_pathfind() -> None:
    result = AgentResult(
        capability="pathfind",
        suite="esco",
        warnings=["capability_not_implemented:pathfind"],
    )

    answer, usage = build_answer(result, llm_client=StubLLMClient(), mode="structured")

    assert "Pathfind is not in this MVP" in answer
    assert "skills of one occupation" in answer
    assert usage is None


def test_structured_answer_keeps_unique_match_summary() -> None:
    result = AgentResult(
        capability="locate",
        suite="esco",
        nodes=[_node(1)],
        confidence=0.95,
    )

    answer, usage = build_answer(result, llm_client=StubLLMClient(), mode="structured")

    assert answer == "candidate 1 (Occupation, id=esco:occupation:1) — confidence 95%"
    assert usage is None


def test_structured_connect_points_at_full_payload_when_truncated() -> None:
    subject = _node(1).model_copy(update={"pref_label": "software developer"})
    skills = [
        _node(i).model_copy(update={"kind": "Skill", "pref_label": f"skill {i}"})
        for i in range(2, 10)
    ]
    result = AgentResult(
        capability="connect",
        suite="esco",
        nodes=[subject, *skills],
        edges=[
            EdgeRef(
                type="HAS_SKILL",
                suite="esco",
                source_node_id=subject.id,
                target_node_id=skill.id,
            )
            for skill in skills
        ],
        confidence=0.95,
    )

    answer, usage = build_answer(result, llm_client=StubLLMClient(), mode="natural")

    assert "8 direct connection(s)" in answer
    assert "full list is in the result payload" in answer
    assert usage is None


def test_structured_connect_answer_names_subject_and_neighbors() -> None:
    subject = _node(1).model_copy(update={"pref_label": "software developer"})
    programming = _node(2).model_copy(
        update={"kind": "Skill", "pref_label": "computer programming"}
    )
    testing = _node(3).model_copy(update={"kind": "Skill", "pref_label": "software testing"})
    result = AgentResult(
        capability="connect",
        suite="esco",
        nodes=[subject, programming, testing],
        edges=[
            EdgeRef(
                type="HAS_SKILL",
                suite="esco",
                source_node_id=subject.id,
                target_node_id=programming.id,
                properties={"relation_type": "essential"},
            ),
            EdgeRef(
                type="HAS_SKILL",
                suite="esco",
                source_node_id=subject.id,
                target_node_id=testing.id,
                properties={"relation_type": "essential"},
            ),
        ],
        confidence=0.95,
    )

    answer, usage = build_answer(result, llm_client=StubLLMClient(), mode="structured")

    assert "software developer" in answer
    assert "computer programming" in answer
    assert "software testing" in answer
    assert "2 direct connection(s)" in answer
    assert "95%" in answer
    assert usage is None


def test_natural_answer_falls_back_when_provider_fails() -> None:
    subject = _node(1).model_copy(update={"pref_label": "software developer"})
    skill = _node(2).model_copy(update={"kind": "Skill", "pref_label": "programming"})
    result = AgentResult(
        capability="connect",
        suite="esco",
        nodes=[subject, skill],
        edges=[
            EdgeRef(
                type="HAS_SKILL",
                suite="esco",
                source_node_id=subject.id,
                target_node_id=skill.id,
            )
        ],
        confidence=0.95,
    )

    class BrokenClient:
        provider = "litellm"
        model = "broken"

        def complete(self, messages: object, **_kwargs: object) -> object:
            raise RuntimeError("upstream")

    answer, stage = build_answer(result, llm_client=BrokenClient(), mode="natural")

    assert "programming" in answer
    assert stage is None


def test_natural_connect_answer_calls_the_llm_ok() -> None:
    subject = _node(1).model_copy(update={"pref_label": "software developer"})
    skill = _node(2).model_copy(update={"kind": "Skill", "pref_label": "programming"})
    result = AgentResult(
        capability="connect",
        suite="esco",
        nodes=[subject, skill],
        edges=[
            EdgeRef(
                type="HAS_SKILL",
                suite="esco",
                source_node_id=subject.id,
                target_node_id=skill.id,
            )
        ],
        confidence=0.95,
    )

    answer, stage = build_answer(result, llm_client=StubLLMClient(), mode="natural")

    assert "programming" in answer
    assert stage is not None
    assert stage.stage == "answer"
    assert stage.calls == 1
