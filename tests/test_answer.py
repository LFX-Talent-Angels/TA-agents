"""User-facing answer packaging tests."""

from talent_angels.assistant.answer import CONNECT_PREVIEW_CAP, build_answer, summarize_result
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


def _connect_result(skill_count: int) -> AgentResult:
    subject = _node(1).model_copy(update={"pref_label": "software developer"})
    skills = [
        _node(100 + i).model_copy(update={"kind": "Skill", "pref_label": f"skill {i}"})
        for i in range(skill_count)
    ]
    return AgentResult(
        capability="connect",
        suite="esco",
        nodes=[subject, *skills],
        edges=[
            EdgeRef(
                type="HAS_SKILL",
                suite="esco",
                source_node_id=subject.id,
                target_node_id=skill.id,
                properties={"relation_type": "essential"},
            )
            for skill in skills
        ],
        confidence=0.95,
    )


def test_summarize_result_shows_every_skill_up_to_the_preview_cap() -> None:
    """GAP F: exactly at the cap, nothing is truncated — no payload pointer."""
    result = _connect_result(CONNECT_PREVIEW_CAP)
    summary = summarize_result(result)
    assert f"skill {CONNECT_PREVIEW_CAP - 1}" in summary
    assert "result payload" not in summary


def test_summarize_result_truncates_one_past_the_preview_cap() -> None:
    result = _connect_result(CONNECT_PREVIEW_CAP + 1)
    summary = summarize_result(result)
    assert "result payload" in summary
    assert f"skill {CONNECT_PREVIEW_CAP}" not in summary


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
    """Truncation now kicks in past CONNECT_PREVIEW_CAP (GAP F raised it from 5)."""
    subject = _node(1).model_copy(update={"pref_label": "software developer"})
    skills = [
        _node(i).model_copy(update={"kind": "Skill", "pref_label": f"skill {i}"})
        for i in range(2, 22)
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

    assert "20 direct connection(s)" in answer
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


def test_the_answer_prompt_carries_no_recall_block() -> None:
    """The deliberate omission, pinned so it stays a decision and not a drift.

    `build_answer` has no `question` parameter, so it cannot append
    `recall_prefix()` the way the other three prompt sites do. That is the
    design — this is the one prompt that restates the user's own words back to
    the user, and a list of past questions in front of "rephrase the following
    taxonomy result" invites the model to answer them instead. The test reads
    the real prompt a client is handed, so it fails if someone adds recall at a
    call site, not only if someone changes the signature.
    """
    from talent_angels.llm.protocol import LLMResult, LLMUsage, Message

    seen: list[list[Message]] = []

    class _Client:
        provider = "litellm"
        model = "test-model"

        def complete(self, messages: list[Message], **_kwargs: object) -> LLMResult:
            seen.append(messages)
            return LLMResult(
                text="rephrased",
                provider="litellm",
                model="test-model",
                usage=LLMUsage(input_tokens=1, output_tokens=1),
            )

    result = AgentResult(
        capability="locate",
        suite="esco",
        nodes=[_node(1)],
        confidence=0.9,
    )

    build_answer(result, llm_client=_Client(), mode="natural")

    assert len(seen) == 1
    prompt = "\n".join(message.content for message in seen[0])
    assert "Past turns" not in prompt, "recall leaked into the answer prompt"
    assert "past question" not in prompt


def test_the_answer_signature_has_no_question_parameter() -> None:
    """Why the test above is not a suggestion to add one.

    `build_answer(result, *, llm_client, mode)` is the shape the decision needs:
    a caller that hands it the question is a caller who has decided the past
    belongs in a rephrasing prompt. Asserted through `inspect` so the signature
    cannot grow a `question` without a test noticing.
    """
    import inspect

    parameters = inspect.signature(build_answer).parameters
    assert "question" not in parameters, (
        f"build_answer grew a {list(parameters)} — see the docstring before adding recall"
    )
    assert list(parameters) == ["result", "llm_client", "mode"]
