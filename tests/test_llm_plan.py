"""LLM planner JSON parsing and fallback."""

from __future__ import annotations

import pytest

from talent_angels.assistant.llm_plan import (
    PLAN_SYSTEM,
    PlanDraft,
    _profile_intent_heuristic,
    interpret_question,
    is_compare,
    parse_plan_text,
    uses_llm_planner,
)
from talent_angels.llm.protocol import LLMResult, LLMUsage, Message
from talent_angels.llm.stub_client import StubLLMClient


class ScriptedLLMClient:
    provider = "litellm"
    model = "openrouter/test"

    def __init__(self, text: str) -> None:
        self.text = text
        self.calls: list[list[Message]] = []

    def complete(self, messages: list[Message]) -> LLMResult:
        self.calls.append(messages)
        return LLMResult(
            text=self.text,
            provider=self.provider,
            model="actual-test-model",
            usage=LLMUsage(input_tokens=11, output_tokens=7),
        )


def test_parse_plan_text_accepts_fenced_json() -> None:
    draft = parse_plan_text(
        """```json
        {"target": "connect", "subject": "software developer",
         "kind": "occupation", "relation_filter": "essential", "suites": ["esco"]}
        ```"""
    )

    assert draft.target == "connect"
    assert draft.subject == "software developer"
    assert draft.kind == "occupation"
    assert draft.relation_filter == "essential"


def test_stub_provider_does_not_call_the_planner() -> None:
    interpreted = interpret_question(
        "What essential skills does a software developer need?",
        suite_name="esco",
        llm_client=StubLLMClient(),
    )

    assert interpreted.heuristic is True
    assert interpreted.stage is None
    assert interpreted.plan.capabilities == ("locate", "connect")
    assert uses_llm_planner(StubLLMClient()) is False


def test_llm_plan_builds_connect_execution_plan() -> None:
    client = ScriptedLLMClient(
        '{"target":"connect","subject":"software developer",'
        '"kind":"occupation","relation_filter":"essential","suites":["esco"]}'
    )

    interpreted = interpret_question(
        "What should a software developer know?",
        suite_name="esco",
        llm_client=client,
    )

    assert interpreted.heuristic is False
    assert interpreted.plan.capabilities == ("locate", "connect")
    assert interpreted.draft is not None
    assert interpreted.draft.subject == "software developer"
    assert interpreted.stage is not None
    assert interpreted.stage.stage == "intent"
    assert interpreted.stage.input_tokens == 11
    assert interpreted.stage.response_model == "actual-test-model"
    assert len(client.calls) == 1


def test_planner_provider_error_falls_back_to_heuristic() -> None:
    class ExplodingClient:
        provider = "litellm"

        def complete(self, messages: list[Message]) -> LLMResult:
            raise RuntimeError("LiteLLM provider request failed")

    interpreted = interpret_question(
        "What essential skills does a software developer need?",
        suite_name="esco",
        llm_client=ExplodingClient(),  # type: ignore[arg-type]
    )

    assert interpreted.heuristic is True
    assert interpreted.draft is None
    assert interpreted.plan.capabilities == ("locate", "connect")


def test_invalid_planner_json_falls_back_to_heuristic() -> None:
    client = ScriptedLLMClient("sorry, I cannot make a plan")

    interpreted = interpret_question(
        "What essential skills does a software developer need?",
        suite_name="esco",
        llm_client=client,
    )

    assert interpreted.heuristic is True
    assert interpreted.draft is None
    assert interpreted.plan.capabilities == ("locate", "connect")
    assert interpreted.stage is not None
    assert interpreted.stage.output_tokens == 7


def test_planner_prompt_asks_for_english_subjects() -> None:
    assert '("enfermero" → "nurse")' in PLAN_SYSTEM


@pytest.mark.parametrize(
    ("question", "intent", "subject"),
    [
        ("I am a plumber", "standing", "plumber"),
        ("I work as a nurse", "standing", "nurse"),
        ("my job is data analyst", "standing", "data analyst"),
        ("I am not a teacher", "reject", "teacher"),
        ("I am working toward data analyst", "goal", "data analyst"),
        ("I want to move into marketing", "goal", "marketing"),
        ("I'd like to switch to nursing", "goal", "nursing"),
        ("I want to work as a chef", "goal", "chef"),
        ("I'm aiming for data scientist", "goal", "data scientist"),
    ],
)
def test_profile_heuristic_reads_statements(question: str, intent: str, subject: str) -> None:
    draft = _profile_intent_heuristic(question)
    assert draft is not None
    assert (draft.profile_intent, draft.subject) == (intent, subject)


@pytest.mark.parametrize("question", ["I am looking for a job", "I'm interested in tech", "chef"])
def test_profile_heuristic_ignores_non_statements(question: str) -> None:
    assert _profile_intent_heuristic(question) is None


@pytest.mark.parametrize(
    "question", ["I want to become a web developer", "I want to be a web developer"]
)
def test_wanting_to_become_is_a_goal_not_a_skills_list(question: str) -> None:
    draft = _profile_intent_heuristic(question)
    assert draft is not None
    assert (draft.target, draft.profile_intent, draft.subject) == (
        "locate",
        "goal",
        "web developer",
    )


@pytest.mark.parametrize(
    ("question", "first", "second"),
    [
        ("data analyst vs data scientist", "data analyst", "data scientist"),
        ("a nurse versus a midwife?", "nurse", "midwife"),
        ("compare accountant and software developer", "accountant", "software developer"),
        (
            "compare accountant and software developer and help me choose",
            "accountant",
            "software developer",
        ),
        ("compare a chef with a baker", "chef", "baker"),
        (
            "what is the difference between a web developer and a software developer?",
            "web developer",
            "software developer",
        ),
    ],
)
def test_compare_heuristic_reads_two_titles(question: str, first: str, second: str) -> None:
    draft = _profile_intent_heuristic(question)
    assert draft is not None
    assert (draft.target, draft.subject, draft.secondary_subject) == ("connect", first, second)
    assert is_compare(draft)


@pytest.mark.parametrize(
    "question", ["what skills does a chef need?", "skills of a salt and pepper cook"]
)
def test_one_title_is_not_a_compare(question: str) -> None:
    assert not is_compare(_profile_intent_heuristic(question))


def test_pathfind_with_two_ends_is_not_a_compare() -> None:
    draft = PlanDraft(target="pathfind", subject="teacher", secondary_subject="data analyst")
    assert not is_compare(draft)


def test_planner_prompt_teaches_the_compare_shape() -> None:
    assert '"secondary_subject":"software developer"' in PLAN_SYSTEM
    assert '"X vs Y" or "X and Y" as two titles is locate' not in PLAN_SYSTEM
