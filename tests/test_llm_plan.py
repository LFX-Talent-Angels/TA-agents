"""LLM planner JSON parsing and fallback."""

from __future__ import annotations

from talent_angels.assistant.llm_plan import (
    interpret_question,
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


def test_skills_question_upgrades_a_locate_plan_to_connect() -> None:
    client = ScriptedLLMClient(
        '{"target":"locate","subject":"software developer","kind":"occupation"}'
    )

    interpreted = interpret_question(
        "What essential skills does a software developer need?",
        suite_name="esco",
        llm_client=client,
    )

    assert interpreted.plan.capabilities == ("locate", "connect")
    assert interpreted.draft is not None
    assert interpreted.draft.target == "connect"
    assert interpreted.draft.subject == "software developer"
    assert interpreted.heuristic is False


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
