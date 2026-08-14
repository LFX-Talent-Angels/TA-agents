"""Typed intent-to-plan tests."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from talent_angels.assistant.planning import (
    ExecutionPlan,
    Intent,
    PlanStep,
    build_plan,
)


@pytest.mark.parametrize(
    ("question", "target", "capabilities"),
    [
        ("software developer", "locate", ("locate",)),
        (
            "What essential skills does a software developer need?",
            "connect",
            ("locate", "connect"),
        ),
        (
            "What is the skill gap from data analyst to data scientist?",
            "pathfind",
            ("locate", "connect", "pathfind"),
        ),
    ],
)
def test_build_plan_is_cumulative_and_suite_scoped(
    question: str, target: str, capabilities: tuple[str, ...]
) -> None:
    plan = build_plan(question, suites=("esco",))

    assert plan.intent.target == target
    assert plan.capabilities == capabilities
    assert all(step.suites == ("esco",) for step in plan.steps)


def test_build_plan_preserves_explicit_relevant_suite_order() -> None:
    plan = build_plan("software developer", suites=("sfia", "esco"))

    assert plan.steps[0].suites == ("sfia", "esco")


@pytest.mark.parametrize("suites", [(), ("",), ("esco", "esco")])
def test_build_plan_rejects_invalid_suite_selection(suites: tuple[str, ...]) -> None:
    with pytest.raises(ValueError, match="suite"):
        build_plan("software developer", suites=suites)


def test_plan_models_are_immutable() -> None:
    plan = build_plan("software developer", suites=("esco",))

    with pytest.raises(ValidationError):
        plan.intent.target = "connect"


def test_builder_rejects_one_string_as_the_suite_sequence() -> None:
    with pytest.raises(ValueError, match="not a string"):
        build_plan("software developer", suites="esco")


def test_plan_step_rejects_empty_suites_when_constructed_directly() -> None:
    with pytest.raises(ValidationError, match="at least one suite"):
        PlanStep(capability="locate", suites=())


def test_execution_plan_rejects_non_cumulative_direct_construction() -> None:
    with pytest.raises(ValidationError, match="requires cumulative steps"):
        ExecutionPlan(
            intent=Intent(target="connect"),
            steps=(PlanStep(capability="pathfind", suites=("esco",)),),
        )


def test_execution_plan_rejects_inconsistent_suite_attachment() -> None:
    with pytest.raises(ValidationError, match="same suite selection"):
        ExecutionPlan(
            intent=Intent(target="connect"),
            steps=(
                PlanStep(capability="locate", suites=("esco",)),
                PlanStep(capability="connect", suites=("onet",)),
            ),
        )
