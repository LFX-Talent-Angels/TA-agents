"""Typed deterministic planning for Locate, Connect, and Pathfind."""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from talent_angels.assistant.intent import (
    CAPABILITY_CONNECT,
    CAPABILITY_LOCATE,
    CAPABILITY_PATHFIND,
    Capability,
    classify_capability,
)

_CAPABILITY_SEQUENCE: dict[Capability, tuple[Capability, ...]] = {
    CAPABILITY_LOCATE: (CAPABILITY_LOCATE,),
    CAPABILITY_CONNECT: (CAPABILITY_LOCATE, CAPABILITY_CONNECT),
    CAPABILITY_PATHFIND: (
        CAPABILITY_LOCATE,
        CAPABILITY_CONNECT,
        CAPABILITY_PATHFIND,
    ),
}


class Intent(BaseModel):
    """The capability the user's question ultimately asks for."""

    model_config = ConfigDict(frozen=True)

    target: Capability


class PlanStep(BaseModel):
    """One capability applied independently to the named taxonomy suites."""

    model_config = ConfigDict(frozen=True)

    capability: Capability
    suites: tuple[str, ...]

    @field_validator("suites")
    @classmethod
    def validate_suites(cls, suites: tuple[str, ...]) -> tuple[str, ...]:
        if not suites:
            raise ValueError("plan step requires at least one suite")
        if any(not suite.strip() for suite in suites):
            raise ValueError("plan suite names must not be blank")
        if len(set(suites)) != len(suites):
            raise ValueError("plan suite names must be unique")
        return suites


class ExecutionPlan(BaseModel):
    """Ordered, cumulative steps owned by the main assistant."""

    model_config = ConfigDict(frozen=True)

    intent: Intent
    steps: tuple[PlanStep, ...]

    @model_validator(mode="after")
    def validate_cumulative_steps(self) -> ExecutionPlan:
        expected = _CAPABILITY_SEQUENCE[self.intent.target]
        capabilities = tuple(step.capability for step in self.steps)
        if capabilities != expected:
            raise ValueError(f"{self.intent.target} intent requires cumulative steps {expected!r}")
        suites = self.steps[0].suites
        if any(step.suites != suites for step in self.steps[1:]):
            raise ValueError("all plan steps must use the same suite selection")
        return self

    @property
    def capabilities(self) -> tuple[Capability, ...]:
        return tuple(step.capability for step in self.steps)


def _validated_suites(suites: Sequence[str]) -> tuple[str, ...]:
    if isinstance(suites, str):
        raise ValueError("suites must be a sequence of suite names, not a string")
    selected = tuple(suites)
    if not selected:
        raise ValueError("plan requires at least one suite")
    if any(not suite.strip() for suite in selected):
        raise ValueError("plan suite names must not be blank")
    if len(set(selected)) != len(selected):
        raise ValueError("plan suite names must be unique")
    return selected


def build_plan_for_capability(target: Capability, *, suites: Sequence[str]) -> ExecutionPlan:
    """Create the cumulative plan ending at an explicit capability."""
    selected = _validated_suites(suites)

    steps = tuple(
        PlanStep(capability=capability, suites=selected)
        for capability in _CAPABILITY_SEQUENCE[target]
    )
    return ExecutionPlan(intent=Intent(target=target), steps=steps)


def build_plan(question: str, *, suites: Sequence[str]) -> ExecutionPlan:
    """Classify a question and create an L, L+C, or L+C+P plan."""
    return build_plan_for_capability(classify_capability(question), suites=suites)
