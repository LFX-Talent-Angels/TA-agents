"""LLM-authored typed plans with heuristic fallback."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from talent_angels.assistant.intent import (
    CAPABILITY_CONNECT,
    CAPABILITY_LOCATE,
    CAPABILITY_PATHFIND,
    Capability,
    classify_capability,
)
from talent_angels.assistant.llm_call import measure_complete
from talent_angels.assistant.planning import ExecutionPlan, build_plan, build_plan_for_capability
from talent_angels.llm import LLMClient, Message
from talent_angels.runlog import StageUsage
from talent_angels.skills.connect.models import ConnectRequest
from talent_angels.skills.locate import ESCO_SUITE_NAME

PLAN_SYSTEM = """You are the LFX Talent Angels planner. Return ONLY a JSON object.

Keys:
- target: locate | connect | pathfind
- subject: short search phrase, or null
- secondary_subject: second pathfind endpoint, or null
- kind: occupation | skill | null
- rel_types: array of relationship names, or null
- relation_filter: essential | optional | null
- suites: array of suite names (default ["esco"])

How to choose target:
- locate = only identify / define a node ("what is X", "where is X in ESCO")
- connect = neighbors, skills, hierarchy around one node
  ("what skills does X need", "essential skills", "neighbors of X")
- pathfind = a route or gap between TWO things
  ("path from A to B", "skill gap from A to B")

If the user asks for skills, neighbors, or what someone needs, target MUST be
connect, not locate. Put only the occupation or skill name in subject — never
the whole sentence. "Be" and "become" are the same wrapper.

Do not invent node IDs. Do not write Cypher.

Examples:
{"target":"locate","subject":"software developer","kind":"occupation"}
{"target":"locate","subject":"firefighter","kind":"occupation"}
{"target":"connect","subject":"software developer","kind":"occupation",
 "rel_types":["HAS_SKILL"],"relation_filter":"essential"}
{"target":"pathfind","subject":"data analyst","secondary_subject":"data scientist"}

Same connect shape for: "what skills does a X need", "what skills I need to be
a X", "skills I need to become a X", "I want to be a X".
Same locate shape for: "what is a X", "what does a X do", "where is X".
"""

_PLAN_RANK = {
    CAPABILITY_LOCATE: 0,
    CAPABILITY_CONNECT: 1,
    CAPABILITY_PATHFIND: 2,
}

_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


class PlanDraft(BaseModel):
    """Validated planner JSON. Invalid drafts fall back to keyword planning."""

    model_config = ConfigDict(extra="ignore")

    target: Capability
    subject: str | None = None
    secondary_subject: str | None = None
    kind: str | None = None
    rel_types: tuple[str, ...] | None = None
    relation_filter: str | None = None
    suites: tuple[str, ...] = Field(default=(ESCO_SUITE_NAME,))

    @field_validator("subject", "secondary_subject", "kind", "relation_filter", mode="before")
    @classmethod
    def blank_to_none(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("rel_types", mode="before")
    @classmethod
    def empty_rel_types(cls, value: object) -> object:
        if value is None:
            return None
        if isinstance(value, list) and not value:
            return None
        return value

    @field_validator("suites", mode="before")
    @classmethod
    def default_suites(cls, value: object) -> object:
        if value is None or value == [] or value == ():
            return (ESCO_SUITE_NAME,)
        return value


@dataclass(frozen=True)
class InterpretedPlan:
    plan: ExecutionPlan
    draft: PlanDraft | None
    heuristic: bool
    stage: StageUsage | None


def uses_llm_planner(client: LLMClient) -> bool:
    return getattr(client, "provider", "none") != "none"


def parse_plan_text(text: str) -> PlanDraft:
    cleaned = text.strip()
    fenced = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned)
    match = _JSON_OBJECT.search(fenced)
    if match is None:
        raise ValueError("planner response did not contain a JSON object")
    payload = json.loads(match.group(0))
    if not isinstance(payload, dict):
        raise ValueError("planner JSON must be an object")
    return PlanDraft.model_validate(payload)


def _prefer_stronger_heuristic_target(question: str, draft: PlanDraft) -> PlanDraft:
    """If keywords clearly ask for more map work than the model chose, upgrade.

    Stops a skills question being planned as locate-only. Never downgrades.
    """
    hinted = classify_capability(question)
    if _PLAN_RANK[hinted] <= _PLAN_RANK[draft.target]:
        return draft
    return draft.model_copy(update={"target": hinted})


def connect_request_from_draft(draft: PlanDraft) -> ConnectRequest | None:
    if not draft.subject:
        return None
    rel_types = draft.rel_types
    if rel_types is None and (draft.relation_filter or draft.target == "connect"):
        rel_types = ("HAS_SKILL",)
    return ConnectRequest(
        subject=draft.subject.strip(),
        rel_types=rel_types or (),
        relation_kind=draft.relation_filter,
    )


def interpret_question(
    question: str,
    *,
    suite_name: str,
    llm_client: LLMClient,
    forced_capability: Capability | None = None,
) -> InterpretedPlan:
    if forced_capability is not None:
        return InterpretedPlan(
            plan=build_plan_for_capability(forced_capability, suites=(suite_name,)),
            draft=None,
            heuristic=True,
            stage=None,
        )
    if not uses_llm_planner(llm_client):
        return InterpretedPlan(
            plan=build_plan(question, suites=(suite_name,)),
            draft=None,
            heuristic=True,
            stage=None,
        )

    try:
        result, stage = measure_complete(
            llm_client,
            [
                Message(role="system", content=PLAN_SYSTEM),
                Message(role="user", content=question),
            ],
            stage="intent",
        )
    except RuntimeError:
        return InterpretedPlan(
            plan=build_plan(question, suites=(suite_name,)),
            draft=None,
            heuristic=True,
            stage=None,
        )
    try:
        draft = parse_plan_text(result.text)
        draft = _prefer_stronger_heuristic_target(question, draft)
        plan = build_plan_for_capability(draft.target, suites=draft.suites or (suite_name,))
        return InterpretedPlan(plan=plan, draft=draft, heuristic=False, stage=stage)
    except (ValueError, ValidationError, json.JSONDecodeError):
        return InterpretedPlan(
            plan=build_plan(question, suites=(suite_name,)),
            draft=None,
            heuristic=True,
            stage=stage,
        )
