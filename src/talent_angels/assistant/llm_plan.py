"""LLM-authored typed plans with heuristic fallback."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from talent_angels.assistant.intent import (
    CAPABILITY_LOCATE,
    Capability,
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
- suites: array of attached suite names (runtime fills this; do not drop
  a suite unless the user named one taxonomy)

How to choose target:
- locate = only identify / define a node ("what is X", "where is X in ESCO")
- connect = neighbors, skills, hierarchy around one node
  ("what skills does X need", "essential skills", "neighbors of X")
- pathfind = a route or gap between TWO things
  ("path from A to B", "skill gap from A to B", "how to become X from Y")
- "X vs Y" or "X and Y" as two titles is locate

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
Same pathfind shape for: "path from X to Y", "skill gap from X to Y",
"how to become X from Y", "how do I move from X to Y".

profile_intent rules:
- Set "goal" when the user states a career destination:
  "my goal is X", "I want to become X", "I am working toward X".
  Set subject to ONLY the destination occupation name (e.g. "data scientist").
  target must be "locate". Example:
  "my goal is data scientist" →
    {"target":"locate","subject":"data scientist","kind":"occupation","profile_intent":"goal"}
- Set "reject" when user denies an occupational identity:
  "I am not a X", "that's not my job", "I don't work as X".
  Set subject to the rejected occupation name only.
  target must be "locate".
- Leave null for all other questions.

suite_override rules:
- Set suite_override to the suite name (lowercase: "onet", "esco") when the user asks to
  see results on a specific taxonomy or switch the view to a different source.
  Examples: "now show me the same on ESCO" → suite_override="esco",
  "show me this on O*NET" → suite_override="onet",
  "can I see the ESCO version?" → suite_override="esco",
  "switch to O*NET" → suite_override="onet".
- Leave null for all other requests.
"""

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
    suites: tuple[str, ...] = Field(default=())

    profile_intent: str | None = None
    suite_override: str | None = None

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
            return ()
        return value


@dataclass(frozen=True)
class InterpretedPlan:
    plan: ExecutionPlan
    draft: PlanDraft | None
    heuristic: bool
    stage: StageUsage | None


_GOAL_RE = re.compile(
    r"^(?:my\s+goal\s+is|i\s+want\s+to\s+become|i\s+am\s+working\s+toward)"
    r"\s+(?:a\s+|an\s+)?(.+)$",
    re.IGNORECASE,
)
_REJECT_RE = re.compile(
    r"^(?:i\s+am\s+not\s+an?\s+|that(?:'s|'s|\s+is)\s+not\s+my\s+(?:job|occupation|role)\s*|i\s+don't\s+work\s+as\s+(?:an?\s+)?)(.+)$",
    re.IGNORECASE,
)


def _profile_intent_heuristic(question: str) -> PlanDraft | None:
    """Return a PlanDraft for goal/reject patterns when the LLM planner fails."""
    m = _GOAL_RE.match(question.strip())
    if m:
        return PlanDraft(
            target=CAPABILITY_LOCATE,
            subject=m.group(1).strip(),
            kind="occupation",
            profile_intent="goal",
        )
    m = _REJECT_RE.match(question.strip())
    if m:
        return PlanDraft(
            target=CAPABILITY_LOCATE,
            subject=m.group(1).strip(),
            kind="occupation",
            profile_intent="reject",
        )
    return None


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


def connect_request_from_draft(
    draft: PlanDraft,
    *,
    skill_rel_types: tuple[str, ...] = ("HAS_SKILL", "USES_SOFTWARE"),
) -> ConnectRequest | None:
    if not draft.subject:
        return None
    # Always use suite-schema skill_rel_types for connect queries. The LLM planner
    # only knows ESCO's "HAS_SKILL" example — it cannot know suite-specific types
    # like O*NET's "USES_SOFTWARE". Ignore draft.rel_types for skill connects.
    rel_types: tuple[str, ...] | None = None
    if draft.target == "connect" or draft.relation_filter or draft.rel_types:
        rel_types = skill_rel_types
    return ConnectRequest(
        subject=draft.subject.strip(),
        rel_types=rel_types or (),
        relation_kind=draft.relation_filter,
    )


def interpret_question(
    question: str,
    *,
    llm_client: LLMClient,
    suite_name: str | None = None,
    suites: tuple[str, ...] | None = None,
    forced_capability: Capability | None = None,
) -> InterpretedPlan:
    selected = suites or ((suite_name,) if suite_name else (ESCO_SUITE_NAME,))
    if forced_capability is not None:
        return InterpretedPlan(
            plan=build_plan_for_capability(forced_capability, suites=selected),
            draft=None,
            heuristic=True,
            stage=None,
        )
    if not uses_llm_planner(llm_client):
        heuristic_draft = _profile_intent_heuristic(question)
        if heuristic_draft is not None:
            return InterpretedPlan(
                plan=build_plan_for_capability(heuristic_draft.target, suites=selected),
                draft=heuristic_draft,
                heuristic=True,
                stage=None,
            )
        return InterpretedPlan(
            plan=build_plan(question, suites=selected),
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
            plan=build_plan(question, suites=selected),
            draft=None,
            heuristic=True,
            stage=None,
        )
    try:
        draft = parse_plan_text(result.text)
        # Suite choice is owned by select_suites / the caller, not the model.
        plan = build_plan_for_capability(draft.target, suites=selected)
        return InterpretedPlan(plan=plan, draft=draft, heuristic=False, stage=stage)
    except (ValueError, ValidationError, json.JSONDecodeError):
        heuristic_draft = _profile_intent_heuristic(question)
        if heuristic_draft is not None:
            return InterpretedPlan(
                plan=build_plan_for_capability(heuristic_draft.target, suites=selected),
                draft=heuristic_draft,
                heuristic=True,
                stage=stage,
            )
        return InterpretedPlan(
            plan=build_plan(question, suites=selected),
            draft=None,
            heuristic=True,
            stage=stage,
        )
