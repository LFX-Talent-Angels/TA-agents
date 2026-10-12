"""Model call: pick which of a unit's candidates match the line (keys.CHOOSE_VERSION).

The model only ever sees a numbered list (1..N, in `outcome.candidates`
order) plus descriptions pulled transiently off `SearchOutcome` — never
stored in the ledger (licensing: pointer, not payload). Code validates every
pick before it reaches `verdict.decide`: an out-of-range or duplicate `n`,
or an occupation/non-occupation kind mismatch, is dropped rather than
trusted.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from talent_angels.assistant.llm_call import measure_complete
from talent_angels.llm import LLMClient, LLMError, Message
from talent_angels.runlog import StageUsage
from talent_angels.tagging import jsonio
from talent_angels.tagging.models import CandidateRef, SearchOutcome, TagUnit
from talent_angels.tagging.prompts import choose_message
from talent_angels.tagging.search import _keep_by_routed_kind

_MAX_PICKS = 3


class Pick(BaseModel):
    n: int
    covers: str = ""

    @field_validator("covers", mode="before")
    @classmethod
    def _no_covers_is_empty(cls, value: object) -> object:
        # An exact pick has nothing partial to explain; models write `null`
        # there, and rejecting the whole reply for it threw away good matches.
        return "" if value is None else value


class Choice(BaseModel):
    """The choose call's raw reply, trusted only as a labelled claim — never taxonomy fact."""

    fit: Literal["exact", "partial", "none"]
    picks: list[Pick] = Field(default_factory=list)
    reason: str = ""
    missing_skill: str | None = None


@dataclass
class ValidatedChoice:
    """`Choice` after code dropped every pick that isn't actually usable."""

    fit: Literal["exact", "partial", "none"]
    picks: list[tuple[CandidateRef, str]]
    reason: str
    missing_skill: str | None


def _kind_compatible(candidate: CandidateRef, unit_kind: str) -> bool:
    routed = "occupation" if unit_kind.casefold() == "occupation" else "task"
    return _keep_by_routed_kind(candidate.kind, routed)


def validate(candidates: list[CandidateRef], unit_kind: str, choice: Choice) -> ValidatedChoice:
    """Drop out-of-range/duplicate/kind-incompatible picks; cap at `_MAX_PICKS`."""
    seen: set[int] = set()
    picks: list[tuple[CandidateRef, str]] = []
    for pick in choice.picks:
        if pick.n < 1 or pick.n > len(candidates):
            continue
        if pick.n in seen:
            continue
        candidate = candidates[pick.n - 1]
        if not _kind_compatible(candidate, unit_kind):
            continue
        seen.add(pick.n)
        picks.append((candidate, pick.covers))
        if len(picks) >= _MAX_PICKS:
            break
    return ValidatedChoice(
        fit=choice.fit, picks=picks, reason=choice.reason, missing_skill=choice.missing_skill
    )


def _call(
    client: LLMClient,
    unit: TagUnit,
    outcome: SearchOutcome,
    descriptions: dict[str, str],
    *,
    suite_display: str,
    hint: str,
    system: str,
    document_label: str,
) -> tuple[Choice, StageUsage]:
    candidates_with_desc = [
        (candidate.label, candidate.kind, descriptions.get(candidate.id, ""))
        for candidate in outcome.candidates
    ]
    user_message = choose_message(unit, candidates_with_desc, suite_display, document_label, hint)
    messages = [
        Message(role="system", content=system),
        Message(role="user", content=user_message),
    ]
    result, stage = measure_complete(client, messages, stage="tag_choose")
    try:
        reply = jsonio.parse_reply(result.text, Choice)
    except ValueError as exc:
        # The call happened and was billed even though its reply is unusable.
        raise _BadReply(stage) from exc
    return reply, stage


class _BadReply(ValueError):
    """A reply that parsed badly — carries the usage so the cost still counts."""

    def __init__(self, stage: StageUsage) -> None:
        super().__init__("unusable choose reply")
        self.stage = stage


def choose(
    client: LLMClient,
    unit: TagUnit,
    outcome: SearchOutcome,
    descriptions: dict[str, str],
    *,
    suite_display: str,
    hint: str = "",
    widened: bool = False,
    system: str,
    document_label: str,
) -> tuple[Choice | None, list[StageUsage]]:
    """One choose call, one retry on a bad reply, then `None` (caller treats as `error`).

    `system` is `prompts.choose_system(profile.document_type)`, computed once per run by
    the caller; `document_label` is `profile.document_label`.
    """
    del widened  # affects verdict.decide's policy branch, not the prompt itself
    stages: list[StageUsage] = []
    for _attempt in range(2):
        try:
            reply, stage = _call(
                client,
                unit,
                outcome,
                descriptions,
                suite_display=suite_display,
                hint=hint,
                system=system,
                document_label=document_label,
            )
        except _BadReply as exc:
            stages.append(exc.stage)
            continue
        except (LLMError, ValueError):
            continue
        stages.append(stage)
        return reply, stages
    return None, stages
