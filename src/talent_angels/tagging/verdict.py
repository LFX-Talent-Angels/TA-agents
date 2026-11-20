"""Pure policy decisions from a `SearchOutcome` (+ a model `Choice`, PR3).

Full policy **tag-v2** (tag-v1 of PILOT-PLAN.md plus the top-3 corroboration
in rule 4; copied here so the rule text ships with the code it governs):

1. Suite failed to open/search -> `unavailable`; every routed kind
   `unknown_kind` -> `n/a`. Neither is cached (see `runner.py`).
2. `--no-llm`: no candidates -> `gap` (`no_candidates`); else `review` with
   top 3 (`search_only`).
3. Model/JSON failure after one retry -> `error` (`model_failed`, not cached).
4. `fit=exact` and every pick corroborated -> `accepted`; else `review`.
   Corroborated = result had no `alias_unconfirmed` AND (`lexical_rank` tier
   <= 3 for some rewrite label, OR `best_rank <= 3`, OR `hits >= 2` and
   `best_rank <= 5`). tag-v2 added `best_rank <= 3`: O*NET tasks are long
   sentences no label ever matches word for word, so a search's own top-3
   is the graph-side signal that agrees with the model.
5. `fit=partial` with >= 1 pick carrying a non-empty `covers` -> `adjacent`
   (max 3 picks).
6. `fit=none`/no valid picks -> one widened attempt (description + parent +
   Element context); still none -> `gap` with `missing_skill` labelled
   "proposed by TA — not taxonomy fact".

Definitions/summary kinds are excluded from the gap list and precision stats
(`select.py`'s `report_kinds`, applied by the writer, not here).

The model's `fit`/`missing_skill` are always a labelled, categorical input
here, never a probability: rule 4 treats "exact" as a claim to corroborate
against the graph, not a confidence score to trust.
"""

from __future__ import annotations

from collections.abc import Sequence

from talent_angels.contracts import NodeRef
from talent_angels.skills.locate.rank import lexical_rank
from talent_angels.tagging.choose import ValidatedChoice
from talent_angels.tagging.models import CandidateRef, SearchOutcome, Verdict, VerdictResult

_MAX_ALTERNATIVES = 3
_CORROBORATION_TIER = 3
_CORROBORATION_HITS = 2
_CORROBORATION_RANK = 5
_TOP_RANK = 3


def _corroborated(candidate: CandidateRef, labels: Sequence[str]) -> bool:
    if candidate.alias_unconfirmed:
        return False
    if labels:
        node = NodeRef(
            id=candidate.id,
            suite=candidate.suite,
            source=candidate.suite,
            source_id=candidate.source_id,
            kind=candidate.kind,
            pref_label=candidate.label,
        )
        if min(lexical_rank(label, node)[0] for label in labels) <= _CORROBORATION_TIER:
            return True
    if candidate.best_rank <= _TOP_RANK:
        return True
    return candidate.hits >= _CORROBORATION_HITS and candidate.best_rank <= _CORROBORATION_RANK


def decide(
    outcome: SearchOutcome,
    choice: ValidatedChoice | None = None,
    *,
    llm_on: bool = False,
    widened: bool = False,
    labels: Sequence[str] = (),
) -> VerdictResult:
    if outcome.status == "unavailable":
        return VerdictResult(verdict=Verdict.UNAVAILABLE, reason_code="suite_unavailable")
    if outcome.status == "n/a":
        return VerdictResult(verdict=Verdict.NOT_APPLICABLE, reason_code="kind_not_supported")

    if not outcome.candidates:
        # True in both modes: a model call over zero candidates is wasted spend.
        return VerdictResult(verdict=Verdict.GAP, reason_code="no_candidates")

    if not llm_on:
        top, rest = outcome.candidates[0], outcome.candidates[1:4]
        return VerdictResult(
            verdict=Verdict.REVIEW, reason_code="search_only", tags=[top], alternatives=rest
        )

    if choice is None:
        return VerdictResult(verdict=Verdict.ERROR, reason_code="model_failed")

    picked_ids = {candidate.id for candidate, _ in choice.picks}
    alternatives = [c for c in outcome.candidates if c.id not in picked_ids][:_MAX_ALTERNATIVES]

    if choice.fit == "exact" and choice.picks:
        tags = [candidate for candidate, _ in choice.picks]
        if all(_corroborated(candidate, labels) for candidate, _ in choice.picks):
            return VerdictResult(
                verdict=Verdict.ACCEPTED,
                reason_code="model_exact_corroborated",
                tags=tags,
                alternatives=alternatives,
                model_reason=choice.reason,
            )
        return VerdictResult(
            verdict=Verdict.REVIEW,
            reason_code="model_exact_uncorroborated",
            tags=tags,
            alternatives=alternatives,
            model_reason=choice.reason,
        )

    if choice.fit == "partial":
        covering = [(candidate, covers) for candidate, covers in choice.picks if covers]
        if covering:
            return VerdictResult(
                verdict=Verdict.ADJACENT,
                reason_code="model_partial",
                tags=[candidate for candidate, _ in covering],
                covers={candidate.id: covers for candidate, covers in covering},
                alternatives=alternatives,
                missing_skill=choice.missing_skill,
                model_reason=choice.reason,
            )

    # fit=none, or fit=exact/partial with no valid (post-validation) picks.
    if not widened:
        return VerdictResult(
            verdict=Verdict.REVIEW,
            reason_code="needs_widening",
            needs_widening=True,
            alternatives=alternatives,
            missing_skill=choice.missing_skill,
            model_reason=choice.reason,
        )
    return VerdictResult(
        verdict=Verdict.GAP,
        reason_code="model_none",
        alternatives=alternatives,
        missing_skill=choice.missing_skill,
        model_reason=choice.reason,
    )
