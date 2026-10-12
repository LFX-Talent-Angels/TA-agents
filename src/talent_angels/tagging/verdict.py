"""Pure policy decisions from a `SearchOutcome` — never a model call.

Full policy **tag-v1** (PILOT-PLAN.md "Verdict policy tag-v1"; copied here so
the rule text ships with the code it governs):

1. Suite failed to open/search -> `unavailable`; every routed kind
   `unknown_kind` -> `n/a`. Neither is cached (see `runner.py`).
2. `--no-llm`: no candidates -> `gap` (`no_candidates`); else `review` with
   top 3 (`search_only`).
3. Model/JSON failure after one retry -> `error` (not cached).
4. `fit=exact` and every pick corroborated -> `accepted`; else `review`.
   Corroborated = result had no `alias_unconfirmed` AND (`lexical_rank` tier
   <= 3 for some rewrite label, OR pick in re-ranked top 5 for >= 2 distinct
   queries).
5. `fit=partial` with >= 1 pick carrying a non-empty `covers` -> `adjacent`
   (max 3 picks).
6. `fit=none`/no valid picks -> one widened attempt (description + parent +
   Element context); still none -> `gap` with `missing_skill` labelled
   "proposed by TA — not taxonomy fact".

Definitions/summary kinds are excluded from the gap list and precision stats
(`select.py`'s `report_kinds`, applied by the writer, not here).

Rules 3-6 need a model `Choice` (PR3) and are not implemented yet — this
module only ever receives `llm_on=False` today. The model's `fit` is always a
labelled, categorical input here, never a probability: rule 4 treats "exact"
as a claim to corroborate against the graph, not a confidence score to trust.
"""

from __future__ import annotations

from talent_angels.tagging.models import SearchOutcome, Verdict, VerdictResult


def decide(
    outcome: SearchOutcome,
    choice: object | None = None,
    *,
    llm_on: bool = False,
    widened: bool = False,
) -> VerdictResult:
    """Rules 1-2 (no-LLM) now; `llm_on=True` raises until PR3 fills rules 3-6."""
    del choice, widened  # PR3 parameters; unused until rules 3-6 land

    if outcome.status == "unavailable":
        return VerdictResult(
            verdict=Verdict.UNAVAILABLE, reason_code="suite_unavailable", tags=[], alternatives=[]
        )
    if outcome.status == "n/a":
        return VerdictResult(
            verdict=Verdict.NOT_APPLICABLE,
            reason_code="kind_not_supported",
            tags=[],
            alternatives=[],
        )
    if llm_on:
        raise NotImplementedError("PR3")

    if not outcome.candidates:
        return VerdictResult(
            verdict=Verdict.GAP, reason_code="no_candidates", tags=[], alternatives=[]
        )

    top, rest = outcome.candidates[0], outcome.candidates[1:4]
    return VerdictResult(
        verdict=Verdict.REVIEW, reason_code="search_only", tags=[top], alternatives=list(rest)
    )
