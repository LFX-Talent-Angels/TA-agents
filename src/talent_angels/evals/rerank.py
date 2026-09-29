"""Score Locate's post-search rerank/auto-select/grouping layer (skills/locate/rank.py).

Distinct from LocateMetrics: that scores the suite's raw search_nodes hit
before any rerank runs (tests/integration/test_golden_locate.py calls
locate() directly). This scores what group_and_sort_locate does to that raw
hit before it ever reaches a user — the layer the Sprint 6/7 notes call
"reordering results so the best match comes first" and "auto-select on any
clear tier gap".

Two failure modes matter equally and are tracked separately:
- false ambiguous: a real tier gap exists, but the reranker still asks the
  user to pick (a needless picker).
- false auto-select: no real tier gap exists, but the reranker silently
  picked one candidate anyway (a wrong-answer risk, not just an annoyance).
Both are visible as one combined boolean per case (auto_select decision
correctness) plus, where the case has a genuine winner, whether that winner
is the *right* one.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RerankMetrics:
    decision_hits: int = 0  # "ambiguous" (yes/no) matched what was expected
    decision_questions: int = 0
    winner_hits: int = 0  # for non-ambiguous cases: the auto-selected id was right
    winner_questions: int = 0
    group_label_hits: int = 0
    group_label_questions: int = 0

    def observe(
        self,
        *,
        expected_ambiguous: bool,
        actual_ambiguous: bool,
        expected_top_id: str | None,
        actual_top_id: str | None,
        expected_group_label: str | None = None,
        actual_group_label: str | None = None,
    ) -> None:
        self.decision_questions += 1
        if actual_ambiguous == expected_ambiguous:
            self.decision_hits += 1

        if not expected_ambiguous:
            self.winner_questions += 1
            if expected_top_id is not None and actual_top_id == expected_top_id:
                self.winner_hits += 1

        if expected_group_label is not None:
            self.group_label_questions += 1
            if actual_group_label == expected_group_label:
                self.group_label_hits += 1

    def as_dict(self) -> dict[str, float | int | None]:
        return {
            "decision_accuracy": (
                self.decision_hits / self.decision_questions if self.decision_questions else None
            ),
            "decision_questions": self.decision_questions,
            "winner_accuracy": (
                self.winner_hits / self.winner_questions if self.winner_questions else None
            ),
            "winner_questions": self.winner_questions,
            "group_label_accuracy": (
                self.group_label_hits / self.group_label_questions
                if self.group_label_questions
                else None
            ),
            "group_label_questions": self.group_label_questions,
        }
