"""Separate unique-match accuracy from ambiguous candidate recall."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass
class LocateMetrics:
    hit_at_1_hits: int = 0
    hit_at_1_questions: int = 0
    candidate_recall_hits: int = 0
    candidate_recall_questions: int = 0

    def observe(self, metric: str, expected_id: str | None, candidate_ids: Sequence[str]) -> None:
        """Record one golden case without treating ambiguity as unique rank 1."""
        if metric not in {"hit_at_1", "candidate_recall"}:
            raise ValueError(f"unknown Locate metric: {metric!r}")
        if expected_id is None:
            return

        if metric == "hit_at_1":
            self.hit_at_1_questions += 1
            if candidate_ids and candidate_ids[0] == expected_id:
                self.hit_at_1_hits += 1
        else:
            self.candidate_recall_questions += 1
            if expected_id in candidate_ids:
                self.candidate_recall_hits += 1

    def as_dict(self) -> dict[str, float | int | None]:
        return {
            "hit_at_1_accuracy": (
                self.hit_at_1_hits / self.hit_at_1_questions if self.hit_at_1_questions else None
            ),
            "hit_at_1_questions": self.hit_at_1_questions,
            "candidate_recall": (
                self.candidate_recall_hits / self.candidate_recall_questions
                if self.candidate_recall_questions
                else None
            ),
            "candidate_recall_questions": self.candidate_recall_questions,
        }
