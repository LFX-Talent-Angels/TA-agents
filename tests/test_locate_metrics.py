"""Locate benchmark metric semantics."""

import pytest

from talent_angels.evals.locate import LocateMetrics


def test_metrics_separate_hit_at_1_from_ambiguous_candidate_recall() -> None:
    metrics = LocateMetrics()

    metrics.observe("hit_at_1", "expected-a", ["expected-a", "other"])
    metrics.observe("hit_at_1", "expected-b", ["other", "expected-b"])
    metrics.observe("candidate_recall", "expected-c", ["other", "expected-c"])

    assert metrics.as_dict() == {
        "hit_at_1_accuracy": 0.5,
        "hit_at_1_questions": 2,
        "candidate_recall": 1.0,
        "candidate_recall_questions": 1,
    }


def test_metrics_ignore_not_found_cases_and_reject_unknown_metric() -> None:
    metrics = LocateMetrics()

    metrics.observe("hit_at_1", None, [])
    assert metrics.as_dict()["hit_at_1_accuracy"] is None

    with pytest.raises(ValueError, match="unknown Locate metric"):
        metrics.observe("precision", "expected", ["expected"])
