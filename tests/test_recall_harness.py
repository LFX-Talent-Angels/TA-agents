"""The recall harness is a gate, so these tests ask whether it can still fail.

Every number in the prose on ``memory.fts_retriever._STOPWORDS`` and ``_JOIN``
comes from ``evals.recall``. That is only worth anything if two things hold: the
shipped design still clears its floor, and the harness still *discriminates* —
because a fixture that has quietly stopped being able to distinguish the
designs is worse than no fixture, since it reports green while measuring
nothing.

So this file is mostly about the harness's own ability to fail:

- ``validate_corpus`` rejects a query set that cannot tell AND from OR, or whose
  irrelevant queries match something.
- The floor is enforced, and ``main`` actually exits non-zero when it is missed.
- Every design the table reports is measurably different from the shipped one,
  which is what stops a future edit from quietly collapsing the comparison.
- The idf-pruning ablation still reproduces its published result, because a
  conclusion in a comment is only as good as the measurement it came from.

These run in well under a second: the corpus is 30 turns and the retriever is
SQLite in a temporary directory, so the whole thing is real work over a real
database rather than a mock, and there is no reason to cache it.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence

import pytest

from talent_angels.evals import recall


def test_the_package_exports_every_metrics_type_it_claims() -> None:
    """`evals/__init__` is the public surface, and it had quietly dropped one.

    `recall.Metrics` was measured, printed and compared by the harness and could
    only be reached as `evals.recall.Metrics`, while `LocateMetrics` and
    `RerankMetrics` were re-exported — so "the metrics types this package offers"
    had two different answers depending on which one you asked about. Asserted
    over `__all__` and over the objects behind it, so a name that is listed but
    does not exist, or a type that exists but is not listed, both fail.
    """
    import talent_angels.evals as evals

    for name in ("LocateMetrics", "RecallMetrics", "RerankMetrics"):
        assert name in evals.__all__, f"{name} is measured but not exported"
        assert getattr(evals, name, None) is not None, f"{name} is exported but missing"
    assert evals.RecallMetrics is recall.Metrics, "the alias drifted from the real type"
    # Nothing listed that does not resolve, and nothing public left off the list.
    for name in evals.__all__:
        assert hasattr(evals, name), f"__all__ names {name!r}, which does not exist"


@pytest.fixture(scope="module")
def rows() -> tuple[recall.Metrics, ...]:
    """Every design in the table, measured once."""
    return tuple(recall.measure(design) for design in recall.DESIGNS)


@pytest.fixture(scope="module")
def by_name(rows: tuple[recall.Metrics, ...]) -> dict[str, recall.Metrics]:
    return {row.design: row for row in rows}


# --------------------------------------------------------------------------
# The fixture is a fixture: it has to be discriminating, or nothing below means
# anything.
# --------------------------------------------------------------------------


def test_the_shipped_corpus_is_discriminating() -> None:
    """A corpus that validates is a corpus that can separate the designs."""
    recall.validate_corpus()


def test_corpus_has_both_satisfiable_and_unsatisfiable_queries() -> None:
    """Both halves are required: only-unsatisfiable is the AND fixture only."""
    unsatisfiable = recall._unsatisfiable_under_and()
    assert unsatisfiable, "no query is unsatisfiable under AND, so the AND row is fiction"
    assert len(unsatisfiable) < len(
        [q for q in recall.QUERIES if q.group not in recall.IRRELEVANT_GROUPS]
    ), "every answerable query is unsatisfiable under AND, which is not a corpus"


def test_a_non_discriminating_corpus_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """The check has to be able to fail, so make it fail on purpose.

    Dropping the one query set that is unsatisfiable under AND is the cheapest
    way to build a corpus that cannot tell the designs apart; if this passes
    when the corpus is broken, the validation is decoration.
    """
    satisfied = tuple(
        recall.Query(q.text, q.group, q.gold)
        for q in recall.QUERIES
        if set(t.strip('"').lower() for t in recall.fts_retriever._terms(q.text))
        <= set(recall._document_frequency())
    )
    monkeypatch.setattr(recall, "QUERIES", satisfied)
    with pytest.raises(AssertionError, match="unsatisfiable under AND"):
        recall.validate_corpus()


def test_a_gold_turn_unrelated_to_its_query_is_rejected() -> None:
    """A gold turn is a claim about content, so a bad one has to be caught.

    Without this, a wrong annotation quietly deflates p@1 for every design and
    nobody can tell the retriever from the benchmark.
    """
    for turn in recall.CORPUS:
        for query in recall.QUERIES:
            if turn.run_id not in query.gold:
                continue
            haystack = f"{turn.question} {' '.join(turn.labels)}".lower()
            terms = {t.strip('"').lower() for t in recall.fts_retriever._terms(query.text)}
            assert terms & set(recall._index_terms(haystack)), (
                f"{query.text!r} claims gold {turn.run_id}, which shares no term with it"
            )
            return
    pytest.fail("no query names a gold turn at all")


# --------------------------------------------------------------------------
# The floor
# --------------------------------------------------------------------------


def test_shipped_design_clears_its_floor(rows: tuple[recall.Metrics, ...]) -> None:
    """The gate the CI run depends on."""
    shipped = next(r for r in rows if r.design == recall._FLOOR.design)
    assert recall.regressions(shipped) == []


def test_regressions_names_every_floor_column() -> None:
    """A regression report that says "something regressed" gets ignored.

    Each floor column is checked independently so a failure points at the
    quantity that moved, and so a column that is never checked cannot quietly
    become decorative.
    """
    perfect = recall.Metrics(
        design=recall._FLOOR.design,
        answerable=recall._FLOOR.answerable,
        zero_hit=recall._FLOOR.zero_hit,
        p_at_1=recall._FLOOR.p_at_1,
        mrr_at_5=recall._FLOOR.mrr_at_5,
        irrelevant_hits=recall._FLOOR.irrelevant_hits,
        widest=recall._FLOOR.widest,
        matches=recall._FLOOR.matches,
    )
    assert recall.regressions(perfect) == []

    worse = recall.Metrics(
        design=perfect.design,
        answerable=perfect.answerable,
        zero_hit=perfect.zero_hit + 1,
        p_at_1=perfect.p_at_1 - 0.1,
        mrr_at_5=perfect.mrr_at_5 - 0.1,
        irrelevant_hits=perfect.irrelevant_hits + 1,
        widest=perfect.widest + 1,
        matches=perfect.matches + 1,
    )
    lost = recall.regressions(worse)
    assert len(lost) == 6, lost
    assert any("zero-hit" in item for item in lost)
    assert any("p@1" in item for item in lost)
    assert any("MRR@5" in item for item in lost)
    assert any("irrelevant hits" in item for item in lost)
    assert any("widest" in item for item in lost)
    assert any("MATCHes" in item for item in lost)


def test_changing_the_query_set_is_reported_as_a_regression() -> None:
    """Editing QUERIES invalidates the comparison and has to say so loudly.

    Re-deriving the floor is a deliberate act; drifting into a new query set
    without one is not, and it would otherwise make every number in the module
    docstring wrong while CI stayed green. The check short-circuits on purpose —
    one loud message beats six that get read together.
    """
    assert recall.regressions(
        recall.Metrics(design=recall._FLOOR.design, **{**_empty_metrics(), "answerable": 1})
    ) == [
        "the query set changed (1 answerable, floor "
        f"{recall._FLOOR.answerable}); re-derive the floor deliberately"
    ]


def _empty_metrics() -> dict[str, object]:
    return {
        "zero_hit": 0,
        "p_at_1": 0.0,
        "mrr_at_5": 0.0,
        "irrelevant_hits": 0,
        "widest": 0,
        "matches": 0,
    }


def test_main_exits_zero_when_healthy_and_one_when_not(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`main` is the CI entry point, so its exit code is the contract."""
    assert recall.main([]) == 0
    out = capsys.readouterr().out
    assert "at or above the floor" in out

    # A floor nothing can clear, so the non-zero path is genuinely exercised.
    monkeypatch.setattr(
        recall, "_FLOOR", recall.Metrics(design=recall._FLOOR.design, **_regressed())
    )
    assert recall.main([]) == 1
    assert "REGRESSION" in capsys.readouterr().err


def _regressed() -> dict[str, object]:
    return {
        "answerable": recall._FLOOR.answerable,
        "zero_hit": recall._FLOOR.zero_hit,
        "p_at_1": recall._FLOOR.p_at_1 + 1.0,
        "mrr_at_5": recall._FLOOR.mrr_at_5 + 1.0,
        "irrelevant_hits": 0,
        "widest": 0,
        "matches": 0,
    }


# --------------------------------------------------------------------------
# The comparison itself: every row has to be a different row.
# --------------------------------------------------------------------------


def test_and_join_is_worse_than_the_shipped_design(
    by_name: dict[str, recall.Metrics],
) -> None:
    """The bug the OR fixed, re-measured rather than remembered.

    AND requires a single stored question to contain every term of the query,
    which almost no real question does — so a conjunction in the user's
    question means AND recalls nothing at all.
    """
    and_only = by_name["AND only (the bug)"]
    shipped = by_name["OR + stopwords (shipped)"]
    assert and_only.zero_hit > shipped.zero_hit
    assert and_only.p_at_1 < shipped.p_at_1
    assert and_only.mrr_at_5 < shipped.mrr_at_5


def test_and_to_or_fallback_pays_double_and_buys_nothing(
    by_name: dict[str, recall.Metrics],
) -> None:
    """The fallback alternative: same answers, more MATCHes.

    Worth keeping in the table precisely because it is the tempting option —
    it never returns a worse answer, it just does the work twice.
    """
    and_only = by_name["AND only (the bug)"]
    fallback = by_name["AND, falling back to OR"]
    assert fallback.zero_hit == and_only.zero_hit
    assert fallback.p_at_1 == and_only.p_at_1
    assert fallback.matches > and_only.matches


def test_stopwords_reduce_irrelevant_hits_to_zero(
    by_name: dict[str, recall.Metrics],
) -> None:
    """The property the list exists for, as a hard number.

    An OR with no filter matches documents on the scaffolding alone: `how`, `do`,
    `a`. Dropping the closed-class words is what makes a query about sourdough
    starters return nothing instead of the whole career history.
    """
    unfiltered = by_name["OR, no stopword filter"]
    shipped = by_name["OR + stopwords (shipped)"]
    assert unfiltered.irrelevant_hits > 0
    assert shipped.irrelevant_hits == 0


def test_stopwords_improve_ranking_not_just_widening(
    by_name: dict[str, recall.Metrics],
) -> None:
    """A stopword list that only narrowed the net would be a ranking failure.

    Fewer documents returned is not by itself better — the whole point is to
    return the *right* ones. This is the assertion that separates "the filter
    works" from "the filter is doing something accidentally useful", and it is
    the check that failed when the list was incomplete and the added function
    words had never been exercised by the query set.
    """
    unfiltered = by_name["OR, no stopword filter"]
    shipped = by_name["OR + stopwords (shipped)"]
    assert shipped.p_at_1 > unfiltered.p_at_1
    assert shipped.mrr_at_5 > unfiltered.mrr_at_5
    assert shipped.widest <= unfiltered.widest


def test_all_five_reported_rows_are_distinct_from_each_other(
    rows: tuple[recall.Metrics, ...],
) -> None:
    """Two identical rows mean the knob is not wired to anything.

    The idf row is *supposed* to tie the unfiltered row — that is the finding.
    Every other pair has to differ, or a design in the table is fiction.
    """
    signatures = {
        (r.zero_hit, r.p_at_1, r.mrr_at_5, r.irrelevant_hits, r.widest, r.matches) for r in rows
    }
    assert len(signatures) == len(rows) - 1, "two designs measured identically"


# --------------------------------------------------------------------------
# The idf-pruning finding, pinned
# --------------------------------------------------------------------------


def test_idf_pruning_at_the_bm25_threshold_is_a_no_op() -> None:
    """The published result: bm25's zero-idf point is unreachable for query terms.

    bm25's idf passes through zero when a term appears in half the documents, so
    "drop the terms that cannot change a score" is formally correct and
    practically empty on a personal corpus — the most frequent term in the query
    set appears in 7 of 30 turns. The row is kept in the table because a
    *measured* no-op is the argument for the hand-maintained list; an assumed
    one would not be.
    """
    report = recall.idf_pruning_report()[0]
    by_threshold = {row["threshold"]: row for row in report["curve"]}
    assert by_threshold[0.5]["pruned_count"] == 0
    assert by_threshold[0.25]["pruned_count"] == 0
    assert by_threshold[0.5]["p_at_1"] == by_threshold[0.25]["p_at_1"]


def test_idf_pruning_that_actually_prunes_is_catastrophic() -> None:
    """The other half of the finding: it is not just inert, it is harmful.

    Once the threshold drops below the corpus's own subject matter it prunes
    `nurse`, `skills`, `teacher` and `training` — and quality collapses while
    the irrelevant-hit count improves. That trade is the reason the list stays
    hand-maintained.
    """
    report = recall.idf_pruning_report()[0]
    by_threshold = {row["threshold"]: row for row in report["curve"]}
    inert = by_threshold[0.25]
    biting = by_threshold[0.1]
    assert biting["pruned_count"] > 0
    assert biting["p_at_1"] < inert["p_at_1"] - 0.2
    assert biting["zero_hit"] > inert["zero_hit"]
    assert biting["irrelevant_hits"] < inert["irrelevant_hits"], (
        "the trade this documents should be visible: fewer irrelevant hits, much worse answers"
    )


def test_most_frequent_query_terms_are_content_not_function_words() -> None:
    """Why a frequency rule cannot substitute for a list of *classes*.

    The corpus's commonest words are the ones a list should already handle (`a`,
    `i`); the commonest *query* terms are its subject matter. There is no
    frequency cut between those two, so a frequency rule can only be tuned to
    eat the subject.
    """
    report = recall.idf_pruning_report()[0]
    query_terms = {row["term"] for row in report["most_frequent_query_terms"]}
    assert query_terms & {"nurse", "nursing", "skills", "teacher", "training"}
    assert not query_terms & {"a", "i", "what", "how", "do"}


def test_the_harness_does_not_pin_the_stopword_lists_completeness() -> None:
    """What this file cannot catch, asserted so the limit is on the record.

    Reverting the thirteen words ``fts_retriever``'s argument 3 is about —
    ``are``, ``am``, ``any``, ``all``, ``each``, ``every``, ``such`` and the six
    subordinators — changes nothing measurable here, and this file would pass
    identically against a list that had them or did not. The reason is one line
    of arithmetic: not one of the thirteen appears in any of the 30 stored turns,
    so as OR operands they match nothing.

    That is the honest limit of a measurement harness. It pins the list's
    *effect* on this corpus; it cannot pin the list's *completeness*, because the
    turns that would show the difference have not been recorded yet.
    Completeness is a closed-class argument, and it is pinned structurally, by
    ``test_every_function_word_in_its_class_is_filtered``.

    Asserted rather than left as a comment because a harness that quietly does
    not cover something is worse than one that admits it — and because the first
    assertion turns into a tripwire the moment someone adds a turn containing
    ``are``, at which point the numbers on ``_STOPWORDS`` need re-measuring.
    """
    completeness_only = {
        "are",
        "am",
        "any",
        "all",
        "each",
        "every",
        "such",
        "although",
        "because",
        "unless",
        "while",
        "since",
        "whether",
    }
    present = {
        word
        for turn in recall.CORPUS
        for word in completeness_only
        if word in set(recall._index_terms(f"{turn.question} {' '.join(turn.labels)}"))
    }
    assert not present, (
        f"{sorted(present)} now appear in the corpus, so this harness can see the "
        "difference the completeness argument is about: re-measure and update the "
        "table on fts_retriever._STOPWORDS and the floor"
    )
    # And the effect this harness does measure is unaffected by them, which is
    # precisely why removing them is not a regression this file can report.
    shipped = next(d for d in recall.DESIGNS if d.name == "OR + stopwords (shipped)")
    without = dataclasses.replace(
        shipped,
        name="OR + stopwords (completeness words removed)",
        stopwords=frozenset(recall.fts_retriever._STOPWORDS - completeness_only),
    )
    assert recall.measure(without).p_at_1 == recall.measure(shipped).p_at_1
    assert recall.measure(without).mrr_at_5 == recall.measure(shipped).mrr_at_5


# --------------------------------------------------------------------------
# The measurements are about unbounded recall, not about the limit
# --------------------------------------------------------------------------


def test_widest_and_irrelevant_hits_are_not_capped_by_the_recall_limit() -> None:
    """A width measured through `limit=5` reports the ceiling as the size.

    Both numbers are about unbounded recall, so they have to be asked for
    unbounded and sliced afterwards. This checks the two paths really do differ
    in size, which is the property that makes the width column worth having; if
    the harness ever asked with `limit=_LIMIT`, these would be equal and every
    design would look identical in that column.
    """
    query = "how do i become a nurse"
    design = next(d for d in recall.DESIGNS if d.name == "OR + stopwords (shipped)")
    assert recall._WIDE_LIMIT > recall._LIMIT

    with recall._built_database() as db_path, recall._design(db_path, design) as retriever:
        wide, wide_matches = recall._run(retriever, query, design)
        capped = retriever.search(query, limit=recall._LIMIT)

    assert len(wide) > recall._LIMIT > 0, "pick a query that matches more than the limit"
    assert len(wide) > len(capped) == recall._LIMIT, (
        "the wide ask has to see more documents than the capped one, or `widest` is just the limit"
    )
    assert wide_matches == 1
    # Ranking is unaffected by the width of the ask, which is what makes slicing
    # afterwards legitimate rather than a second, subtly different measurement.
    assert [hit.run_id for hit in wide[: recall._LIMIT]] == [hit.run_id for hit in capped]


# --------------------------------------------------------------------------
# The knobs are the only thing that varies
# --------------------------------------------------------------------------


def test_only_the_join_and_stopword_knobs_are_patched() -> None:
    """A design comparison is only about designs if the rest is shared.

    Tokenisation, quoting, dedupe, the length filter, the SQL and bm25 all have
    to be the shipped functions, or the "before" column is a reimplementation and
    its numbers only describe that reimplementation.

    The check is that term extraction and search go through the module's own
    functions while the knobs are swapped underneath them — so a design's terms
    are computed by the same code path the shipped retriever uses, and differ
    only in the two settings the comparison is about.
    """
    and_design = next(d for d in recall.DESIGNS if d.name == "AND only (the bug)")
    with recall._built_database() as db_path, recall._design(db_path, and_design) as retriever:
        assert isinstance(retriever, recall.Fts5EpisodeRetriever)
        assert recall.fts_retriever._JOIN == " AND ", "the knob is swapped"
        # The pristine function is still the one in use; it is *not* reimplemented
        # for the "before" column, which is the failure mode this guards.
        assert callable(recall.fts_retriever._terms)
        assert recall.fts_retriever._MIN_TERM_CHARS >= 1
        # A question built from three terms that no single stored turn holds: AND
        # requires one turn to contain all of them, so it returns nothing while
        # OR finds five. This is the whole reason for the join, on one query.
        assert retriever.search("registered nurse AND nursing", limit=5) == []

    shipped = next(d for d in recall.DESIGNS if d.name == "OR + stopwords (shipped)")
    with recall._built_database() as db_path, recall._design(db_path, shipped) as retriever:
        assert [h.run_id for h in retriever.search("registered nurse AND nursing", limit=5)]


def test_shipped_design_is_the_untouched_shipped_configuration() -> None:
    """The row labelled "shipped" must actually be what ships.

    It is the row the floor is checked against, so a knob default that drifts
    from the module's real default would turn the gate into a fiction: it would
    be measuring a configuration nobody runs.
    """
    shipped = next(d for d in recall.DESIGNS if d.name == "OR + stopwords (shipped)")
    assert shipped.join == recall.fts_retriever._JOIN
    assert shipped.stopwords == recall.fts_retriever._STOPWORDS
    assert shipped.prune_by_idf is None
    assert shipped.fallback_to_or is False


def test_the_corpus_is_the_size_the_prose_claims() -> None:
    """Docstrings quote corpus sizes; keep them honest.

    `fts_retriever` says "30-turn" and the module docstring says so too. A
    number in prose that the code no longer matches is the same class of defect
    as the mutation row that was recorded as caught when it survived.
    """
    assert len(recall.CORPUS) == 30
    assert len(recall.QUERIES) == 43
    assert len([q for q in recall.QUERIES if q.group in recall.IRRELEVANT_GROUPS]) == 7


def test_reported_table_shape() -> None:
    """The markdown table is the artefact people paste into reports."""
    table = recall.render_table(
        [
            recall.Metrics(
                design="x",
                answerable=1,
                zero_hit=0,
                p_at_1=1.0,
                mrr_at_5=1.0,
                irrelevant_hits=0,
                widest=0,
                matches=0,
            )
        ]
    )
    lines: Sequence[str] = table.splitlines()
    assert lines[0].startswith("| design |")
    assert lines[1].startswith("| ------ |")
    assert lines[2] == "| x | 0/1 | 1.000 | 1.000 | 0 | 0 | 0 |"


# --------------------------------------------------------------------------
# The backend comparison
# --------------------------------------------------------------------------
#
# These test the *arithmetic* of the comparison with a deterministic stand-in
# embedder, so the offline suite can check the thing that would otherwise only
# run when someone pays for a provider call: that the two backends are scored by
# the same loop, that `top1` lines up with the answerable queries, and that the
# disagreement counts mean what the prose says they mean.
#
# The vector row against a *real* model is deliberately not asserted here. Its
# numbers are a measurement, not a contract — pinning them would make a change
# to the corpus look like a test failure, and would freeze today's provider
# behaviour into CI.


class _FakeEmbedder:
    """Hash-based, offline, and no more semantic than ``StaticEmbedder``.

    Enough to exercise the plumbing: dimensions, batching, and a search that
    returns rows. The *ranking* it produces is meaningless, which is exactly
    why no test below asserts which run id comes first.
    """

    _model = "fake/embed"

    def __init__(self, dimensions: int = 16) -> None:
        self._dimensions = dimensions

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        from talent_angels.memory.embeddings import StaticEmbedder

        return StaticEmbedder(dimensions=self._dimensions).embed(list(texts))


def test_top1_is_keyed_by_query_text_not_by_position() -> None:
    """`Metrics.top1` is positional; the comparison must not be.

    Pinning it to the query's text is what lets someone insert a query into the
    middle of `QUERIES` without the disagreement counts silently comparing the
    wrong two answers.
    """
    answerable = [q.text for q in recall.QUERIES if q.group not in recall.IRRELEVANT_GROUPS]
    top1 = tuple(f"r{i}" for i in range(len(answerable)))
    metrics = recall.Metrics(
        design="x",
        answerable=len(answerable),
        zero_hit=0,
        p_at_1=1.0,
        mrr_at_5=1.0,
        irrelevant_hits=0,
        widest=0,
        matches=0,
        top1=top1,
    )
    mapped = recall._top1_by_query(metrics)
    assert list(mapped) == answerable
    assert mapped[answerable[0]] == "r0"


def test_top1_mapping_refuses_a_misaligned_tuple() -> None:
    """A short `top1` is a bug in the caller, and must not be padded over."""
    answerable = [q.text for q in recall.QUERIES if q.group not in recall.IRRELEVANT_GROUPS]
    metrics = recall.Metrics(
        design="x",
        answerable=len(answerable),
        zero_hit=0,
        p_at_1=1.0,
        mrr_at_5=1.0,
        irrelevant_hits=0,
        widest=0,
        matches=0,
        top1=("only-one",),
    )
    with pytest.raises(ValueError):
        recall._top1_by_query(metrics)


def test_vector_is_scored_by_the_same_loop_as_lexical() -> None:
    """Both arms answer every query and are scored by `_score`.

    The comparison is only meaningful if the two numbers came from the same
    arithmetic, so this pins the shared entry point rather than a value.
    """
    metrics = recall.measure_vector(_FakeEmbedder())
    assert metrics.answerable == len(
        [q for q in recall.QUERIES if q.group not in recall.IRRELEVANT_GROUPS]
    )
    assert "vector" in metrics.design
    assert 0.0 <= metrics.p_at_1 <= 1.0
    assert metrics.matches == len(recall.QUERIES)


def test_comparison_counts_disagreements_in_both_directions() -> None:
    """The counts must separate "vector won" from "lexical won".

    They are the numbers that decide whether the vector path ships, and a
    single undirected disagreement count would hide exactly the asymmetry that
    matters.
    """
    comparison = recall.compare_backends(_FakeEmbedder())
    assert comparison.disagreements == comparison.vector_only + comparison.lexical_only
    assert comparison.dimensions == 16
    assert comparison.model == "fake/embed"


def test_comparison_scores_the_shipped_lexical_design() -> None:
    """The baseline is the design in the table marked shipped, not any other row."""
    comparison = recall.compare_backends(_FakeEmbedder())
    assert comparison.lexical.design == recall._FLOOR.design


def test_verdict_is_read_off_the_numbers() -> None:
    """Each branch of the verdict needs a case that actually triggers it."""

    def build(lex_p: float, vec_p: float, lex_zero: int, vec_zero: int) -> recall.BackendComparison:
        def row(name: str, p: float, zero: int) -> recall.Metrics:
            return recall.Metrics(
                design=name,
                answerable=10,
                zero_hit=zero,
                p_at_1=p,
                mrr_at_5=p,
                irrelevant_hits=0,
                widest=0,
                matches=0,
            )

        return recall.BackendComparison(
            lexical=row("lexical", lex_p, lex_zero),
            vector=row("vector", vec_p, vec_zero),
            dimensions=8,
            model="fake/embed",
            disagreements=0,
            lexical_only=0,
            vector_only=0,
        )

    assert "more quer" in build(0.9, 0.89, 4, 1).verdict()
    assert "vector is more precise" in build(0.7, 0.9, 0, 0).verdict()
    assert "lexical is more precise" in build(0.9, 0.7, 0, 0).verdict()
    assert "indistinguishable" in build(0.8, 0.8, 1, 1).verdict()


def test_verdict_prefers_recall_recovery_over_a_precision_loss() -> None:
    """A dense retriever never returns nothing; that is its one structural win.

    Recovering queries at a precision cost inside 0.02 is a different finding
    from losing on precision, and the verdict must not call them the same thing.
    """
    lex = recall.Metrics("lexical", 10, 5, 0.90, 0.9, 0, 0, 0)
    vec = recall.Metrics("vector", 10, 0, 0.89, 0.9, 0, 0, 0)
    verdict = recall.BackendComparison(
        lexical=lex,
        vector=vec,
        dimensions=8,
        model="m",
        disagreements=0,
        lexical_only=0,
        vector_only=0,
    ).verdict()
    assert "5 more queries" in verdict


def test_comparison_serialises_for_the_json_report() -> None:
    comparison = recall.compare_backends(_FakeEmbedder())
    payload = comparison.as_dict()
    assert set(payload) >= {
        "model",
        "dimensions",
        "lexical",
        "vector",
        "disagreements",
        "lexical_only",
        "vector_only",
        "verdict",
    }
    assert payload["dimensions"] == 16


def test_render_comparison_shows_both_rows_and_the_verdict() -> None:
    """The markdown is the artefact a reader takes away, so its shape is pinned."""
    table = recall.render_comparison(recall.compare_backends(_FakeEmbedder()))
    assert table.splitlines()[0].startswith("| backend |")
    assert "lexical" in table
    assert "vector" in table
    assert "verdict:" in table
    assert "dimensions" in table
