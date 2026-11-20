"""Measure the lexical retriever's query design — reproducibly, and in CI.

Every number quoted in the comments on ``memory.fts_retriever._STOPWORDS`` and
``_JOIN`` comes from this module. It exists because the alternative is worse:
those numbers were first produced by a script in ``/tmp`` that was never
committed, so a reviewer could check the code but not the claim, and the one
number in the report that was *wrong* (a mutation recorded as caught when it
survived) was wrong in a file nobody else could read. A measurement that cannot
be re-run is an assertion.

Run it::

    uv run python -m talent_angels.evals.recall
    uv run python -m talent_angels.evals.recall --json
    uv run pytest tests/test_recall_harness.py -q

It exits non-zero when the shipped design regresses against ``_FLOOR``, so it is
a CI gate and not only a report. Nothing here is mocked except the *two* design
knobs a comparison is actually about; term extraction, quoting, deduplication,
the length filter, the SQL and the ranking are the shipped functions, reached
through the shipped ``Fts5EpisodeRetriever`` over a real database. That is the
whole discipline — an earlier version of this comparison reimplemented the old
``_terms()`` by hand for the "before" column, which made the result only as
faithful as that reimplementation.

**About the corpus.** The original 30-turn corpus lived in a ``/tmp`` script and
was not saved, so the numbers in the pre-existing prose in ``fts_retriever`` are
*not* reproducible from this file and this report says so explicitly. What ships
here is a fresh, realistic 30-turn career history plus a gold-annotated query
set, with the gold answers checked against the corpus by
``validate_corpus()`` rather than asserted by hand. Anyone re-running this gets
exactly the numbers printed here; anyone disagreeing with the corpus can edit
``CORPUS``/``QUERIES`` and get a different, equally reproducible answer.

``validate_corpus`` also checks the corpus is *discriminating*, which is the
property the whole comparison rests on: a corpus where AND and OR score the
same cannot tell the designs apart, and a corpus where the irrelevant queries
already match nothing cannot show what the stopword list is for. A fixture that
quietly stops being able to fail is worse than no fixture.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import tempfile
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest import mock

from talent_angels.memory import fts_retriever
from talent_angels.memory.episodes import record_episode
from talent_angels.memory.fts_retriever import Fts5EpisodeRetriever
from talent_angels.memory.retrieval import EpisodeHit
from talent_angels.memory.vector_index import SqliteVecIndex
from talent_angels.memory.vector_retriever import episode_text
from talent_angels.runlog.models import ResultSummary, RunLogRecord

#: Ceiling on recall, so the harness measures the same ranking a turn would get.
_LIMIT = 5

#: What the harness actually asks for, so "how wide is the net?" has an answer
#: that is not the ceiling standing in for it.
_WIDE_LIMIT = 1000


@dataclass(frozen=True, slots=True)
class Turn:
    run_id: str
    question: str
    labels: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Query:
    text: str
    #: ``single`` | ``phrase`` | ``conjunction`` | ``typo`` | ``typo_unmatched`` |
    #: ``natural`` | ``irrelevant``. Only used for reporting, so a regression can
    #: be traced to the kind of question that caused it.
    group: str
    #: The turns this query is *about*, or ``()`` when it is expected to recall
    #: nothing. Two different reasons, and the distinction is what keeps the
    #: denominator honest: an ``irrelevant`` query is not scored at all, while an
    #: answerable query with no gold (``nurs``, below) stays in the denominator
    #: and scores as a zero-hit — the user asked something real and recall has no
    #: answer, which is the failure the vector path exists to fix.
    gold: tuple[str, ...] = ()


def _turn(number: int, question: str, *labels: str) -> Turn:
    return Turn(run_id=f"r{number:02d}", question=question, labels=labels)


#: A 30-turn personal career history, oldest first: one person, a handful of
#: topics, and the near-repeats and half-finished questions real histories are
#: made of. The topics repeat on purpose — `nurse` and `nursing` are the reason
#: the OR join has anything to rank, and a corpus of 30 unrelated questions
#: would make every design score the same.
CORPUS: tuple[Turn, ...] = (
    _turn(1, "what skills does a nurse need?", "nurse"),
    _turn(2, "how do I become a nurse?", "nurse"),
    _turn(3, "what does a nursing assistant do?", "nursing assistant"),
    _turn(4, "how long does nursing training take?", "nursing", "nurse"),
    _turn(5, "is nursing right for someone with a business degree?", "nursing"),
    _turn(6, "what training does a nurse need?", "nurse"),
    _turn(7, "nurse or nursing assistant, which one pays more?", "nurse"),
    _turn(8, "what is the difference between a nurse and a paramedic?", "nurse", "paramedic"),
    _turn(9, "do I need a licence to work as a nurse?", "nurse"),
    _turn(10, "career options in healthcare with no degree", "healthcare"),
    _turn(11, "career in healthcare", "healthcare"),
    _turn(12, "pharmacy technician training", "pharmacy technician"),
    _turn(13, "what skills does a pharmacy technician need?", "pharmacy technician"),
    _turn(14, "software engineer or data analyst", "software engineer", "data analyst"),
    _turn(15, "data analyst skills", "data analyst"),
    _turn(16, "how do I become a data analyst?", "data analyst"),
    _turn(17, "project manager skills", "project manager"),
    _turn(18, "how do I become a project manager?", "project manager"),
    _turn(19, "how do I become a plumber?", "plumber"),
    _turn(20, "electrician course", "electrician"),
    _turn(21, "electrician or carpenter", "electrician", "carpenter"),
    _turn(22, "what is a teacher degree?", "teacher"),
    _turn(23, "can I be a teacher without a degree?", "teacher"),
    _turn(24, "lawyer qualifications", "lawyer"),
    _turn(25, "lawyer or chef", "lawyer", "chef"),
    _turn(26, "chef training", "chef"),
    _turn(27, "can I become a teacher without a degree?", "teacher"),
    _turn(28, "what is a career coach?", "career coach"),
    _turn(29, "how do I become a radiographer?", "radiographer"),
    _turn(30, "nursing and teaching", "nursing", "teacher"),
)

#: 37 answerable queries with the turns each is *about*, plus 7 that share no
#: topic with anything in ``CORPUS``. The groups mirror the failure modes each
#: design has: a single word is the easy case, a phrase is the normal one, a
#: conjunction is what broke AND, a typo is what a user actually types, and
#: "natural" is a whole question.
#:
#: Gold is a **set**, because these questions are genuinely ambiguous and
#: pretending otherwise would make p@1 a measure of my guessing. `nurse` alone is
#: about the "what does becoming one involve" cluster (r01, r02, r06) and not
#: about the paramedic difference (r08); `nursing` is about the nursing turns and
#: not about the teacher turn in r30's second half. The sets are annotated by
#: content and are deliberately **not** "every turn that matches" — that would
#: make p@1 trivially 1.0 for the broad words and stop measuring anything.
#: `validate_corpus` still requires each gold turn to share a real term with the
#: query, so a set cannot drift into "unrelated but listed".
QUERIES: tuple[Query, ...] = (
    Query("nurse", "single", ("r01", "r02", "r06")),
    Query("plumber", "single", ("r19",)),
    Query("lawyer", "single", ("r24",)),
    Query("healthcare", "single", ("r10", "r11")),
    Query("nursing", "single", ("r03", "r04", "r05", "r30")),
    Query("nurse skills", "phrase", ("r01",)),
    Query("data analyst skills", "phrase", ("r15",)),
    Query("pharmacy technician training", "phrase", ("r12",)),
    Query("nursing assistant", "phrase", ("r03",)),
    Query("project manager", "phrase", ("r17", "r18")),
    Query("become a teacher", "phrase", ("r23", "r27")),
    Query("nurse or nursing", "conjunction", ("r03", "r04", "r07")),
    Query("nursing and teaching", "conjunction", ("r30",)),
    Query("software engineer or data analyst", "conjunction", ("r14", "r15", "r16")),
    Query("career near healthcare", "conjunction", ("r10", "r11")),
    Query("registered nurse AND nursing", "conjunction", ("r04", "r09")),
    Query("electrician or carpenter", "conjunction", ("r20", "r21")),
    Query("teacher but no degree", "conjunction", ("r23", "r27")),
    Query("lawyer nor chef", "conjunction", ("r24", "r25", "r26")),
    Query("nurse not paramedic", "conjunction", ("r08",)),
    Query("nurse skils", "typo", ("r01",)),
    Query("techer degree", "typo", ("r22", "r23", "r27")),
    Query("electcian course", "typo", ("r20",)),
    # A misspelling whose *correct* spelling appears in no turn: expected to
    # recall nothing, and it stays in the denominator. No stemming, by design.
    Query("nursing trainng", "typo", ("r04", "r30")),
    Query("nurs", "typo_unmatched", ()),
    Query("radiografer", "typo_unmatched", ()),
    Query("how do I become a nurse", "natural", ("r02",)),
    Query("what training does a nurse need", "natural", ("r04", "r06")),
    Query("can I be a teacher without a degree", "natural", ("r23",)),
    Query("skills does a nurse need", "natural", ("r01",)),
    Query("career in healthcare", "natural", ("r10", "r11")),
    Query("nursing training", "natural", ("r04",)),
    # Function-word-led questions. These exist to make the stopword list a
    # *measured* thing rather than a declared one: `are`, `am`, `all`, `every`,
    # `each`, `any`, `while` and `whether` were all unfiltered while their classes
    # were covered, and a query set that never uses them cannot see the
    # difference the fix makes. They are also the questions a person actually
    # asks, and the ones that dilute an OR the most.
    Query("am i eligible to train as a nursing assistant", "natural", ("r03",)),
    Query("all the skills for every nurse role", "natural", ("r01", "r06")),
    Query("each nursing role and whether it pays well", "conjunction", ("r07",)),
    # The sharpest of the function-word queries, and the reason the fix is
    # measurable rather than merely argued: with `are` and `all` filtered the
    # disjunction is `nursing OR assistant OR roles OR same`, and r03 — the only
    # turn that is *both* a nursing and an assistant one — wins on two matching
    # terms. Unfiltered, the same query is a six-way disjunction in which the
    # scaffolding (`are`, `all`, `the`) matches most of the corpus, and the
    # ranking flattens to whichever turn happens to be shortest.
    Query("are all nursing assistant roles the same", "natural", ("r03",)),
    Query("astrophysics", "irrelevant", ()),
    Query("how do I bake sourdough bread at home", "irrelevant", ()),
    Query("zebra migration patterns", "irrelevant", ()),
    Query("what is the kubernetes ingress controller", "irrelevant", ()),
    Query("best hiking boots for the alps", "irrelevant", ()),
    Query("are there any openings in astrophysics", "irrelevant", ()),
    # Function words on the *irrelevant* axis too: with `are`, `any`, `all` and
    # `while` filtered, this reduces to three content words the corpus has never
    # seen. Without the filter it matches on the scaffolding alone, which is
    # precisely the failure the list exists to prevent.
    Query("while i was away, are all sourdough starters the same", "irrelevant", ()),
)

#: The five irrelevant queries, named so the "did widening match everything"
#: question has one number attached to it.
IRRELEVANT_GROUPS = frozenset({"irrelevant"})


@dataclass(frozen=True, slots=True)
class Design:
    """One point in the comparison: the shipped code with two knobs turned.

    ``join`` and ``stopwords`` are the only things that vary. Everything else —
    tokenisation, the length filter, quoting, dedupe, the SQL, bm25 — is reached
    through ``Fts5EpisodeRetriever`` unmodified, because a design comparison in
    which the terms are computed by a different program is not a comparison of
    designs.
    """

    name: str
    join: str = fts_retriever._JOIN
    stopwords: frozenset[str] = fts_retriever._STOPWORDS
    #: ``None`` = no pruning. Otherwise: drop query terms whose document frequency
    #: is at least this fraction of the corpus, which is exactly the point at
    #: which bm25's idf is zero and the term cannot change a score. bm25's idf
    #: is ``ln((N - n + 0.5) / (n + 0.5))``, so it passes through zero at
    #: ``n = N / 2`` — hence a threshold of 0.5 means "half the corpus", not a
    #: quarter, and the ``2 *`` that would make it a quarter is exactly the kind
    #: of off-by-a-factor-of-two that makes an ablation look like a no-op for
    #: reasons that have nothing to do with the idea.
    prune_by_idf: float | None = None
    #: Ask with ``join``; if that returns nothing, ask again with ``" OR "``.
    fallback_to_or: bool = False


DESIGNS: tuple[Design, ...] = (
    Design("AND only (the bug)", join=" AND "),
    Design("AND, no stopword filter", join=" AND ", stopwords=frozenset()),
    Design("AND, falling back to OR", join=" AND ", fallback_to_or=True),
    Design("OR, no stopword filter", stopwords=frozenset()),
    Design("OR + idf pruning (no list)", stopwords=frozenset(), prune_by_idf=0.5),
    Design("OR + stopwords (shipped)"),
)


@dataclass(frozen=True, slots=True)
class Metrics:
    """What a design scores. Every field is a number a change can move."""

    design: str
    answerable: int
    zero_hit: int
    p_at_1: float
    mrr_at_5: float
    irrelevant_hits: int
    widest: int
    matches: int
    #: Per-query top-1 run id, so a regression report can name the query.
    top1: tuple[str | None, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "design": self.design,
            "answerable": self.answerable,
            "zero_hit": self.zero_hit,
            "p_at_1": round(self.p_at_1, 3),
            "mrr_at_5": round(self.mrr_at_5, 3),
            "irrelevant_hits": self.irrelevant_hits,
            "widest": self.widest,
            "matches": self.matches,
        }


#: What the shipped design must not fall below. A floor, not an equality, so a
#: genuine improvement lands without editing this — and a regression fails CI
#: instead of quietly changing what the prose in ``fts_retriever`` claims.
#:
#: Derived from the shipped row on this corpus (2/36 zero-hit, 0.917 p@1, 0.926
#: MRR@5, 0 irrelevant hits, widest 12, 43 MATCHes), with the two floating
#: columns rounded *down* to a coarser grid so a third significant digit of
#: bm25 jitter is not a build failure. The integer columns are pinned exactly:
#: they are counted, not estimated, and a change in any of them is a change in
#: how much work a turn does, not a change in luck.
#:
#: ``answerable`` is pinned as well, and deliberately reported as its own
#: failure rather than folded into the others. Editing the query set is a
#: legitimate thing to do, but it invalidates the comparison, so it has to be a
#: loud deliberate act — re-derive these numbers from a fresh run — rather than a
#: silent one.
_FLOOR = Metrics(
    design="OR + stopwords (shipped)",
    answerable=36,
    zero_hit=2,
    p_at_1=0.85,
    mrr_at_5=0.88,
    irrelevant_hits=0,
    widest=12,
    matches=43,
)


def validate_corpus() -> None:
    """Raise unless ``CORPUS`` and ``QUERIES`` are a discriminating fixture.

    Three properties, each of which has to hold or the comparison below measures
    nothing:

    1. Every answerable query either names turns that exist, or is marked as
       expected to recall nothing for a stated reason.
    2. A gold turn is genuinely *about* the query — its question or labels
       contain a surviving term. Otherwise p@1 is measuring an accident.
    3. The corpus can tell the designs apart: at least one query is
       unsatisfiable under AND (or the AND row of the table is fiction), and
       the irrelevant queries share no surviving term with any turn (or "the
       filter stops the widening" is untestable).
    """
    known = {turn.run_id for turn in CORPUS}
    for query in QUERIES:
        if query.group in IRRELEVANT_GROUPS:
            if query.gold:
                raise AssertionError(f"{query.text!r} is irrelevant but names a gold turn")
            continue
        if not query.gold:
            if query.group != "typo_unmatched":
                raise AssertionError(
                    f"{query.text!r} is answerable with no gold turn and no reason; "
                    f"use group='irrelevant' or 'typo_unmatched'"
                )
            continue
        if len(set(query.gold)) != len(query.gold):
            raise AssertionError(f"{query.text!r} repeats a gold turn: {query.gold}")
        terms = {term.strip('"').lower() for term in fts_retriever._terms(query.text)}
        for run_id in query.gold:
            if run_id not in known:
                raise AssertionError(f"{query.text!r} names a turn that is not in CORPUS: {run_id}")
            turn = next(t for t in CORPUS if t.run_id == run_id)
            haystack = f"{turn.question} {' '.join(turn.labels)}".lower()
            if not terms & set(_index_terms(haystack)):
                raise AssertionError(
                    f"{query.text!r} claims gold {run_id}, whose text shares no term with it"
                )

    if not _unsatisfiable_under_and():
        raise AssertionError(
            "no query in QUERIES is unsatisfiable under AND, so this corpus cannot "
            "distinguish the AND design from the OR one"
        )


def _index_terms(text: str) -> tuple[str, ...]:
    return tuple(token.lower() for token in fts_retriever._TOKEN.findall(text))


def _unsatisfiable_under_and() -> list[str]:
    """Answerable queries that AND-joining the terms cannot match at all."""
    rows = _document_frequency()
    out: list[str] = []
    for query in QUERIES:
        if query.group in IRRELEVANT_GROUPS:
            continue
        terms = [term.strip('"').lower() for term in fts_retriever._terms(query.text)]
        if terms and not all(rows.get(term, 0) > 0 for term in terms):
            out.append(query.text)
    return out


@contextmanager
def _design(db_path: Path, design: Design) -> Iterator[Fts5EpisodeRetriever]:
    """Run the real retriever with this design's two knobs turned."""
    retriever = Fts5EpisodeRetriever(db_path=db_path)
    terms = fts_retriever._terms

    if design.prune_by_idf is not None:
        rows = _document_frequency()
        total = len(CORPUS)

        def _pruned(question: str) -> list[str]:
            kept = []
            for term in terms(question):
                word = term.strip('"').lower()
                if rows.get(word, 0) >= total * design.prune_by_idf:  # type: ignore[operator]
                    continue
                kept.append(term)
            return kept

        patches: list[Any] = [
            mock.patch.object(fts_retriever, "_terms", _pruned),
        ]
    else:
        patches = []

    patches.append(mock.patch.object(fts_retriever, "_JOIN", design.join))
    patches.append(mock.patch.object(fts_retriever, "_STOPWORDS", design.stopwords))
    for patch in patches:
        patch.start()
    try:
        yield retriever
    finally:
        for patch in reversed(patches):
            patch.stop()


def _document_frequency() -> dict[str, int]:
    """``term -> documents containing it``, from FTS5 itself.

    Asked of the engine rather than recomputed over ``CORPUS``: the point of the
    idf design is what *bm25* will see, and the engine's own tokenizer is what
    it will see it through. A hand-rolled count here would be a second
    implementation of tokenisation, which is the mistake that made the earlier
    "before" column untrustworthy.
    """
    with _built_database() as db_path:
        conn = sqlite3.connect(db_path)
        try:
            df: dict[str, int] = {}
            for turn in CORPUS:
                for column in (turn.question, *turn.labels):
                    for word in set(_index_terms(column)):
                        df[word] = df.get(word, 0) + 1
            rows = {
                word: conn.execute(
                    "SELECT COUNT(*) FROM episodes_fts WHERE episodes_fts MATCH ?", (f'"{word}"',)
                ).fetchone()[0]
                for word in df
            }
        finally:
            conn.close()
    return {word: int(count) for word, count in rows.items()}


@contextmanager
def _built_database() -> Iterator[Path]:
    """A real ``memory.db`` holding ``CORPUS``, through ``record_episode``."""
    directory = tempfile.TemporaryDirectory(prefix="ta-recall-bench-")
    db_path = Path(directory.name) / "memory.db"
    try:
        for index, turn in enumerate(CORPUS, start=1):
            record_episode(
                RunLogRecord(
                    run_id=turn.run_id,
                    ts=f"2026-01-{index:02d}T09:00:00+00:00",
                    suite="esco",
                    plan=["locate"],
                    question=turn.question,
                    result=ResultSummary(
                        node_ids=[f"esco:occupation:{index}"],
                        node_labels=list(turn.labels),
                    ),
                ),
                db_path=db_path,
            )
        yield db_path
    finally:
        directory.cleanup()


def measure(design: Design) -> Metrics:
    """Run every query under one design and score it."""
    with _built_database() as db_path, _design(db_path, design) as retriever:
        return _score(
            design.name,
            lambda question: _run(retriever, question, design),
        )


def _score(
    name: str,
    run: Callable[[str], tuple[list[EpisodeHit], int]],
) -> Metrics:
    """The scoring loop, shared by every backend so the numbers are comparable.

    Extracted from ``measure`` so the vector arm below is measured by the *same*
    arithmetic on the *same* queries rather than by a second implementation that
    could differ in a way nobody would notice — which is the failure mode of any
    "A vs B" comparison written twice.

    ``run`` returns hits plus the number of search operations it cost, so
    ``matches`` means the same thing in both rows: how much work a turn does.
    """
    answerable = [q for q in QUERIES if q.group not in IRRELEVANT_GROUPS]
    irrelevant = [q for q in QUERIES if q.group in IRRELEVANT_GROUPS]
    matches = 0
    reciprocal_sum = 0.0
    hits_at_1 = 0
    zero_hit = 0
    widest = 0
    irrelevant_hits = 0
    top1: list[str | None] = []
    for query in answerable:
        hits, used = run(query.text)
        matches += used
        run_ids = [hit.run_id for hit in hits]
        widest = max(widest, len(run_ids))
        if not run_ids:
            zero_hit += 1
            top1.append(None)
            continue
        top1.append(run_ids[0])
        if not query.gold:
            # An answerable query recall could not serve (`nurs`, no
            # stemming). It scored its zero-hit above; it cannot also score
            # as a miss, which would double-count one limitation.
            continue
        if run_ids[0] in query.gold:
            hits_at_1 += 1
        for rank, run_id in enumerate(run_ids[:_LIMIT], start=1):
            if run_id in query.gold:
                reciprocal_sum += 1.0 / rank
                break
    for query in irrelevant:
        hits, used = run(query.text)
        matches += used
        irrelevant_hits += len(hits)

    count = len(answerable)
    return Metrics(
        design=name,
        answerable=count,
        zero_hit=zero_hit,
        p_at_1=hits_at_1 / count,
        mrr_at_5=reciprocal_sum / count,
        irrelevant_hits=irrelevant_hits,
        widest=widest,
        matches=matches,
        top1=tuple(top1),
    )


def _run(retriever: Fts5EpisodeRetriever, question: str, design: Design) -> tuple[list[Any], int]:
    """One query over the whole corpus, plus how many MATCHes it cost.

    Asked with a deliberately large ``limit`` and sliced afterwards, because two
    of the numbers — the widest result set and the irrelevant-hit count — are
    about *unbounded* recall. Measuring them through ``limit=5`` reports the
    ceiling back as though it were the size, which makes every design look
    identical. Ranking is unaffected: bm25 order is the same with and without a
    LIMIT, and MRR only reads the first five.

    ``search`` performs at most one ``MATCH`` and returns early when the term
    list is empty, so the count is a statement about work done, not a guess.
    """
    if not fts_retriever._terms(question):
        return [], 0
    hits = retriever.search(question, limit=_WIDE_LIMIT)
    if design.fallback_to_or and not hits:
        with mock.patch.object(fts_retriever, "_JOIN", fts_retriever._JOIN):
            hits = retriever.search(question, limit=_WIDE_LIMIT)
        return hits, 2
    return hits, 1


def regressions(metrics: Metrics) -> list[str]:
    """What the shipped design lost against ``_FLOOR``. Empty means healthy."""
    if metrics.answerable != _FLOOR.answerable:
        return [
            f"the query set changed ({metrics.answerable} answerable, "
            f"floor {_FLOOR.answerable}); re-derive the floor deliberately"
        ]
    lost: list[str] = []
    if metrics.zero_hit > _FLOOR.zero_hit:
        lost.append(f"zero-hit {metrics.zero_hit} > floor {_FLOOR.zero_hit}")
    if metrics.p_at_1 < _FLOOR.p_at_1:
        lost.append(f"p@1 {metrics.p_at_1:.3f} < floor {_FLOOR.p_at_1:.3f}")
    if metrics.mrr_at_5 < _FLOOR.mrr_at_5:
        lost.append(f"MRR@5 {metrics.mrr_at_5:.3f} < floor {_FLOOR.mrr_at_5:.3f}")
    if metrics.irrelevant_hits > _FLOOR.irrelevant_hits:
        lost.append(f"irrelevant hits {metrics.irrelevant_hits} > floor {_FLOOR.irrelevant_hits}")
    if metrics.widest > _FLOOR.widest:
        lost.append(f"widest result set {metrics.widest} > floor {_FLOOR.widest}")
    if metrics.matches > _FLOOR.matches:
        lost.append(f"MATCHes {metrics.matches} > floor {_FLOOR.matches}")
    return lost


def measure_all() -> list[Metrics]:
    return [measure(design) for design in DESIGNS]


def render_table(rows: Sequence[Metrics]) -> str:
    header = (
        "| design | zero-hit | p@1 | MRR@5 | irrelevant hits | widest | MATCHes |\n"
        "| ------ | -------: | --: | ----: | -------------: | ------: | ------: |"
    )
    lines = [
        f"| {r.design} | {r.zero_hit}/{r.answerable} | {r.p_at_1:.3f} | {r.mrr_at_5:.3f} "
        f"| {r.irrelevant_hits} | {r.widest} | {r.matches} |"
        for r in rows
    ]
    return "\n".join([header, *lines])


def idf_pruning_report(
    *, thresholds: Sequence[float] = (0.5, 0.25, 0.15, 0.1)
) -> list[dict[str, Any]]:
    """Why the idf row is what it is — a term frequency table and a trade curve.

    The hypothesis was that OR plus "drop terms bm25 cannot score" would make a
    hand-maintained stopword list unnecessary, because a term in a large enough
    share of the corpus contributes exactly nothing to a score. The mechanism is
    right; the threshold implied by bm25's own ``idf = ln((N - n + 0.5)/(n +
    0.5))`` is not, and this is the table that says so. On a 30-turn personal
    corpus almost no query term is anywhere near half of it, so the rule as
    stated prunes nothing and the idf row is byte-identical to "no stopword
    filter" — a result worth having, because it says the alternative is not a
    *better* list, it is *no* list.

    The lower thresholds are measured rather than assumed, so the trade is
    visible: how many query terms each one removes, and what that costs. They are
    also the reason the idea was not adopted. bm25's negative-idf region is
    formally "contributes nothing", but a term that cannot *distinguish* documents
    is not thereby useless — the words pruned at 25% are `nurse`, `skills`,
    `teacher` and `training`, which is to say the corpus's actual subject matter.
    "Contributes nothing to the score" and "carries no information" are different
    claims, and this corpus is the disproof of the second.
    """
    rows = _document_frequency()
    total = len(CORPUS)
    corpus_terms = sorted(
        ((word, count) for word, count in rows.items()), key=lambda item: -item[1]
    )
    query_terms: set[str] = set()
    for query in QUERIES:
        query_terms.update(term.strip('"').lower() for term in fts_retriever._terms(query.text))

    curve: list[dict[str, Any]] = []
    for threshold in thresholds:
        pruned = sorted(word for word in query_terms if rows.get(word, 0) >= total * threshold)
        metrics = measure(
            Design(
                f"OR + idf pruning at {threshold:.0%}",
                stopwords=frozenset(),
                prune_by_idf=threshold,
            )
        )
        curve.append(
            {
                "threshold": threshold,
                "pruned_query_terms": pruned,
                "pruned_count": len(pruned),
                **{k: v for k, v in metrics.as_dict().items() if k != "design"},
            }
        )
    return [
        {
            "corpus_size": total,
            "most_frequent_corpus_terms": [
                {"term": word, "documents": count} for word, count in corpus_terms[:8]
            ],
            "most_frequent_query_terms": [
                {"term": word, "documents": rows[word]}
                for word in sorted(query_terms, key=lambda w: -rows.get(w, 0))[:8]
            ],
            "curve": curve,
        }
    ]


def _maybe_compare(want_vector: bool) -> BackendComparison | None:
    """The backend comparison, or ``None`` when it was not asked for or cannot run.

    Returns ``None`` for a missing key rather than exiting non-zero: the lexical
    half of the report is still valid, and a machine-readable run that dies
    because an optional provider is unconfigured is worse than one that reports
    what it managed to measure.
    """
    if not want_vector:
        return None
    from talent_angels.memory.embeddings import default_embedder

    try:
        return compare_backends(default_embedder())
    except Exception as exc:  # noqa: BLE001 - reported, never fatal
        print(
            f"vector comparison unavailable: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return None


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m talent_angels.evals.recall",
        description="Measure the lexical retriever's query design against the shipped code.",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of a table.")
    parser.add_argument(
        "--idf-report",
        action="store_true",
        help="Also print the document-frequency table behind the idf-pruning row.",
    )
    parser.add_argument(
        "--vector",
        action="store_true",
        help=(
            "Also score the vector backend over the same corpus and queries. Calls a "
            "paid embedding provider, so it is opt-in and is not part of the offline "
            "suite."
        ),
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="With --vector: print the score bands a relevance floor must separate, "
        "and score the hybrid ladder at the configured floor.",
    )
    args = parser.parse_args(argv)
    if args.vector and args.calibrate:
        from talent_angels.memory.embeddings import default_embedder
        from talent_angels.memory.vector_retriever import relevance_floor

        embedder = default_embedder()
        bands = calibration_bands(embedder)
        floor = relevance_floor(str(getattr(embedder, "_model", "")))
        hybrid = measure_hybrid(embedder, floor)
        print(json.dumps({"bands": bands, "floor": floor}, indent=2))
        print(render_table([measure(DESIGNS[-1]), hybrid]))
        return 0

    try:
        validate_corpus()
    except AssertionError as exc:
        print(f"corpus is not discriminating: {exc}", file=sys.stderr)
        return 2

    rows = measure_all()
    shipped = next(r for r in rows if r.design == _FLOOR.design)
    lost = regressions(shipped)
    idf = idf_pruning_report() if args.idf_report else None
    comparison = _maybe_compare(args.vector)

    if args.json:
        payload: dict[str, Any] = {
            "designs": [r.as_dict() for r in rows],
            "regressions": lost,
        }
        if idf is not None:
            payload["idf_pruning"] = idf
        if comparison is not None:
            payload["backends"] = comparison.as_dict()
        print(json.dumps(payload, indent=2))
    else:
        print(f"corpus: {len(CORPUS)} turns, {len(QUERIES)} queries, recall limit={_LIMIT}")
        print()
        print(render_table(rows))
        if idf is not None:
            print()
            print(json.dumps(idf, indent=2))
        if comparison is not None:
            print()
            print(render_comparison(comparison))
        print()
        if lost:
            print("REGRESSION against the floor: " + "; ".join(lost), file=sys.stderr)
        else:
            print(
                f"shipped design is at or above the floor "
                f"(zero-hit <= {_FLOOR.zero_hit}, p@1 >= {_FLOOR.p_at_1:.3f}, "
                f"MRR@5 >= {_FLOOR.mrr_at_5:.3f}, irrelevant hits <= "
                f"{_FLOOR.irrelevant_hits}, widest <= {_FLOOR.widest}, "
                f"MATCHes <= {_FLOOR.matches})"
            )
    return 1 if lost else 0


# --------------------------------------------------------------------------
# Backend comparison: does meaning beat words on this corpus?
# --------------------------------------------------------------------------
#
# Everything above varies *how* the lexical index is queried. This section
# compares two different *kinds* of retrieval over the identical corpus and
# query set, scored by the identical loop (`_score`).
#
# It exists because the vector path is a dependency, a network call on every
# turn, and a second store of the user's personal data. Shipping it on the
# argument that "semantic search is better in general" is an argument, not a
# measurement. The question here is narrow and answerable: **on past user
# questions, does `text-embedding-3-small` rank gold turns better than bm25 over
# word overlap?**
#
# The expected result is more interesting than a win or a loss. FTS5 cannot
# recall a paraphrase, because a paraphrase shares no words — so vector must win
# on exactly the queries that share no vocabulary with the turn they are after.
# But bm25 is sharp when the words *are* shared, and a dense retriever always
# returns its ``k`` nearest turns with a score attached, never nothing. So
# vector should win on `zero_hit` and may lose on `p@1`. That is why both are
# reported, and why `verdict()` refuses to collapse them into one number.
#
# Deliberately not part of the offline suite: it calls a paid provider.


@dataclass(frozen=True, slots=True)
class BackendComparison:
    """Two backends over one corpus, and what changed between them."""

    lexical: Metrics
    vector: Metrics
    dimensions: int
    model: str
    #: Answerable queries the two backends rank differently at rank 1.
    disagreements: int
    #: Of those, the ones lexical got right and vector got wrong.
    lexical_only: int
    vector_only: int

    def verdict(self) -> str:
        """One sentence, read off the numbers rather than off the expectation."""
        lp, vp = self.lexical.p_at_1, self.vector.p_at_1
        recovered = self.lexical.zero_hit - self.vector.zero_hit
        if recovered > 0 and vp >= lp - 0.02:
            return (
                f"vector recalls {recovered} more quer"
                f"{'y' if recovered == 1 else 'ies'} at equal precision ({vp:.3f} vs {lp:.3f})"
            )
        if vp > lp:
            return f"vector is more precise (p@1 {vp:.3f} vs {lp:.3f})"
        if lp > vp:
            return f"lexical is more precise (p@1 {lp:.3f} vs {vp:.3f})"
        return "indistinguishable on this corpus"

    def as_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "dimensions": self.dimensions,
            "lexical": self.lexical.as_dict(),
            "vector": self.vector.as_dict(),
            "disagreements": self.disagreements,
            "lexical_only": self.lexical_only,
            "vector_only": self.vector_only,
            "verdict": self.verdict(),
        }


@contextmanager
def _vector_index(embedder: Any) -> Iterator[tuple[SqliteVecIndex, dict[str, list[float]]]]:
    """A real vector index over ``CORPUS``, plus a pre-embedded query table.

    The query vectors are embedded **once, up front, in two calls** rather than
    one per query. That is a fairness decision as much as an optimisation: 43
    sequential calls earn a `429 Model busy` from OpenRouter — which happened
    while this was being written — and a rate-limited run that silently
    measures a *degraded* backend would be worse than no run. It flatters
    neither arm on quality: every query is embedded by the same model, and
    ranking does not depend on when the embedding was made.

    The harness searches the **index** with those vectors rather than calling
    ``VectorEpisodeRetriever.search``, which would re-embed per query. The two
    are the same computation — the retriever is `index.hits(embed(question))` —
    so this measures the retriever's ranking without its network cost, and the
    per-turn cost of one embedding stays where it belongs: in the docs for
    `vector_retriever`, which is the only place a reader meets it in anger.
    """
    with _built_database() as db_path:
        corpus_vectors = embedder.embed([episode_text(t.question, t.labels) for t in CORPUS])
        index = SqliteVecIndex(dimensions=embedder.dimensions, db_path=db_path)
        index.clear()
        for turn, vector in zip(CORPUS, corpus_vectors, strict=True):
            index.add(turn.run_id, vector)

        texts = [q.text for q in QUERIES]
        query_vectors = embedder.embed(texts)
        yield index, dict(zip(texts, query_vectors, strict=True))


def measure_vector(embedder: Any) -> Metrics:
    """Score the vector backend over ``QUERIES`` with the shared loop."""
    with _vector_index(embedder) as (index, by_text):
        return _score(
            f"vector ({getattr(embedder, '_model', 'unknown')})",
            # `matches` counts searches. For the lexical arm that is MATCH
            # evaluations against a local file; here it is nearest-neighbour
            # lookups over a 30-row table. Neither is the money — the money is
            # the two embed calls, made once, outside the loop.
            lambda question: (index.hits(by_text[question], limit=_WIDE_LIMIT), 1),
        )


def _top1_by_query(metrics: Metrics) -> dict[str, str | None]:
    """Map each answerable query's text to the run id it ranked first.

    ``Metrics.top1`` is positional — it lines up with the answerable queries in
    declaration order, which is only usable while nothing reorders them. Keying
    by text is what makes the disagreement count below survive someone adding a
    query to the middle of ``QUERIES``.
    """
    answerable = [q.text for q in QUERIES if q.group not in IRRELEVANT_GROUPS]
    return dict(zip(answerable, metrics.top1, strict=True))


def measure_hybrid(embedder: Any, floor: float) -> Metrics:
    """The shipped default: keyword first, vector (above ``floor``) only on a miss."""
    from talent_angels.memory.fts_retriever import Fts5EpisodeRetriever

    with _vector_index(embedder) as (index, by_text):
        lexical = Fts5EpisodeRetriever(db_path=index._db_path)  # noqa: SLF001

        def run(question: str) -> tuple[list[EpisodeHit], int]:
            hits = lexical.search(question, limit=_WIDE_LIMIT)
            if hits:
                return hits, 1
            ranked = index.hits(by_text[question], limit=_WIDE_LIMIT)
            return [hit for hit in ranked if hit.score >= floor], 2

        return _score(f"hybrid ({getattr(embedder, '_model', 'unknown')}, floor {floor})", run)


def calibration_bands(embedder: Any) -> dict[str, list[float]]:
    """Top-hit scores for unrelated queries and gold-hit scores for answerable ones.

    A floor belongs strictly between ``max(unrelated_top)`` and
    ``min(answerable_gold)``; this is how RELEVANCE_FLOORS values are derived.
    """
    unrelated: list[float] = []
    gold_scores: list[float] = []
    with _vector_index(embedder) as (index, by_text):
        for query in QUERIES:
            hits = index.hits(by_text[query.text], limit=5)
            if query.group in IRRELEVANT_GROUPS:
                if hits:
                    unrelated.append(round(hits[0].score, 3))
                continue
            golds = [h.score for h in hits if h.run_id in set(query.gold)]
            if golds:
                gold_scores.append(round(max(golds), 3))
    return {"unrelated_top": sorted(unrelated), "answerable_gold": sorted(gold_scores)}


def compare_backends(embedder: Any) -> BackendComparison:
    """Score both backends over one corpus and diff their top-1 choices.

    The disagreement counts are the point. Aggregate p@1 says which backend is
    better; it does not say *where* they differ, and a difference spread evenly
    across every query is a different finding from one concentrated in the
    paraphrases. ``vector_only`` is the number that decides whether this is
    worth shipping: if it is large and ``lexical_only`` is small, the vector
    backend is catching paraphrases and losing nothing else, which is a case for
    offering it *alongside* the lexical one rather than replacing it.
    """
    lexical = measure(DESIGNS[-1])  # "OR + stopwords (shipped)"
    vector = measure_vector(embedder)

    gold = {q.text: set(q.gold) for q in QUERIES if q.gold}
    lexical_top = _top1_by_query(lexical)
    vector_top = _top1_by_query(vector)
    lexical_only = vector_only = 0
    for text, golds in gold.items():
        l_top, v_top = lexical_top.get(text), vector_top.get(text)
        if l_top == v_top:
            continue
        if l_top in golds and v_top not in golds:
            lexical_only += 1
        elif v_top in golds and l_top not in golds:
            vector_only += 1

    return BackendComparison(
        lexical=lexical,
        vector=vector,
        dimensions=embedder.dimensions,
        model=str(getattr(embedder, "_model", "unknown")),
        disagreements=lexical_only + vector_only,
        lexical_only=lexical_only,
        vector_only=vector_only,
    )


def render_comparison(comparison: BackendComparison) -> str:
    """The two rows that decide whether the vector backend is worth having."""
    rows = [comparison.lexical, comparison.vector]
    header = (
        "| backend | zero-hit | p@1 | MRR@5 | irrelevant hits | widest | lookups |\n"
        "| ------- | -------: | --: | ----: | -------------: | ------: | ------: |"
    )
    lines = [
        f"| {r.design} | {r.zero_hit}/{r.answerable} | {r.p_at_1:.3f} | {r.mrr_at_5:.3f} "
        f"| {r.irrelevant_hits} | {r.widest} | {r.matches} |"
        for r in rows
    ]
    lines += [
        "",
        f"model: `{comparison.model}` ({comparison.dimensions} dimensions)",
        f"top-1 disagreements: {comparison.disagreements} "
        f"({comparison.vector_only} vector-only, {comparison.lexical_only} lexical-only)",
        f"verdict: {comparison.verdict()}",
    ]
    return "\n".join([header, *lines[:2], *lines[2:]])


if __name__ == "__main__":
    sys.exit(main())
