"""Lexical episode recall over SQLite FTS5.

FTS5 is compiled into the bundled SQLite, so this is a real implementation with
no new dependency — chosen deliberately as the baseline the vector path has to
beat, and as the offline fallback when `sqlite-vec` is absent.

Three kinds of test live here, in that order of importance:

- **The retriever itself.** Does it find the right turn, rank it, and stay
  inside the prompt budget?
- **The ways it can fail.** A malformed query, a missing index, a locked or
  corrupt database, a SQLite with no FTS5 at all. Recall is an enhancement; the
  contract is that none of these can become a failed turn. Each one is a test
  that would otherwise only be discovered in production.
- **That it recalls at all.** The `_terms` / `_JOIN` group, which is not a
  detail: a retriever that matches nothing raises nothing, logs nothing, and
  passes every test above. The measurements behind those decisions are in
  `fts_retriever._JOIN`.
- **The wiring.** The four system-prompt sites, and the switch that turns them
  on. A retriever nobody calls is a library, not a feature, and the wiring is
  exactly the part that a "it works" report hides. Four, not three: the agent
  loop, the two phrasing calls in `session.phrase`, and `assistant.synthesize`.
  The fourth was missed when recall first landed — the function built a prompt
  from the profile and notes blocks and nothing else, and no test noticed,
  because nothing asserted that recall appeared in the prompt it builds. The
  count is spelled out here for the same reason: a list of sites is only worth
  keeping if something checks it against the source.
"""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Sequence
from pathlib import Path

import pytest

from talent_angels.memory.episodes import clear_episodes, record_episode, sync_fts
from talent_angels.memory.fts_retriever import _BUSY_TIMEOUT_S, _JOIN, Fts5EpisodeRetriever, _terms
from talent_angels.memory.retrieval import (
    _RECALL_LINE_CHARS,
    RECALL_LIMIT,
    Retriever,
    recall_prefix,
)
from talent_angels.runlog.models import ResultSummary, RunLogRecord

#: This module's logger, so a caplog assertion cannot pass on someone else's
#: warning — `retrieval` logs under the same message shape.
LOGGER = "talent_angels.memory.fts_retriever"

_TS = "2026-01-01T00:00:00+00:00"


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "memory.db"


def _record(
    db_path: Path,
    run_id: str,
    question: str,
    labels: tuple[str, ...],
    *,
    ts: str = _TS,
) -> None:
    record_episode(
        RunLogRecord(
            run_id=run_id,
            ts=ts,
            suite="esco",
            plan=["locate"],
            question=question,
            result=ResultSummary(node_ids=["esco:occupation:1"], node_labels=list(labels)),
        ),
        db_path=db_path,
    )


def _warnings(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [record for record in caplog.records if record.name == LOGGER]


def _bullets(out: str) -> list[str]:
    return [line for line in out.splitlines() if line.strip()][1:]


def _search(db_path: Path, question: str, **kwargs: int):
    return Fts5EpisodeRetriever(db_path=db_path).search(question, **kwargs)


def _fts_data_rows(db_path: Path) -> int:
    """Rows in the FTS5 inverted index — the one count `DELETE` does not clear.

    A contentless-looking count of "is anything still indexed" would be answered
    by `SELECT COUNT(*) FROM episodes_fts`, which a `DELETE` does drive to zero.
    This is the shadow table underneath, and a delete marker leaves its rows in
    place, so it is the number that separates "unsearchable" from "gone".
    """
    conn = sqlite3.connect(db_path)
    try:
        return int(conn.execute("SELECT COUNT(*) FROM episodes_fts_data").fetchone()[0])
    finally:
        conn.close()


#: What FTS5 itself writes for an *empty* table: its structure record and its
#: averages record. Measured, not assumed — a non-zero "baseline" here is why the
#: assertion above is `== _FTS_DATA_BASELINE` and not `== 0`.
_FTS_DATA_BASELINE = 2

#: A question that looks like the one in `tests/test_erase_scopes.py`, so the
#: leak is measured on text a real user would type.
PII_QUESTION = "I am Priya Raman, based in Bangalore, email priya.raman@example.com"

#: The tokenised forms, which are what an FTS5 leak actually leaves behind.
PII_TOKENS = (b"priya", b"raman", b"bangalore", b"example")


# --- the retriever ---------------------------------------------------------


@pytest.fixture(scope="module")
def _fts5(tmp_path_factory: pytest.TempPathFactory) -> None:
    """Skip the whole file when this Python's sqlite3 has no FTS5.

    Module-scoped and autouse, so a build without FTS5 reports *these tests did
    not run* rather than a wall of errors. That distinction matters: a red suite
    that nobody can fix is a red suite people learn to ignore, and ignoring it
    costs the coverage in every other test file too.

    A `skip` and not a silent pass, with the reason naming the feature, so the
    run log still records that recall is unverified on this machine.

    The probe is a real ``CREATE VIRTUAL TABLE`` on a real connection rather than
    an import or a version check, because FTS5 is a loadable extension: a Python
    that reports the right SQLite version can still fail to load it.
    """
    probe = tmp_path_factory.mktemp("fts5") / "probe.db"
    conn = sqlite3.connect(probe)
    try:
        try:
            conn.execute("CREATE VIRTUAL TABLE probe USING fts5(x)")
        except sqlite3.OperationalError as exc:
            pytest.skip(f"this Python's sqlite3 has no FTS5, so recall is untested: {exc}")
    finally:
        conn.close()


@pytest.fixture(autouse=True)
def _require_fts5(_fts5: None) -> None:
    """Every test in this file needs FTS5; ``_fts5`` skips them all if it is absent."""


def test_fts5_is_available(db_path: Path, _require_fts5: None) -> None:
    """The probe itself is a test, so a skipped file still shows why.

    A file that skips silently is indistinguishable from a file that was never
    run; keeping the probe as a real assertion means the run log has a named,
    passing (or explicitly skipped) test behind every other result in here.
    """
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("CREATE VIRTUAL TABLE probe USING fts5(x)")
    finally:
        conn.close()


def test_the_retriever_satisfies_the_protocol() -> None:
    """`isinstance`, because Task 1 made the protocol runtime-checkable for this.

    An annotation is checked by nobody at runtime; the seam's whole promise is
    that a caller can hold a `Retriever` without knowing which backend answered.
    """
    assert isinstance(Fts5EpisodeRetriever(), Retriever)


def test_finds_a_past_question_by_shared_words(db_path: Path) -> None:
    _record(db_path, "r1", "what skills does a nurse need?", ("nurse",))
    _record(db_path, "r2", "how do I become a plumber?", ("plumber",))

    hits = _search(db_path, "nurse skills")

    assert [h.run_id for h in hits] == ["r1"]
    assert hits[0].node_labels == ("nurse",)


def test_returns_nothing_for_an_unrelated_question(db_path: Path) -> None:
    _record(db_path, "r1", "what skills does a nurse need?", ("nurse",))
    assert _search(db_path, "astrophysics") == []


def test_respects_the_limit(db_path: Path) -> None:
    for i in range(10):
        _record(db_path, f"r{i}", f"nurse question number {i}", ("nurse",))

    hits = _search(db_path, "nurse", limit=3)

    assert len(hits) == 3


def test_the_default_limit_is_the_seams_limit(db_path: Path) -> None:
    """Ten matching turns, no `limit=` argument, three back.

    The cap under test is the *default*, so a retriever hardcoded to a limit of
    ten — or to none at all — fails here and nowhere else. It is the same number
    `recall_prefix` re-slices to, which is what makes an over-long hit list
    unreachable rather than merely unlikely.
    """
    for i in range(10):
        _record(db_path, f"r{i}", f"nurse question number {i}", ("nurse",))

    assert len(_search(db_path, "nurse")) == RECALL_LIMIT


def test_a_limit_of_zero_or_below_recalls_nothing(db_path: Path) -> None:
    """`limit` is a ceiling, so zero or below means nothing — not "one".

    The seam re-slices to `max(limit, 0)`, so the prompt is safe either way; that
    safety net is exactly what would hide this. `LIMIT 0` in SQL returns nothing
    and `LIMIT -1` returns *everything* (it reads as "no limit"), so a floor of
    `max(1, limit)` quietly answers a request for none with one hit. Anyone
    reading a benchmark against this retriever counts that hit.
    """
    _record(db_path, "r1", "what skills does a nurse need?", ("nurse",))

    assert _search(db_path, "nurse", limit=0) == []
    assert _search(db_path, "nurse", limit=-5) == []


def test_results_are_capped_even_if_a_broad_query_matches_everything(db_path: Path) -> None:
    for i in range(50):
        _record(db_path, f"r{i}", "skills", ("nurse",))
    assert len(_search(db_path, "skills", limit=3)) == 3


def test_a_capitalised_question_still_matches(db_path: Path) -> None:
    """People type `NURSE SKILLS?` and get nothing back otherwise.

    FTS5's default tokenizer folds case, so this is free — and free behaviour is
    exactly what a refactor drops silently. If recall ever moves behind
    `sqlite-vec`, the guarantee has to be re-earned here rather than assumed.
    """
    _record(db_path, "r1", "what skills does a nurse need?", ("nurse",))

    assert [h.run_id for h in _search(db_path, "NURSE")] == ["r1"]


def test_an_accented_question_still_matches(db_path: Path) -> None:
    """The tokenizer must keep letters, not just `[A-Za-z]`.

    `_TOKEN` is a character class, and the obvious way to write it is
    ``[A-Za-z0-9]+`` — which silently reduces `enfermería` to the fragments
    `enfermer` and `a`, so a question in any language with diacritics matches
    the wrong rows or none. FTS5's own tokenizer strips diacritics on both sides
    of the comparison, so passing the word through whole is all it takes.

    The recorded labels are deliberately *unaccented*: ``json.dumps`` escapes
    non-ASCII, so the label column holds ``enfermer\\u00eda`` and contains the
    fragment `enfermer` as its own token. An accented label would let the
    truncated query match through the wrong column and hide the defect this is
    here to catch.
    """
    _record(db_path, "r1", "qué es enfermería en Barcelona", ("salud",))

    assert [h.run_id for h in _search(db_path, "enfermería")] == ["r1"]
    assert [h.run_id for h in _search(db_path, "enfermeria")] == ["r1"]


def test_single_letter_terms_are_dropped_on_purpose() -> None:
    """`_MIN_TERM_CHARS` is a precision decision, pinned as one.

    Every one-character token in a question is an article, a preposition or a
    stray `a`, and bm25 rewards a term that appears in few documents — so
    indexing them makes the ranking worse, not the coverage better. The number
    is asserted here, once, so tuning recall quality is a one-line edit in a
    test that says what is being traded rather than a hunt through tests that
    import the constant and follow it.
    """
    from talent_angels.memory.fts_retriever import _MIN_TERM_CHARS

    assert _MIN_TERM_CHARS == 2


def test_hits_are_ordered_best_first_and_score_higher_is_better(db_path: Path) -> None:
    """Both halves in one test, because a wrong negation satisfies neither.

    `bm25()` returns "smaller is better", and the seam's contract says "higher is
    better", so the retriever negates. Drop the negation and the order is still
    right — the SQL does that — so the *score* is the only thing that notices,
    and a score no current caller reads is a score nothing tests.
    """
    _record(db_path, "weak", "nurse", ("nurse",))
    _record(db_path, "strong", "nurse nurse nurse nurse", ("nurse",))

    hits = _search(db_path, "nurse")

    scores = [hit.score for hit in hits]
    assert scores == sorted(scores, reverse=True), (
        f"score direction is not higher-is-better: {scores}"
    )
    assert hits[0].run_id == "strong", f"the best match is not first: {hits}"
    assert max(scores) > 0.0, "a negated bm25 rank should be positive"


def test_a_hit_carries_the_turn_it_came_from(db_path: Path) -> None:
    """Every field of the hit, from the real row — including the timestamp.

    `EpisodeHit` promises a `ts`, and `episodes` has always had one. A retriever
    that hardcodes `ts=""` satisfies the type and starves the field, which is how
    a field in a typed contract becomes a field nobody trusts. Nothing renders it
    today, which is why it has to be pinned here rather than left to a caller.
    """
    _record(
        db_path, "r1", "what skills does a nurse need?", ("nurse",), ts="2026-02-03T04:05:06+00:00"
    )

    (hit,) = _search(db_path, "nurse")

    assert hit.ts == "2026-02-03T04:05:06+00:00"
    assert hit.question == "what skills does a nurse need?"
    assert hit.node_labels == ("nurse",)


def test_the_index_and_the_table_stay_in_step_on_a_rewrite(db_path: Path) -> None:
    """Re-recording a run_id replaces it in the table — and in the index.

    `record_episode` documents itself as idempotent per run_id, and the table
    honours that with a primary key. The index is a separate table with no such
    key, so it needs its own replacement, or a retried turn leaves the earlier
    wording searchable: the user asked about a nurse, the retry recorded a
    plumber, and recall still answers the nurse.

    Asserted as a row count as well as a search, because the visible symptom
    depends on the two questions sharing no words — a corpus where they happen
    to overlap would hide a duplicate behind a plausible-looking result.
    """
    _record(db_path, "r1", "what skills does a nurse need?", ("nurse",))
    _record(db_path, "r1", "how do I become a plumber?", ("plumber",))

    assert _search(db_path, "nurse") == [], "the replaced wording is still searchable"
    assert [h.run_id for h in _search(db_path, "plumber")] == ["r1"]
    conn = sqlite3.connect(db_path)
    try:
        assert conn.execute("SELECT COUNT(*) FROM episodes_fts").fetchone()[0] == 1
    finally:
        conn.close()


@pytest.mark.parametrize(
    "question",
    [
        "",
        "   ",
        '"""',
        "AND",
        "OR",
        "NOT",
        "-",
        "a*",
        ":",
        "^",
        "{a}",
        "*",
        "(((()",
    ],
    ids=lambda q: f"q{len(q)}",
)
def test_a_question_that_is_pure_fts5_syntax_never_raises(
    db_path: Path, question: str, caplog: pytest.LogCaptureFixture
) -> None:
    """Every one of these is a syntax error handed to `MATCH` unquoted.

    FTS5's query language is not a regular language: `-` and `:` are operators,
    `AND`/`OR`/`NOT`/`NEAR` are keywords, and an unbalanced quote runs off the
    end of the expression. Each of them raises `OperationalError`, and a raise
    here is a failed turn. Sanitising the query is the fix; this is the test that
    says the sanitiser is doing it rather than the `except` clause hiding it —
    every case must come back `[]` **and** log nothing. Silence is the proof the
    query never reached SQLite, and it matters: a warning per turn is how the
    log line that matters stops being read.

    Every question here is *purely* syntax, with no word left that could match,
    so `[]` is the only correct answer. The list deliberately stops at that
    boundary — the cases that put an operator next to a real word are in the next
    test, because for those `[]` is the bug this module had: an empty result was
    asserted for inputs that ought to recall the turn they name.
    """
    _record(db_path, "r1", "what skills does a nurse need?", ("nurse",))

    with caplog.at_level(logging.WARNING, logger=LOGGER):
        assert _search(db_path, question) == [], f"{question!r} recalled something"

    assert not _warnings(caplog), f"{question!r} reached SQLite and failed there"


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        # An operator glued to a word is still an operator, and still inert: the
        # word is quoted into a literal and the punctuation is dropped by
        # `_TOKEN`. The word is what decides the result.
        ("nurse AND", ["r1"]),
        ("nurse-plumber", ["r1"]),
        ("nurse:question", ["r1"]),
        ("nurse AND nursing", ["r1"]),
        # A keyword that survives the stopword filter is a word, not syntax —
        # and there is no document containing it, so it matches nothing.
        ("NEAR(", []),
        ("(unbalanced", []),
    ],
    ids=lambda v: f"q{len(v)}",
)
def test_an_operator_next_to_a_real_word_is_inert_and_the_word_still_recalls(
    db_path: Path, question: str, expected: list[str], caplog: pytest.LogCaptureFixture
) -> None:
    """The operator is a word and the word is a word; neither is syntax.

    This is the half of injection-proofing the previous test could not express.
    `nurse-plumber` contains two real words, so asserting `[]` for it asserted
    that a query matching half the corpus returns nothing — which is what AND
    between the terms did, and which hid the operator next to it. Quoting is
    what makes `-`, `:` and `AND` inert, and the expected results above are the
    proof: the words are found and nothing is logged.
    """
    _record(db_path, "r1", "what skills does a nurse need?", ("nurse",))

    with caplog.at_level(logging.WARNING, logger=LOGGER):
        assert [h.run_id for h in _search(db_path, question)] == expected

    assert not _warnings(caplog), f"{question!r} reached SQLite and failed there"


def test_fts_syntax_in_a_question_is_matched_as_words_not_as_operators(db_path: Path) -> None:
    """The other half of injection-proofing: the words must still be *findable*.

    Quoting each term is what makes `NEAR` a word to match instead of an
    operator to execute. A test that only asserts "no exception" would pass
    against a retriever that sanitised by deleting every suspicious character,
    which is quiet and wrong: `search("NEAR")` has to find a turn that says
    "NEAR".

    `NEAR` and not `AND`, and that substitution is the point: the stopword filter
    drops `and`, `or` and `not`, so those three are no longer available to
    demonstrate that a keyword survives as a literal. `NEAR` is an FTS5 keyword
    that no question scaffolding contains, so it is the one keyword that reaches
    the MATCH expression — and reaching it as `"NEAR"` is what this pins. See
    `test_the_five_conjunctions_are_filtered` for the other side of that trade.
    """
    _record(db_path, "r1", "what does a nurse need NEAR what pays well?", ("nurse",))

    assert [h.run_id for h in _search(db_path, "NEAR")] == ["r1"]


def test_a_question_of_only_stopwords_is_not_an_error(db_path: Path) -> None:
    """Stopword-only input never reaches SQLite, so there is nothing to rank.

    Two things are being kept apart here, and the second used to be doing the
    work. `search` returning `[]` is the contract: recall never raises. But `[]`
    on its own does not show *why*, and before the stopword filter it had a
    different reason — the words were queried, ANDed, and no document contained
    all of them. That made this test pass identically against the buggy
    retriever, which is exactly how a filter can be deleted and the suite stay
    green. So the term extraction is asserted directly, and the search result
    beside it.
    """
    _record(db_path, "r1", "what skills does a nurse need?", ("nurse",))

    assert _terms("the and of to a") == [], "a stopword survived the filter"
    assert _search(db_path, "the and of to a") == []


def test_the_five_conjunctions_are_filtered() -> None:
    """`or`, `and`, `not`, `but`, `nor` — the words that made recall silent.

    Pinned as a set and separately from the length filter, because they are the
    case where the two disagree: every one of them is two characters or more
    and therefore passed `_MIN_TERM_CHARS`, and every one of them is almost
    never present in a stored question. Joined with AND they made the query
    unsatisfiable, so `nurse or nursing` recalled nothing and reported no error.
    A retriever that keeps any of them has the bug back.
    """
    for word in ("or", "and", "not", "but", "nor"):
        assert _terms(word) == [], f"{word!r} is not being filtered"
        assert _terms(f"nurse {word.upper()} nursing") == ['"nurse"', '"nursing"']


def test_every_function_word_in_its_class_is_filtered() -> None:
    """The list has to close each class, not sample it.

    The gap this was written for: `is`, `was`, `were`, `be`, `been` and `being`
    were filtered and `are` and `am` were not. Same verb, same argument, and the
    consequence is not that recall is slightly worse — it is that
    `_terms("are there any openings")` came back as
    `['"are"', '"any"', '"openings"']`, so the disjunction matched two words that
    appear in most of the corpus and the single word naming the topic was diluted
    to a third of the score. Measured over the query set in ``evals.recall``, the
    two words that were missing were worth 0.084 of p@1.

    Enumerating the five conjunctions, as the test above does, cannot catch that:
    a list can be perfect on every word anyone thought to write down and still
    miss `are`. So each closed class is asserted *as a class* here, and the
    second half of the test asserts the inverse — the neighbours of those classes
    which must survive, because a filter that has learned "closed class" instead
    of "this specific list" will start eating `no` and `without`, and the failure
    will be silent under-recall rather than a loud error.
    """
    for word in (
        # determiners
        "a",
        "an",
        "the",
        "all",
        "any",
        "each",
        "every",
        "such",
        # the verb `be`, in every form that reaches a question
        "am",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        # prepositions
        "at",
        "as",
        "by",
        "for",
        "from",
        "in",
        "into",
        "of",
        "on",
        "than",
        "then",
        "there",
        "to",
        "with",
        # subordinating conjunctions
        "although",
        "because",
        "unless",
        "while",
        "since",
        "whether",
        # coordinating conjunctions
        "and",
        "but",
        "nor",
        "not",
        "or",
        "yet",
        "so",
    ):
        assert _terms(word) == [], f"{word!r} is not being filtered"
        assert _terms(f"nurse {word.upper()} nursing") == ['"nurse"', '"nursing"'], (
            f"{word!r} survived in a question around a content word"
        )


def test_the_words_beside_the_filtered_classes_still_survive() -> None:
    """The inverse of the test above, and the one that fails quietly.

    Everything here sits immediately outside a filtered class: a negating
    determiner, a degree negation, a modal, a comparative, a content verb. A
    filter that has generalised to "the whole class" will take these too, and
    the visible effect is that a real question recalls *fewer* correct turns —
    no error, no exception, just a worse answer. The OR makes it worse: dropping
    a term from a disjunction is what creates the silent widening the list exists
    to prevent, so a too-aggressive list and no list at all fail in the same
    direction, for the same reason.
    """
    assert _terms("are there any nursing openings") == ['"nursing"', '"openings"']
    assert _terms("all of the skills for every nurse role") == [
        '"skills"',
        '"nurse"',
        '"role"',
    ]
    assert _terms("each nursing role and whether it pays well") == [
        '"nursing"',
        '"role"',
        '"pays"',
        '"well"',
    ]
    # Negations that are not the coordinating conjunctions.
    assert _terms("can I be a teacher without a degree") == [
        '"teacher"',
        '"without"',
        '"degree"',
    ]
    assert _terms("a nurse with no degree") == ['"nurse"', '"no"', '"degree"']
    # Modals and auxiliaries that are not `be`.
    assert _terms("can I become a nurse") == ['"become"', '"nurse"']
    assert _terms("do I need a licence to work as a nurse") == [
        '"need"',
        '"licence"',
        '"work"',
        '"nurse"',
    ]


def test_a_stopword_only_question_recalls_nothing_at_all(db_path: Path) -> None:
    """Both halves together: a question with no content word has no answer.

    Once the classes are complete, these reduce to nothing — so the retriever
    returns no hits, which is correct. This is the end state argument 1 on
    `_STOPWORDS` is for, and it is a whole-question assertion rather than a
    per-word one so that the word-level tests above cannot collectively pass
    while the real behaviour is still broken.
    """
    _record(db_path, "r1", "what skills does a nurse need?", ("nurse",))
    for question in (
        "the and of to a",
        "are there any",
        "is there any at all",
        "while there was, as of in on by",
    ):
        assert _terms(question) == [], f"{question!r} kept a term"
        assert _search(db_path, question) == [], question


def test_question_scaffolding_is_filtered_but_content_words_are_not() -> None:
    """The stopword list is a list of function words, not a list of short words.

    Two failure modes to keep apart. Filtering too little leaves the unsatisfiable
    query; filtering too much throws away the word that identifies the topic, and
    the test that catches *that* is a miss, not an error — the most expensive kind
    to notice. So the second half below is the load-bearing half: `nurse` has to
    survive a filter that removed six other words of the same question.

    `become` surviving is deliberate and is the interesting one. It is a verb,
    not a function word, and it is what a stopword list written by feel tends to
    swallow: it appears in most of the corpus, so bm25 gives it a low idf, so
    leaving it in costs almost nothing — and on the queries where it is the only
    other word (`how do I become a nurse`) removing it would leave the query as a
    bare single term.
    """
    assert _terms("how do I become a nurse") == ['"become"', '"nurse"']
    # Words the list could plausibly have swallowed by accident. `no`,
    # `without` and `degree` discriminate between turns — dropping any of them
    # costs real recall, and `teacher but no degree` returns nothing without
    # `no`.
    assert _terms("can I become a teacher without a degree") == [
        '"become"',
        '"teacher"',
        '"without"',
        '"degree"',
    ]
    assert _terms("teacher but no degree") == ['"teacher"', '"no"', '"degree"']
    # A preposition, which is the class most likely to be trimmed by someone
    # tightening this list: it reads as harmless because it is short, and
    # "career in healthcare" still recalls the right turns without it — it just
    # also recalls every document containing the letter-pair "in".
    assert _terms("career in healthcare") == ['"career"', '"healthcare"']
    # Accented words are not English function words, so nothing removes them — and
    # the repeated one is deduplicated rather than matched twice.
    assert _terms("enfermería and enfermería") == ['"enfermería"']
    # The filter is case-folded in both directions: FTS5 folds case on both
    # sides of the comparison, so a user who types `AND` means the same word.
    assert _terms("Nurse OR Nursing") == ['"Nurse"', '"Nursing"']
    assert _terms("Is A Nurse") == ['"Nurse"']


def test_joining_the_terms_with_and_makes_recall_silent_again(db_path: Path) -> None:
    """The regression test for the reported bug, stated as the inverse.

    A test that only asserts "nurse or nursing now recalls" says the fix happened.
    This says *why* it had to, by running the conjunction the retriever used to
    build against the same database: `AND` between the terms of a disjunction is
    unsatisfiable whenever no document contains them all, and it returns no rows
    with no error and no warning. Asserted on the operator rather than on a
    result list, so it keeps its meaning if the stopword list or the corpus is
    ever retuned.
    """
    _record(db_path, "r1", "what skills does a nurse need?", ("nurse",))
    _record(db_path, "r2", "how long does nursing training take?", ("nursing",))
    retriever = Fts5EpisodeRetriever(db_path=db_path)

    def _rows(match: str) -> list[str]:
        conn = sqlite3.connect(db_path)
        try:
            return [
                row[0]
                for row in conn.execute(
                    "SELECT run_id FROM episodes_fts WHERE episodes_fts MATCH ?", (match,)
                )
            ]
        finally:
            conn.close()

    for question in ("nurse or nursing", "nurse OR nursing", "nursing and teaching"):
        terms = _terms(question)
        assert _JOIN.join(terms), f"{question!r} produced no expression"
        assert retriever.search(question, limit=3), f"{question!r} recalled nothing"
        conjunction = " AND ".join(terms)
        assert _rows(conjunction) == [], (
            f"{question!r} is satisfiable under AND, so this corpus cannot "
            f"demonstrate the defect: {conjunction}"
        )


def test_a_multi_word_question_recalls_rather_than_demanding_every_word(
    db_path: Path,
) -> None:
    """A question is not a document, and the words of one rarely all co-occur.

    `can I become a nurse` has a content word — `nurse` — and no stored question
    contains the phrase, so requiring every surviving term would match nothing
    and the assistant would have no idea the user had asked this before. This is
    the shape of almost every question a career assistant receives, which is why
    the join is a disjunction.
    """
    _record(db_path, "r1", "what skills does a nurse need?", ("nurse",))

    assert [h.run_id for h in _search(db_path, "can I become a nurse")] == ["r1"]
    assert [h.run_id for h in _search(db_path, "how do I train as a nurse")] == ["r1"]
    assert [h.run_id for h in _search(db_path, "I am thinking of nursing")] == []


def test_the_turn_matching_most_of_the_question_ranks_first(db_path: Path) -> None:
    """The disjunction widens the candidate set; the ranking has to recover it.

    This is the load-bearing half of choosing OR, and the reason precision
    survives it: `bm25()` sums over the terms a document *did* match, so the turn
    that answers more of the question scores higher and comes first. Without
    this, OR is just "return more, ranked by whatever".

    The `run_id`s are chosen so that insertion order is *not* the answer: `c-both`
    is recorded last and has to come first, and `a-nurse-only` is recorded first
    and has to come third. A retriever that ordered by rowid, or by `run_id`, or
    by anything but the score would pass this test with ids that happened to
    agree with the ranking — and `ORDER BY` is exactly the kind of clause that
    gets edited for performance.
    """
    _record(db_path, "a-nurse-only", "what skills does a nurse need?", ("nurse",))
    _record(db_path, "b-nursing-only", "how long does nursing training take?", ("nursing",))
    _record(db_path, "c-both", "can a nurse move into nursing management?", ("nurse",))

    hits = _search(db_path, "nurse or nursing", limit=3)

    assert [h.run_id for h in hits] == ["c-both", "a-nurse-only", "b-nursing-only"], (
        f"the turn matching both terms is not first: {[h.run_id for h in hits]}"
    )
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True), f"score order disagrees: {scores}"
    assert scores[0] > scores[1], f"the best match is not scored highest: {scores}"


def test_a_one_character_token_is_dropped_even_when_it_is_not_a_stopword(
    db_path: Path,
) -> None:
    """`_MIN_TERM_CHARS` does a job the stopword list cannot.

    The length filter used to be the only filter, and it caught the articles and
    prepositions by accident of their length. Now `_STOPWORDS` catches those by
    name, which leaves the length filter looking redundant — and a redundant-looking
    filter is one a reader deletes. So the case it still owns is pinned here: a
    one-character token that is a real word, or at least a real token, and is in
    no stopword list. "vitamin c" is not a contrived example in a career assistant
    with a healthcare suite; the single letter is a unit, a size, a grade, a
    language tag or a typo, and it matches broadly while discriminating nothing.

    `test_single_letter_terms_are_dropped_on_purpose` pins the *number*; this pins
    the *behaviour*, so deleting either the constant or the comparison is caught.
    """
    assert _terms("vitamin c deficiency") == ['"vitamin"', '"deficiency"']
    assert _terms("nurse x") == ['"nurse"']
    assert _terms("grade b nurse") == ['"grade"', '"nurse"']


def test_a_repeated_word_is_not_scored_twice(db_path: Path) -> None:
    """FTS5 scores each *operand*, so a repeated term multiplies a document's score.

    `"nurse" OR "nurse" OR "nurse"` scores three times `"nurse"`, which inflates
    every document containing the word and reorders the results by how often the
    user happened to repeat themselves — "nursing nursing, what about nursing?"
    would outrank a better turn purely on repetition. The fix is to keep one
    operand per distinct word.
    """
    _record(db_path, "r1", "what skills does a nurse need?", ("nurse",))

    assert _terms("nurse nurse nurse") == ['"nurse"']
    once = _search(db_path, "nurse")[0].score
    thrice = _search(db_path, "nurse nurse nurse")[0].score
    assert once == pytest.approx(thrice), f"repetition changed the score: {once} vs {thrice}"


def test_an_irrelevant_question_still_recalls_nothing(db_path: Path) -> None:
    """Widening the join must not turn recall into returning everything.

    The obvious risk of OR: a disjunction matches more, so if the filter were
    wrong — an empty stopword list, or a question of nothing but function words —
    every document would come back and the recall block would be noise. It does
    not, because every surviving term still has to be in *some* document. Both
    halves are asserted: the content word is absent, and the question is padded
    with the scaffolding a real question would have.
    """
    _record(db_path, "r1", "what skills does a nurse need?", ("nurse",))
    _record(db_path, "r2", "how do I become a plumber?", ("plumber",))

    for question in (
        "astrophysics",
        "how do I bake sourdough bread at home",
        "zebra migration patterns",
        "what is the kubernetes ingress controller",
    ):
        assert _search(db_path, question) == [], f"{question!r} recalled something"


def test_recall_from_the_real_index_stays_inside_the_prompt_budget(db_path: Path) -> None:
    """The *pair* holds the budget when the retriever is at its widest.

    The seam's own slicing and clamping are unit-tested in `tests/test_recall.py`
    with a stub, and that is the right place for them. This test is only worth
    having for the thing a stub cannot show: the real retriever returning far
    more documents than the budget allows, and the block still coming out the
    other side intact.

    So the first assertion is the load-bearing one — that the query genuinely
    matched more than `RECALL_LIMIT` documents. Without it this is the test it
    used to be: a duplicate of the seam tests that "passes whatever the
    retriever does", which means it would have stayed green if recall started
    matching nothing at all. The order — assert the wide match, *then* the
    budget — is what makes it sensitive to a recall regression as well as a
    seam regression.
    """
    for i in range(50):
        _record(db_path, f"r{i}", f"skills question {i} " + "detail " * 30, ("nurse", "icu"))
    retriever = Fts5EpisodeRetriever(db_path=db_path)

    matched = retriever.search("skills", limit=100)
    assert len(matched) > RECALL_LIMIT, (
        "this test is only meaningful while the real retriever matches more "
        "documents than the budget renders"
    )

    out = recall_prefix("skills", retriever=retriever)

    bullets = _bullets(out)
    assert len(bullets) <= RECALL_LIMIT
    assert bullets, "a wide match still has to render something"
    for line in out.splitlines():
        assert len(line) <= _RECALL_LINE_CHARS, f"recall line over the ceiling: {line!r}"


# --- the retriever must never fail a turn ----------------------------------


def test_retriever_returns_empty_when_the_database_does_not_exist(tmp_path: Path) -> None:
    """A first run has no memory.db at all. That is not an error.

    And it must stay that way afterwards: `sqlite3.connect` *creates* a file, so
    a retriever that opened the path before asking whether it was there would
    leave a first run with an empty `memory.db` it never wrote anything to. The
    next reader cannot tell that from a database that exists and is empty, and
    `clear_episodes` would start reporting a store that has nothing in it.
    """
    absent = tmp_path / "absent.db"

    assert Fts5EpisodeRetriever(db_path=absent).search("nurse") == []

    assert not absent.exists(), "recall created the database it was asked to read"


def test_a_missing_index_table_recalls_nothing_instead_of_raising(
    db_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A database written before recall existed has no `episodes_fts` in it.

    That is the state every existing install is in until `sync_fts` runs, and it
    is the state a user who upgrades without backfilling stays in. `no such
    table` is an `OperationalError`, so the guard covers it — but silently
    returning nothing is indistinguishable from "no past turns", so it has to
    warn, or nobody ever learns the backfill did not run.
    """
    _record(db_path, "r1", "what skills does a nurse need?", ("nurse",))
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("DROP TABLE episodes_fts")
        conn.commit()
    finally:
        conn.close()

    with caplog.at_level(logging.WARNING, logger=LOGGER):
        assert _search(db_path, "nurse") == []

    assert _warnings(caplog), "a missing index was passed off as 'nothing to recall'"


def test_a_locked_database_recalls_nothing_instead_of_raising(
    db_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Another process holding a write lock must not stall or fail the turn.

    The obvious "fix" is a long busy timeout, which converts a failure into a
    multi-second hang on the interactive path. The timeout is therefore pinned
    as a decision below, and this test is the behaviour it buys: a locked
    database is an ordinary empty result plus a warning.
    """
    _record(db_path, "r1", "what skills does a nurse need?", ("nurse",))
    blocker = sqlite3.connect(db_path, timeout=0.1)
    try:
        blocker.execute("BEGIN EXCLUSIVE")
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            assert _search(db_path, "nurse") == []
    finally:
        blocker.rollback()
        blocker.close()

    assert _warnings(caplog), "a locked database was passed off as 'nothing to recall'"


def test_the_busy_timeout_is_short_because_a_stall_is_worse_than_a_miss() -> None:
    """The number, pinned as a decision, the way the seam pins its own budget.

    Python's default is five seconds. Every read here is one indexed `MATCH` on a
    local file that answers in microseconds, so five seconds of waiting means
    something else is wrong — and the user is waiting through it.
    """
    assert _BUSY_TIMEOUT_S <= 1.0


def test_a_database_that_will_not_open_recalls_nothing_instead_of_raising(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A path that exists but cannot be opened as a database.

    A directory where `memory.db` should be — a half-finished install, a mount
    that came back as something else. `sqlite3.connect` raises
    `OperationalError`, and it raises from `connect`, *outside* the query guard
    the corrupt-file test covers. Without its own guard the "never raises"
    contract on `search` is a claim about the wrong line of code.
    """
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        assert Fts5EpisodeRetriever(db_path=tmp_path).search("nurse") == []

    assert _warnings(caplog), "an unopenable database was passed off as 'nothing to recall'"


def test_a_corrupt_database_file_recalls_nothing_instead_of_raising(
    db_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Bytes that are not a database. `DatabaseError`, not `OperationalError`.

    The guard catches `sqlite3.Error`, which covers both — but a half-written
    file after a crash is the *likely* cause of this class, and a retriever
    that only caught the operational errors would turn a truncated file into a
    failed turn.
    """
    db_path.write_bytes(b"this is not a database" * 64)

    with caplog.at_level(logging.WARNING, logger=LOGGER):
        assert _search(db_path, "nurse") == []

    assert _warnings(caplog), "a corrupt database was passed off as 'nothing to recall'"


def test_a_row_with_unreadable_labels_still_recalls_the_turn(
    db_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A malformed `node_labels` degrades the display, it does not drop the turn.

    The label list is rendered as ` → nurse, icu`; a row that will not parse is
    one turn quoted without its labels, which is far better than losing the
    recall — and losing it silently is the failure mode of a retriever that
    parses inside the `try` and re-raises.
    """
    _record(db_path, "r1", "what skills does a nurse need?", ("nurse",))
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("UPDATE episodes_fts SET node_labels = ?", ("not json at all",))
        conn.commit()
    finally:
        conn.close()

    with caplog.at_level(logging.WARNING, logger=LOGGER):
        hits = _search(db_path, "nurse")

    assert [hit.run_id for hit in hits] == ["r1"]
    assert hits[0].node_labels == ()


# --- backfill --------------------------------------------------------------


def test_backfill_indexes_rows_written_before_the_index_existed(db_path: Path) -> None:
    """A user's existing history must become searchable without a re-record.

    Asserted on the whole hit, not on "search finds something": a backfill that
    rebuilt the index but left a column out would still pass a search-only check
    and then hand the prompt a hit with a hole in it. A backfill is a restore,
    so what it produces has to equal what a fresh record would have produced.
    """
    _record(
        db_path, "r1", "what skills does a nurse need?", ("nurse",), ts="2026-02-03T04:05:06+00:00"
    )
    _record(db_path, "r2", "how do I become a plumber?", ("plumber",))
    expected = Fts5EpisodeRetriever(db_path=db_path).search("nurse skills")

    # Drop the index the way an upgrade-from-old-version would.
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("DROP TABLE IF EXISTS episodes_fts")
        conn.commit()
    finally:
        conn.close()
    assert Fts5EpisodeRetriever(db_path=db_path).search("nurse") == []

    assert sync_fts(db_path=db_path) == 2
    assert Fts5EpisodeRetriever(db_path=db_path).search("nurse skills") == expected
    assert [h.run_id for h in Fts5EpisodeRetriever(db_path=db_path).search("nurse")] == ["r1"]


def test_the_backfill_is_idempotent(db_path: Path) -> None:
    """Running it twice must not double the corpus.

    An upgrade path that a user might reasonably run again, on a table that is
    *appended* to rather than rebuilt, turns every past question into a duplicate
    and the ranking into noise. Asserted on the row count, which is the thing
    that would be wrong, not just the visible results.
    """
    _record(db_path, "r1", "what skills does a nurse need?", ("nurse",))

    assert sync_fts(db_path=db_path) == 1
    assert sync_fts(db_path=db_path) == 1
    conn = sqlite3.connect(db_path)
    try:
        assert conn.execute("SELECT COUNT(*) FROM episodes_fts").fetchone()[0] == 1
    finally:
        conn.close()


def test_sync_fts_reports_zero_when_there_is_no_database(tmp_path: Path) -> None:
    """No file means nothing was indexed. Creating one to report "0" would be a lie."""
    assert sync_fts(db_path=tmp_path / "absent.db") == 0
    assert not (tmp_path / "absent.db").exists()


# --- a stale index, which is worse than a missing one ------------------------


def _legacy_database(db_path: Path, questions: Sequence[str]) -> None:
    """A `memory.db` from before the lexical index existed: rows, no `episodes_fts`."""
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            "CREATE TABLE episodes (run_id TEXT PRIMARY KEY, ts TEXT NOT NULL, suite TEXT, "
            "capability TEXT, plan TEXT, question TEXT NOT NULL, node_labels TEXT NOT NULL, "
            "warnings TEXT NOT NULL, satisfied INTEGER NOT NULL)"
        )
        conn.executemany(
            "INSERT INTO episodes (run_id, ts, suite, capability, plan, question, "
            "node_labels, warnings, satisfied) VALUES "
            "(?, ?, 'esco', 'locate', '[\"locate\"]', ?, '[]', '[]', 1)",
            [(f"old-{i}", _TS, question) for i, question in enumerate(questions)],
        )
        conn.commit()
    finally:
        conn.close()
    assert not db_path.exists() or not _table_exists(db_path, "episodes_fts")


def _table_exists(db_path: Path, name: str) -> bool:
    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (name,)
        ).fetchone()
    finally:
        conn.close()
    return row is not None


def test_an_upgraded_database_loses_its_history_and_says_so(
    db_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The worst state in this feature, and the one the tests used to miss.

    Every existing install is in this state until `sync_fts` runs. It is a trap
    in two steps, and only the second step is interesting:

    1. With no `episodes_fts`, the first post-upgrade `record_episode` calls
       `_ensure_fts`, which **creates** an empty table rather than failing.
    2. From then on every query *succeeds*. "no such table" never happens, so
       the warning that pointed at `sync_fts` never fires, and the five episodes
       the user already had are simply not there. No error, no log, no symptom
       the user could describe — just an assistant that has forgotten everything
       and looks like it always has.

    So the assertion is not "the table is missing" (that is the recoverable
    state) but "the index exists, the query succeeds, it returns nothing, and
    the user is told why".
    """
    _legacy_database(db_path, ["how do I become a nurse?", "career in healthcare"])
    _record(db_path, "new-1", "what skills does a plumber need?", ("plumber",))
    assert _table_exists(db_path, "episodes_fts"), "precondition: the upgrade created the table"
    assert _search(db_path, "nurse") == [], "precondition: the old turn is unrecallable"

    with caplog.at_level(logging.WARNING, logger=LOGGER):
        assert _search(db_path, "nurse") == []

    messages = [record.getMessage() for record in _warnings(caplog)]
    assert any("sync_fts" in message for message in messages), (
        f"a stale index was passed off as 'nothing to recall': {messages}"
    )


def test_the_drift_warning_remedy_actually_restores_the_history(
    db_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A warning that names a command nobody can run is not a remedy.

    The first half of this is that the warning tells the reader to run
    `sync_fts`; the second is that doing so brings back the turns. The second is
    the part that was untestable while `sync_fts` had zero non-test callers and
    no user-reachable entry point at all.
    """
    _legacy_database(db_path, ["how do I become a nurse?"])
    _record(db_path, "new-1", "what skills does a plumber need?", ("plumber",))
    assert _search(db_path, "nurse") == []

    assert sync_fts(db_path=db_path) == 2

    assert [h.question for h in _search(db_path, "nurse")] == ["how do I become a nurse?"]
    # caplog accumulates for the whole test, and the searches above legitimately
    # warned. Only what happens *after* the rebuild is under test here.
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        assert _search(db_path, "nurse")
    assert not _warnings(caplog), "the index is current; it should stop complaining"


def test_an_in_sync_index_does_not_warn(db_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """The other direction, or the check is just noise on every turn.

    A warning that fires whenever recall works is a warning nobody reads, so the
    count comparison has to be silent on a healthy database — including one with
    nothing in it, which is the state of every new install.
    """
    _record(db_path, "r1", "what skills does a nurse need?", ("nurse",))

    with caplog.at_level(logging.WARNING, logger=LOGGER):
        assert _search(db_path, "nurse")

    assert not _warnings(caplog), "a healthy index was reported as stale"


def test_the_drift_check_reads_and_never_writes(db_path: Path) -> None:
    """ "Recall never writes" has to survive the count comparison.

    The drift check is the newest thing in `_connect` and the only one that
    looks at two tables rather than one, so it is the obvious place for a
    repair-on-read to creep in. Asserted on the bytes: an empty database after
    the check, byte for byte, which is what a write would break even if it
    happened to write the same logical value.
    """
    _legacy_database(db_path, ["how do I become a nurse?"])
    _record(db_path, "new-1", "what skills does a plumber need?", ("plumber",))

    _search(db_path, "nurse")
    before = db_path.read_bytes()

    _search(db_path, "nurse")
    _search(db_path, "plumber")

    assert db_path.read_bytes() == before, "a read-only recall path wrote to the database"


# --- a user-reachable rebuild -------------------------------------------------


def test_the_rebuild_command_is_reachable_and_fixes_a_stale_index(
    db_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The documented remedy needs a command. Found by `rg sync_fts`: none.

    The drift warning tells an operator to run something. If that something is
    only reachable from a test, the warning is a dead end and the five episodes
    stay unrecallable forever. Driven through `cli.main` rather than by calling
    `sync_fts`, because the thing being tested is that a *user* can do it.
    """
    from talent_angels import cli

    _legacy_database(db_path, ["how do I become a nurse?"])
    _record(db_path, "new-1", "what skills does a plumber need?", ("plumber",))
    assert _search(db_path, "nurse") == []

    assert cli.main(["recall-rebuild", "--db", str(db_path)]) == 0

    assert [h.question for h in _search(db_path, "nurse")] == ["how do I become a nurse?"]
    assert "indexed" in capsys.readouterr().out


def test_the_rebuild_command_says_so_when_there_is_no_database(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """No database is not a success. Exit 1, so a script can tell."""
    from talent_angels import cli

    missing = tmp_path / "absent.db"

    assert cli.main(["recall-rebuild", "--db", str(missing)]) == 1
    assert "nothing to rebuild" in capsys.readouterr().err
    assert not missing.exists(), "the command created a database it was asked to read"


def test_the_rebuild_command_scrubs_a_database_the_erase_path_could_not(
    db_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """It doubles as a manual scrub, and the bytes are the proof.

    A database written by a build whose erase path only *unindexed* the rows
    still holds the user's tokens in `episodes_fts_data` — `search` returns
    nothing and the file still says their name. That is not reachable from any
    command on a current build, so it is simulated by writing the residue the old
    way; what is tested is that the reachable command removes it.
    """
    from talent_angels import cli

    _record(db_path, "r1", PII_QUESTION, ("nurse",))
    conn = sqlite3.connect(db_path)
    try:
        # The old erase: drop the rows, leave the postings.
        conn.execute("DELETE FROM episodes")
        conn.execute("DELETE FROM episodes_fts")
        conn.commit()
    finally:
        conn.close()
    raw = db_path.read_bytes().lower()
    assert [token for token in PII_TOKENS if token in raw], "precondition: residue is present"

    assert cli.main(["recall-rebuild", "--db", str(db_path), "--vacuum"]) == 0

    raw = db_path.read_bytes().lower()
    assert not [token for token in PII_TOKENS if token in raw], (
        "recall-rebuild did not scrub the tokenised residue"
    )
    assert _fts_data_rows(db_path) == _FTS_DATA_BASELINE
    assert "vacuumed" in capsys.readouterr().out


# --- erasure -----------------------------------------------------------------


def test_a_bulk_erase_leaves_no_tokenised_residue_in_the_file(db_path: Path) -> None:
    """The regression test for the leak that survived every table-level check.

    An FTS5 table stores the question twice: in its content rows and,
    tokenised, in the `episodes_fts_data` inverted index. `DELETE FROM
    episodes_fts` rewrites the content rows and only appends a *delete marker*
    to the postings — so `search` returns nothing, the table counts are zero,
    `PRAGMA secure_delete=ON` has nothing to overwrite (the rows are live in
    the b-tree, not free pages), `VACUUM` has nothing to reclaim, and the user's
    name is still in the file.

    So this asserts the three things that were simultaneously true while the
    PII was on disk, and cannot be true after an erase that is really an erase:

    1. the **lowercase token** is absent from the raw bytes — not the verbatim
       sentence, because the tokenizer de-punctuates and lowercases whatever it
       stores, so a verbatim search is blind to exactly this leak;
    2. the **whole** question, not one lucky word;
    3. `episodes_fts_data` is back at its **post-create baseline**, which is the
       number that says the inverted index itself is empty rather than merely
       unmatchable.
    """
    _record(db_path, "r1", PII_QUESTION, ("nurse",))
    assert _fts_data_rows(db_path) > _FTS_DATA_BASELINE, "precondition: the index is populated"

    assert clear_episodes(db_path=db_path) == 1

    raw = db_path.read_bytes().lower()
    leaked = [token for token in PII_TOKENS if token in raw]
    assert not leaked, (
        f"{leaked} still readable in memory.db after clear_episodes — the FTS5 "
        f"inverted index kept the tokens, so this is not erasure"
    )
    assert _fts_data_rows(db_path) == _FTS_DATA_BASELINE, (
        "episodes_fts_data did not return to its post-create baseline, so rows "
        "are still live in the index"
    )


def test_re_recording_a_run_leaves_no_tokenised_residue(db_path: Path) -> None:
    """The same guarantee for the per-run replace, which is a second leak path.

    `record_episode` used to `DELETE FROM episodes_fts WHERE run_id = ?` before
    inserting the new wording. Retrieval is correct afterwards — the new question
    is the only one searchable — and the old question's *tokens* are still in the
    file. A retried turn therefore rewrites what the assistant can see and
    leaves on disk what the user thought they had replaced.

    The corpus is shaped so both halves are provable: the second question shares
    `skills` with the first, so a "did the replace work?" check that only looked
    for the old wording's *unique* tokens could not tell a scrub from a partial
    one.

    The row count is the third half, and its baseline is measured *after the
    first record* rather than taken from `_FTS_DATA_BASELINE`. That is a
    deliberate difference from the bulk test: one current turn is legitimately
    still indexed here, so "back to the empty-table baseline" is the wrong
    assertion to write — it would be satisfied by dropping the user's surviving
    turn on the floor. What must not happen is the index *growing*, because a
    growing `episodes_fts_data` is a delete marker that was never reclaimed.
    """
    _record(db_path, "r1", PII_QUESTION, ("nurse",))
    baseline = _fts_data_rows(db_path)
    assert baseline > _FTS_DATA_BASELINE, "precondition: the first record is indexed"

    _record(db_path, "r1", "what skills does a plumber need?", ("plumber",))

    assert _fts_data_rows(db_path) == baseline, (
        f"the index grew from {baseline} to {_fts_data_rows(db_path)} rows on replace: "
        "the old question's postings are still there, just no longer reachable"
    )
    raw = db_path.read_bytes().lower()
    leaked = [token for token in PII_TOKENS if token in raw]
    assert not leaked, (
        f"{leaked} still readable in memory.db after re-recording run r1 — "
        f"replacing a turn has to scrub the old tokens, not just unindex them"
    )
    # And the surviving row really is the new wording, so this is not a scrub
    # that threw the index away.
    assert [h.question for h in _search(db_path, "plumber")] == ["what skills does a plumber need?"]


def test_an_erased_episode_is_no_longer_recalled(db_path: Path) -> None:
    """Forgetting has to cover the index, not just the table.

    An FTS index is a second copy of the user's words. Leaving it behind means
    `/reset-all` reports success and the text is still retrievable — the same
    class of bug as the SQLite free pages.
    """
    _record(db_path, "r1", "what skills does a nurse need?", ("nurse",))
    assert _search(db_path, "nurse") != []

    assert clear_episodes(db_path=db_path) == 1

    assert _search(db_path, "nurse") == []
    # Every token, not one of them. The single-word version of this assertion
    # passed against a database that still held `skills`, `need` and `what`:
    # `nurse` happened to land on a page the engine reused, and a test that
    # passes by luck is a test that will stop passing without anyone noticing.
    raw = db_path.read_bytes().lower()
    assert b"nurse" not in raw and b"skills" not in raw, (
        "the question text survived the clear in the FTS5 index's own pages"
    )


def test_a_search_closes_the_connection_it_opened(
    db_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every turn opens a database; none of them may leave it open.

    A leaked handle per recall is not a crash, it is a slow exhaustion — the
    kind of bug that only shows up on a laptop that has been open all day, and
    it would be blamed on something else. The `finally` is the only thing
    holding it, so the check is that the connection is unusable afterwards,
    which is also how a caller would notice.
    """
    _record(db_path, "r1", "what skills does a nurse need?", ("nurse",))

    opened: list[sqlite3.Connection] = []
    real_connect = sqlite3.connect

    def _tracking(*args: object, **kwargs: object) -> sqlite3.Connection:
        conn = real_connect(*args, **kwargs)  # type: ignore[arg-type]
        opened.append(conn)
        return conn

    monkeypatch.setattr(sqlite3, "connect", _tracking)

    assert _search(db_path, "nurse") != []

    assert opened, "the search opened no connection, so it read nothing"
    for conn in opened:
        with pytest.raises(sqlite3.ProgrammingError):
            conn.execute("SELECT 1")


# --- a SQLite without FTS5 -------------------------------------------------


def test_recording_an_episode_still_works_without_fts5(
    db_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The index is derived data, so a SQLite that cannot hold it is not a failure.

    `CREATE VIRTUAL TABLE ... USING fts5` raises on a build compiled without
    FTS5, and it raises inside `_connect` — which every episode write goes
    through. Left unguarded, that turns a *missing optional feature* into a
    broken turn for every user on that build, which is the opposite of recall
    being an enhancement. Simulated rather than asserted against a real build,
    because no CI runner here is going to be recompiled.
    """
    import talent_angels.memory.episodes as episodes

    monkeypatch.setattr(episodes, "_ensure_fts", lambda _conn: False)

    _record(db_path, "r1", "what skills does a nurse need?", ("nurse",))

    assert episodes.recent_episodes(db_path=db_path)[0].question == (
        "what skills does a nurse need?"
    )
    assert sync_fts(db_path=db_path) == 0
    assert clear_episodes(db_path=db_path) == 1, "the erase must still report honestly"


# --- the wiring ------------------------------------------------------------


def test_recall_mode_defaults_to_hybrid_and_normalises(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unset, blank and padded values all mean the default — not "some retriever".

    **The default is `hybrid`, deliberately, and this used to assert `off`.** It
    was `off` so a fresh install behaved exactly as it did before recall
    existed; it is `hybrid` now because the measured numbers said so (0 of 36
    questions wrongly-unanswered, at lexical's own precision, 0 irrelevant
    hits) and because a feature shipped dormant is a feature nobody discovers.

    What did not change is the normalisation: a `.env` holding `TA_RECALL=  `
    must resolve to the documented default rather than to a backend chosen by
    accident, and casing and padding must not select a different one.
    """
    from talent_angels.env import recall_mode

    monkeypatch.delenv("TA_RECALL", raising=False)
    assert recall_mode() == "hybrid"
    for value, expected in (
        ("", "hybrid"),
        ("   ", "hybrid"),
        ("LEXICAL", "lexical"),
        ("  Lexical  ", "lexical"),
        ("bogus", "bogus"),
    ):
        monkeypatch.setenv("TA_RECALL", value)
        assert recall_mode() == expected


def test_an_unrecognised_mode_falls_back_to_recalling_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unknown mode is not a crash, and does not silently pick *some* backend.

    A typo in a `.env` must not be a stack trace on the first turn, and must not
    guess either. `vector` is deliberately absent from this list: it was here
    while the backend did not exist, and is now implemented, so it belongs with
    the modes that resolve — see
    `test_vector_mode_builds_the_vector_retriever`.
    """
    from talent_angels.env import episode_retriever
    from talent_angels.memory.retrieval import NullRetriever

    for mode in ("semantic", "fts6", "yes"):
        monkeypatch.setenv("TA_RECALL", mode)
        retriever = episode_retriever()
        assert isinstance(retriever, NullRetriever), f"mode={mode!r} picked {retriever!r}"
        assert retriever.search("nurse") == []


def test_vector_mode_builds_the_vector_retriever(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`TA_RECALL=vector` resolves to the vector backend, not to nothing.

    The counterpart to the test above, and the assertion that the deferral this
    mode was held in has actually ended. It checks the *type*, not that a search
    works: building a retriever must not touch the network or the database, so
    this is still a pure configuration test.
    """
    from talent_angels.env import episode_retriever
    from talent_angels.memory.vector_retriever import VectorEpisodeRetriever

    monkeypatch.setenv("TA_RECALL", "vector")
    assert isinstance(episode_retriever(), VectorEpisodeRetriever)


def test_vector_mode_does_not_warn_that_it_is_unimplemented(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The "no backend yet" warning must not fire for a mode that has one.

    Load-bearing in the other direction from the old test: leaving the warning in
    place would tell every vector user their setting is broken, and a warning
    that is always wrong is worse than no warning.
    """
    from talent_angels.env import episode_retriever

    _unwarned_env(monkeypatch)
    monkeypatch.setenv("TA_RECALL", "vector")
    with caplog.at_level(logging.WARNING, logger="talent_angels.env"):
        episode_retriever()

    assert [r for r in caplog.records if r.name == "talent_angels.env"] == []


def test_recall_mode_normalises_but_does_not_validate(monkeypatch: pytest.MonkeyPatch) -> None:
    """The contract, in both halves, because only one half used to be true.

    `recall_mode` is a *reader*: it strips and lowercases, defaults blank to
    `off`, and hands anything else back verbatim. An earlier docstring claimed
    an unrecognised value was `off`, which was false — `TA_RECALL=bogus` came
    back as `"bogus"`. The behaviour was the right one (see
    `test_vector_mode_says_it_is_unimplemented_instead_of_going_quiet`: the
    verbatim value is how a documented-but-unimplemented mode is told apart from
    a typo), so the docstring was corrected rather than the code. This test is
    what stops the two from drifting apart again, and it is also the test a
    reviewer should read before "fixing" the verbatim return.
    """
    from talent_angels.env import recall_mode

    monkeypatch.setenv("TA_RECALL", "  BOGUS  ")
    assert recall_mode() == "bogus", "an unrecognised mode must come back as written"
    # Still harmless: the only value that selects a backend is `lexical`.
    monkeypatch.setenv("TA_RECALL", "bogus")
    from talent_angels.env import episode_retriever
    from talent_angels.memory.retrieval import NullRetriever

    assert isinstance(episode_retriever(), NullRetriever)


def test_lexical_mode_actually_selects_the_fts5_retriever(monkeypatch: pytest.MonkeyPatch) -> None:
    """The switch has to reach a real backend, not just stop returning the null one."""
    from talent_angels.env import episode_retriever

    monkeypatch.setenv("TA_RECALL", "lexical")
    assert isinstance(episode_retriever(), Fts5EpisodeRetriever)


def _unwarned_env(monkeypatch: pytest.MonkeyPatch):
    """`env` with the warn-once latch cleared, so one test cannot silence another.

    The latch is process-global by design — a warning that fires per turn is
    worse than no warning — which means it is also shared across the suite. Left
    to chance it would make the assertion below order-dependent: whichever test
    ran first would consume the warning and the rest would see nothing.
    """
    from talent_angels import env

    monkeypatch.setattr(env, "_warned_unavailable", set())
    return env


def test_a_mode_with_no_backend_warns_exactly_once(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A documented value with no backend gets one named, actionable line.

    `recall_mode()` advertises `off | lexical | vector`, so a value added to
    `recall_mode`'s docstring before its backend lands would otherwise give a
    null retriever and no signal at all that the setting is why recall is dead.

    The mode is injected rather than written out, because every documented mode
    has an implementation now — the table is empty, and a test that named a real
    one would fail the day that mode ships, which is the wrong reason to fail.
    Pinning the *mechanism* keeps it true across that edit.
    """
    from talent_angels import env
    from talent_angels.memory.retrieval import NullRetriever

    _unwarned_env(monkeypatch)
    monkeypatch.setattr(env, "_UNIMPLEMENTED_MODES", frozenset({"hypothetical"}))
    monkeypatch.setenv("TA_RECALL", "hypothetical")
    with caplog.at_level(logging.WARNING, logger="talent_angels.env"):
        retriever = env.episode_retriever()

    assert isinstance(retriever, NullRetriever), "the unimplemented value must not crash or guess"
    warnings = [r for r in caplog.records if r.name == "talent_angels.env"]
    assert len(warnings) == 1, f"expected one warning, got {caplog.text}"
    message = warnings[0].getMessage()
    assert "hypothetical" in message, f"the warning does not name the configured value: {message!r}"
    assert "lexical" in message, f"the warning does not say what to use instead: {message!r}"


def test_the_unimplemented_warning_cannot_fire_every_turn(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Once, not once per call — a per-turn warning trains everyone to ignore it.

    `episode_retriever()` runs on every turn, so a latch that did not exist would
    put the same line in the log forever, and the one warning that matters (a
    stale FTS index) would stop being readable.
    """
    from talent_angels import env

    _unwarned_env(monkeypatch)
    monkeypatch.setattr(env, "_UNIMPLEMENTED_MODES", frozenset({"hypothetical"}))
    monkeypatch.setenv("TA_RECALL", "hypothetical")
    with caplog.at_level(logging.WARNING, logger="talent_angels.env"):
        for _ in range(5):
            env.episode_retriever()

    assert len([r for r in caplog.records if r.name == "talent_angels.env"]) == 1


def test_a_mode_nobody_asked_for_is_not_warned_about(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """`lexical` and `off` are the honest answers for a typo, so they stay quiet.

    Warning about every unrecognised value would make the warning about a
    documented-but-unimplemented one meaningless — and a `.env` typo is the case
    the docstring explicitly promises to swallow in silence.
    """
    from talent_angels.env import episode_retriever

    _unwarned_env(monkeypatch)
    for mode in ("off", "lexical", "sematic"):
        monkeypatch.setenv("TA_RECALL", mode)
        with caplog.at_level(logging.WARNING, logger="talent_angels.env"):
            episode_retriever()

    assert [r for r in caplog.records if r.name == "talent_angels.env"] == []


def test_every_documented_mode_has_a_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No mode `recall_mode` advertises may be on the unimplemented list.

    Asserted through the table itself, so a future edit that adds `"lexical"` —
    or re-adds `"vector"` — cannot pass on the strength of the warning tests
    alone. Normalisation is still checked: the set is consulted after
    `recall_mode` has lowercased and stripped, so an entry that only matched one
    spelling would be caught by the same assertion.
    """
    from talent_angels.env import _UNIMPLEMENTED_MODES, recall_mode

    for mode in ("off", "lexical", "vector", "  VECTOR  "):
        monkeypatch.setenv("TA_RECALL", mode)
        assert recall_mode() not in _UNIMPLEMENTED_MODES, (
            f"{mode!r} is advertised by recall_mode but has no backend: {_UNIMPLEMENTED_MODES}"
        )


def test_the_unimplemented_table_is_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The documented gap has closed, so the table has nothing left in it.

    Separate from the test above because that one only proves the *advertised*
    modes are implemented. This one fails if a mode is added without being
    documented, which is the mistake that makes the previous test pass vacuously.
    """
    from talent_angels.env import _UNIMPLEMENTED_MODES

    assert _UNIMPLEMENTED_MODES == frozenset(), (
        f"modes without backends remain: {sorted(_UNIMPLEMENTED_MODES)}"
    )


def _seed_recallable_episode(db_path: Path) -> None:
    _record(db_path, "r1", "what skills does a nurse need?", ("nurse",))


def test_phrase_chat_carries_the_recall_block_when_recall_is_on(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The wiring itself: the block has to arrive in the system prompt.

    A retriever that nothing calls is a library, not a feature, and this is the
    only assertion that would notice the call being dropped from a prompt site —
    every other test in the file passes with the wiring deleted.
    """
    from talent_angels.llm.protocol import LLMResult, LLMUsage, Message
    from talent_angels.memory import episodes as episodes_module
    from talent_angels.session import phrase

    monkeypatch.setenv("TA_RECALL", "lexical")
    _seed_recallable_episode(episodes_module.episodes_db_path())

    class _Client:
        provider = "litellm"
        model = "test"

        def __init__(self) -> None:
            self.calls: list[list[Message]] = []

        def complete(self, messages: list[Message]) -> LLMResult:
            self.calls.append(messages)
            return LLMResult(text="Sure.", provider="litellm", model="test", usage=LLMUsage())

    client = _Client()
    phrase.phrase_chat(client, user_text="nurse skills", fallback="miss", hint="Search missed.")

    system_prompt = client.calls[0][0].content
    assert "what skills does a nurse need?" in system_prompt
    assert "not taxonomy fact" in system_prompt
    # The block sits between the other prefixes and the instructions, not at the
    # end where it would read as part of the trailing hint.
    assert system_prompt.index("not taxonomy fact") < system_prompt.index("You are LFX")


def test_phrase_map_carries_the_recall_block_when_recall_is_on(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Same for the second site, which is the one that actually answers questions.

    `phrase_map` runs on every successful lookup, so it is where "we covered this
    before" would be felt; `phrase_chat` covers the misses.
    """
    from talent_angels.contracts import AgentResult, NodeRef
    from talent_angels.llm.protocol import LLMResult, LLMUsage, Message
    from talent_angels.memory import episodes as episodes_module
    from talent_angels.session import phrase

    monkeypatch.setenv("TA_RECALL", "lexical")
    _seed_recallable_episode(episodes_module.episodes_db_path())

    class _Client:
        provider = "litellm"
        model = "test"

        def __init__(self) -> None:
            self.calls: list[list[Message]] = []

        def complete(self, messages: list[Message]) -> LLMResult:
            self.calls.append(messages)
            return LLMResult(
                text="Nurses need a range of skills.", provider="l", model="t", usage=LLMUsage()
            )

    result = AgentResult(
        capability="locate",
        suite="esco",
        nodes=[
            NodeRef(
                id="esco:occupation:1",
                suite="esco",
                source="esco",
                source_id="1",
                kind="Occupation",
                pref_label="nurse",
            )
        ],
        confidence=0.9,
    )
    client = _Client()
    phrase.phrase_map(
        client, question="nurse skills", result=result, fallback="fb", card="occupation: nurse"
    )

    assert "what skills does a nurse need?" in client.calls[0][0].content


def test_the_tool_loop_prompt_matches_the_question_not_the_decorated_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The loop's own prompt text must not be part of the recall query.

    `run_tool_loop` builds `user_content` as the question plus `kind=…` and the
    bound-node block. The terms are joined with **OR**, so searching on that
    string does not ask for an unsatisfiable query — it asks for a *diluted*
    one: `occupation`, `kind` and `bound` are all real words in the index, so
    each one widens the candidate set with turns that have nothing to do with
    what the user asked, and `LIMIT 3` then spends the recall budget on them.
    Silent, and it would look exactly like "recall has bad results".

    The corpus is shaped so a diluted query and an undiluted one are
    *distinguishable*, which is the point: seeding a single turn whose question
    is the query itself makes both forms return the same hit, so this test
    passed against a call site that queried `user_content` — which is the defect
    a previous report recorded as caught when it was not.
    """
    from talent_angels.assistant.agent_loop import run_tool_loop
    from talent_angels.llm.protocol import LLMResult
    from talent_angels.memory import episodes as episodes_module
    from tests.fakes.taxonomy import FakeToolResult
    from tests.test_agent_loop import FakeSuite, ScriptedToolClient

    monkeypatch.setenv("TA_RECALL", "lexical")
    db = episodes_module.episodes_db_path()
    _record(db, "r1", "what skills does a nurse need?", ("nurse",))
    # A turn that shares the word `occupation` with the loop's own decoration and
    # nothing else with the question. Searching the decorated string reaches it;
    # searching the question does not.
    _record(db, "r2", "is there an occupation survey I should read?", ("survey",))
    _record(db, "r3", "what is a career coach?", ("career",))

    client = ScriptedToolClient(
        [
            LLMResult(text='{"final":"Nothing more to add."}', provider="l", model="t"),
            # The loop forces a search on round 0 whatever the model asks for, so
            # the final answer has to be scripted again for round 1.
            LLMResult(text='{"final":"Nothing more to add."}', provider="l", model="t"),
        ]
    )
    run_tool_loop(
        question="what skills does a nurse need?",
        suite=FakeSuite(FakeToolResult(nodes=[])),
        suite_name="esco",
        llm_client=client,
        kind="occupation",
        bound_node=None,
    )

    system_prompt = client.calls[0][0][0].content
    assert "what skills does a nurse need? → nurse" in system_prompt, (
        "the tool-loop prompt did not carry a recalled turn"
    )
    # The discriminating half: the turn that shares a word with the decoration is
    # the one the dilated query adds, so its presence is the fingerprint of
    # `recall_prefix(user_content, …)`.
    assert "occupation survey" not in system_prompt, (
        "recall was queried on the decorated prompt: the bound-node decoration's "
        "own words pulled in a turn the question never mentions"
    )


def test_recall_only_reaches_a_prompt_when_it_has_something_to_say(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The invariant that survived the default flip.

    This asserted "no block by default" and its own docstring said it would have
    to be rewritten on purpose if the default ever flipped — so here it is, and
    the rewrite is not "now expect a block". The property that makes recall safe
    is narrower and more useful than "off by default": **a question with nothing
    to recall must leave the system prompt byte-identical.**

    That is what protects a user who never asked for recall, has no matching
    history, and is not a power user. It also holds for the default `hybrid`
    path, where the second rung is consulted only on an empty first result — so
    "nothing to say" has to stay byte-identical through *two* backends, one of
    which is an embedding call.
    """
    from talent_angels.llm.protocol import LLMResult, LLMUsage, Message
    from talent_angels.memory import episodes as episodes_module
    from talent_angels.session import phrase

    # Explicitly off, so this isolates the *matching* case from the default: the
    # question shares no content with the seeded episode, and the point is that
    # the prompt does not change rather than which mode produced the retriever.
    monkeypatch.setenv("TA_RECALL", "off")
    _seed_recallable_episode(episodes_module.episodes_db_path())

    class _Client:
        provider = "litellm"
        model = "test"

        def __init__(self) -> None:
            self.calls: list[list[Message]] = []

        def complete(self, messages: list[Message]) -> LLMResult:
            self.calls.append(messages)
            return LLMResult(text="Sure.", provider="l", model="t", usage=LLMUsage())

    client = _Client()
    phrase.phrase_chat(client, user_text="nurse skills", fallback="miss", hint="")

    assert "not taxonomy fact" not in client.calls[0][0].content


def test_every_recall_call_site_is_known_and_counted() -> None:
    """The four sites in this file's header, checked against the source.

    The header says there are four sites that render `recall_prefix`. That
    sentence is a claim about code that lives in four other modules, so it can go
    stale silently — and it did: `assistant.synthesize` built its system prompt
    from the profile and notes blocks and nothing else, and the suite was green,
    because nothing compared the claim to the code.

    So the count is derived rather than trusted. The scan is over `src/` for
    calls to `recall_prefix`, which is the thing the claim is about; a new call
    site fails this test and has to be added to the header, and a header that
    says four while the source says five is a failing test rather than a stale
    comment.

    Deliberately a *count and a set of module names*, not an exhaustive
    behavioural test of each site. The behaviour is covered next to each site —
    the loop's is above, `synthesize`'s in `tests/test_synthesize.py`, and the
    two phrasing calls' in `tests/test_phrase.py` — and duplicating it here
    would be a fifth copy to keep in step. This test's job is to make sure there
    is not a *sixth* site nobody is testing.
    """
    import talent_angels

    package_root = Path(talent_angels.__file__).parent
    definition = "def recall_prefix("
    sites: dict[str, int] = {}
    for path in sorted(package_root.rglob("*.py")):
        count = 0
        for line in path.read_text(encoding="utf-8").splitlines():
            # An import names the function; a call uses it. Counting the two
            # together would credit every module that merely imports it, and
            # subtracting the import line from a tally that never included it
            # would undercount every module that calls it twice. The definition
            # is neither, and it lives in the same package.
            if line.startswith(definition):
                continue
            if "recall_prefix(" in line and "import" not in line:
                count += line.count("recall_prefix(")
        if count:
            sites[path.relative_to(package_root).with_suffix("").as_posix()] = count

    assert sites == {
        "assistant/agent_loop": 1,
        "assistant/synthesize": 1,
        "session/phrase": 2,
    }, f"the call sites changed; update this file's header and the tests for them: {sites}"
