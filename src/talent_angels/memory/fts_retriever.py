"""Lexical episode recall using SQLite FTS5.

The first real ``Retriever``. Chosen to go first for three reasons:

1. FTS5 is already compiled into the bundled SQLite — no new dependency, no
   network, and it works in the offline test suite.
2. It needs no embedding model, so there is no provider, no dimension, and no
   cost to reason about.
3. It is the baseline. A vector index that cannot beat exact word overlap on
   "nurse skills" is not worth its dependency, and this makes that comparison
   cheap to run (Task 4).

Point 3 was a plan, and the plan has since been carried out — see the measured
table below.

**The vector path was built, measured, and lost.** ``evals.recall --vector``
scores both backends over this same corpus with the same loop:

| backend                    | zero-hit | p@1  | MRR@5 | irrelevant hits | widest |
| -------------------------- | -------: | ---: | ----: | -------------: | ------: |
| **lexical (this one)**     |    2/36  | 0.917| 0.926 |      **0**     |   12    |
| vector (3-small, 1536-dim) |  **0/36**| 0.889| 0.912 |    **210**     |   30    |

So this shipped. The reading is not that bm25 beats embeddings — on a corpus
where the words differ, it cannot, and the zero-hit column says so plainly
(0/36 against 2/36). It is that a dense retriever **always** returns its ``k``
nearest turns, so it never returns nothing, and the price of that is 210
irrelevant hits against 0: an off-topic question gets 30 past turns in the
prompt, none of them about anything the user asked. Vector fixed 1 query and
broke 2. On a 30-turn personal history that trade is not worth a network call
per turn and a second copy of the user's questions, so ``TA_RECALL`` stays
``lexical`` and the vector backend is kept as the *measured* alternative
rather than the default. Re-measure before reversing that; the numbers above
are the argument, and the harness that produced them is in the tree.

The trade-off that is real and worth stating: FTS5 matches words, not meaning.
"how do I become a nurse" will not recall "what training does nursing need".
That gap is real — the zero-hit column is exactly it — and a *hybrid* that
falls back to the vector backend only when this one returns nothing is the shape
this data points at. It is not built, because a recall floor is a new
behaviour, not a retriever, and it wants its own measurement.

Two deliberate non-features:

- **No write path.** Recall never writes. The index is built and repaired by
  ``memory.episodes`` (``record_episode`` / ``sync_fts``), so a turn that
  recalls nothing costs one read and cannot leave a half-built index behind. A
  retriever that rebuilt its own index on a miss would be writing to the user's
  memory inside a read.
- **No stemming or synonym expansion.** ``porter`` would help and would also make
  the baseline harder to reason about; the vector path is where meaning belongs.
  What is here is exact word overlap, which is the number the vector path has to
  beat.
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
from pathlib import Path

from talent_angels.memory.episodes import episodes_db_path
from talent_angels.memory.retrieval import RECALL_LIMIT, EpisodeHit

logger = logging.getLogger(__name__)

#: Words to match, from a question. Letters and digits, in any script, with
#: ``_`` excluded so an underscore-joined identifier is not one word: splitting
#: it gives two terms where the user meant one, and `vitamin_c` is a language
#: tag rather than two topics.
#:
#: This pattern is about **tokenisation**, not safety. It is worth being precise
#: about that, because the obvious claim — "nothing here can be FTS5 syntax" — is
#: false: the *keyword* operators ``AND``, ``OR``, ``NOT`` and ``NEAR`` are
#: ordinary alphanumeric words and all four survive `[^\W_]+` intact. The
#: module's own ``test_fts_syntax_in_a_question_is_matched_as_words_not_as_operators``
#: is the proof: a stored question containing ``NEAR`` is found by searching for
#: ``NEAR``, which only works because the term arrives at `MATCH` as ``"NEAR"``.
#: What *is* excluded is the punctuation — ``(``, ``)``, ``"``, ``{``, ``}``,
#: ``-``, ``:``, ``*``, ``^`` and the rest. So the safety comes from **quoting**
#: (`_terms`), and the character class's job is to decide what counts as a word
#: in the first place.
#:
#: The separate case question *is* settled by quoting too, in the other
#: direction: FTS5 matches its keywords case-sensitively, so an unquoted
#: uppercase ``AND`` is a syntax error while a lowercase ``and`` is an ordinary
#: literal. That is why `_STOPWORDS` is case-folded — a user who types `AND`
#: means the word, and must not fail the turn.
_TOKEN = re.compile(r"[^\W_]+")

#: One-letter tokens are dropped: "a", "I", "x". FTS5 would rank them above the
#: word that carries the meaning, because bm25 rewards a term that appears in few
#: documents, and a single letter matches broadly while discriminating nothing —
#: it is a unit, a size, a grade, a language tag or a typo. Two-letter tokens
#: **survive** this filter and are handled by `_STOPWORDS` instead, which is the
#: right division: `or`, `not`, `in`, `to`, `is` and `how` are two letters and are
#: not noise, while `am` and `an` are not either.
#:
#: This is the *length* filter, not a stopword list, and it is not a substitute
#: for one. See `_STOPWORDS`.
_MIN_TERM_CHARS = 2

#: English function words, dropped on top of the length filter.
#:
#: Two different jobs, and it is worth keeping them apart because only the first
#: one is an improvement.
#:
#: **1. The conjunctions make the query unsatisfiable.** FTS5 treats
#: space-separated MATCH operands as AND, so before this list `nurse or nursing`
#: was the query `"nurse" AND "or" AND "nursing"` — and no stored question
#: contains the word "or", so the query returned nothing. Measured on the
#: 30-turn corpus in ``evals.recall``, over 36 answerable questions,
#: AND-joining returned **zero hits for 13 of them** (15/36 with no filter at
#: all, so the filter rescues 2 on its own — `lawyer nor chef` and `nurse not
#: paramedic`, whose extra operand was the only thing unsatisfiable about them —
#: and the join rescues the other 13), including `career near healthcare`,
#: `registered nurse AND nursing`, `teacher but no degree`, and the four
#: function-word-led questions at the end of ``QUERIES``. The failure is silent:
#: no error, no warning, just a recall that never fires, and the assistant
#: confidently behaves as though the user has never asked anything. That is the
#: worst failure shape available to a feature that is supposed to work.
#:
#: **2. They are what makes the disjunction safe.** This is the reason the list
#: is a closed-class list and not a short one. Join the terms with OR and keep
#: `how`, `do`, `in`, `at`, `what` and `is`, and the OR now matches on *those*:
#: they occur in most real questions, so an unrelated one matches them and the
#: disjunction returns the world. The full comparison, 43 queries over the same
#: 30 turns — 36 answerable and 7 that share no topic with anything in it:
#:
#: | design                    | zero-hit | p@1  | MRR@5 | irrelevant | widest | MATCHes |
#: | ------------------------- | -------: | ---: | ----: | ---------: | ------: | ------: |
#: | AND only (the bug)        |    13/36 | 0.639| 0.639 |          0 |     7  |      43 |
#: | AND, no stopword filter   |    15/36 | 0.583| 0.583 |          0 |     7  |      43 |
#: | AND, falling back to OR   |    13/36 | 0.639| 0.639 |          0 |     7  |      63 |
#: | OR, no stopword filter    |     2/36 | 0.833| 0.880 |     **21** |    13  |      43 |
#: | OR + idf pruning (no list)|     2/36 | 0.833| 0.880 |     **21** |    13  |      43 |
#: | OR + stopwords (shipped)  |     2/36 | 0.917| 0.926 |      **0** |    12  |      43 |
#:
#: Two things in that table are worth reading carefully, because both cut against
#: what was written here before the harness existed.
#:
#: The **AND-to-OR fallback row is not a cheaper fix, it is the same fix at twice
#: the cost**: 63 MATCHes instead of 43 for byte-identical answers. It is in the
#: table because it is the tempting option, not because it is a good one.
#:
#: The **idf row is a no-op, and the numbers say the idea is worse than useless
#: rather than merely redundant**. bm25's idf passes through zero at half the
#: corpus, so "drop the terms that cannot change a score" is formally correct —
#: and prunes **nothing** here, because the most frequent term in the whole
#: query set is `nurse`, in 7 of 30 turns. Take the threshold below that and it
#: starts eating the corpus's actual subject matter: at 15% it prunes 4 terms and
#: p@1 falls to 0.528; at 10% it prunes 11 — `nurse`, `skills`, `teacher`,
#: `training` — and p@1 falls to 0.444. Irrelevant hits *do* fall, from 21 to 6.
#: That is the trap: the rule is inert until the day it is aggressive, and on the
#: day it is aggressive it prunes the words that carry the meaning. "Contributes
#: nothing to the score" and "carries no information" are different claims, and
#: this corpus is the disproof of the second. So the list stays hand-maintained,
#: and the classes it covers are asserted in
#: ``test_every_function_word_in_its_class_is_filtered``.
#:
#: The last two rows also **correct an earlier claim in this comment**, which said
#: filtering *costs* 0.066 of precision@1 and 0.026 of MRR. That was measured
#: before the query set contained any function-word-led question, so it could not
#: see the thing the list is for. With those questions in, filtering **gains**
#: 0.084 of p@1 and 0.046 of MRR while taking irrelevant hits from 21 to 0.
#:
#: But do not attribute that gain to the six words argument 3 is about. Measured
#: directly — the shipped list against the same list minus `are`, `am`, `any`,
#: `all`, `each`, `every`, `such` and the six subordinators — the two are
#: **identical**: same p@1 to four decimal places, same MRR, same zero-hits, same
#: irrelevant hits, and zero per-query top-1 differences out of 36. The reason is
#: checkable in one line: not one of those thirteen words appears in any of the
#: 30 stored turns, so as an OR operand they match nothing and cost nothing.
#: They are insurance, not an improvement, and saying otherwise would be the
#: second wrong number in this comment.
#:
#: That is also the honest limit of the measurement harness, and it is worth
#: stating rather than leaving to be discovered: `evals.recall` pins the list's
#: *effect*, not its *completeness*. Reverting these thirteen is a no-op on this
#: corpus and the harness stays green. What catches it is the structural
#: assertion in `test_every_function_word_in_its_class_is_filtered` — the
#: argument for them is that the members of a closed class are interchangeable,
#: not that they are frequent in a thirty-turn sample. A measurement cannot
#: support that claim, because the data that would show it does not exist yet.
#:
#: Deliberately **not** here: `no`, `without`, `need`, `become`, `much`,
#: `more`, `best`. They are not function words, and dropping them costs real
#: recall — `teacher but no degree` returns nothing without `no`, and
#: `can i become a teacher without a degree` ranks the "without a degree" turn
#: first because of `without`. A stopword list is a bet that a word never
#: discriminates, and the cost of being wrong is silent under-recall.
#:
#: **3. A closed class has to be closed, or the argument above stops applying.**
#: `is`, `was`, `were`, `be`, `been` and `being` were filtered and `are` and `am`
#: were not — the same verb, the same argument, and both are everywhere in real
#: questions. `_terms("are there any openings")` returned
#: `['"are"', '"any"', '"openings"']`, so the disjunction matched on two words
#: that appear in most of the corpus and the one word that names the topic was
#: diluted by a fixed 2-to-1. Same shape of gap in the determiners (`any`,
#: `all`, `each`, `every`, `such`) and the subordinators (`while`, `because`,
#: `although`, `unless`, `since`, `whether`): a list that covers the class
#: *partially* is indistinguishable from one that does not cover it, and the
#: failure is a silent widening rather than a silent narrowing.
#:
#: Note what that example is and is not. It shows the *mechanism* — the
#: disjunction is diluted 2-to-1 — and the measurement above shows it is not
#: worth 0.084 of p@1, because no turn in the sample contains those words. The
#: argument is that a stored question *will* contain them: a user who once asked
#: "are there any openings in nursing" put `are` and `any` in the index, and
#: today that turn is one term harder to find. Whether that has happened yet is
#: not knowable from a thirty-turn sample, which is why the fix is justified by
#: the closed-class argument and pinned by a structural test, not by a delta.
#:
#: The generalisation is the useful part: the members of a closed class are
#: interchangeable, so filtering some of them and not others cannot be defended
#: on content grounds — it can only be an oversight. Which is why
#: `test_every_function_word_in_its_class_is_filtered` now pins the classes as
#: classes, in both directions, rather than enumerating the five conjunctions.
#:
#: Case-folded, because FTS5's own tokenizer folds case on both sides of the
#: comparison: a user who types `AND` or `Not` means the same word, and the
#: quoted term would otherwise be a perfectly good literal that no stored
#: question contains.
_STOPWORDS = frozenset(
    {
        # conjunctions — the ones that made every disjunction return nothing
        "and",
        "but",
        "nor",
        "not",
        "or",
        "yet",
        "so",
        # determiners, prepositions, and the verb `be`
        "a",
        "all",
        "an",
        "any",
        "are",
        "each",
        "every",
        "such",
        "the",
        "at",
        "as",
        "be",
        "been",
        "being",
        "by",
        "for",
        "from",
        "in",
        "into",
        "is",
        "of",
        "on",
        "than",
        "then",
        "there",
        "to",
        "was",
        "were",
        "with",
        # subordinating conjunctions — the clauses they introduce are the
        # reason a *whole* question never matches, because no stored question
        # contains the word "because"
        "although",
        "because",
        "unless",
        "while",
        "since",
        "whether",
        # pronouns
        "he",
        "her",
        "his",
        "i",
        "it",
        "its",
        "me",
        "my",
        "she",
        "their",
        "them",
        "they",
        "us",
        "we",
        "you",
        "your",
        # auxiliaries and modals
        "can",
        "could",
        "did",
        "do",
        "does",
        "had",
        "has",
        "have",
        "am",
        "may",
        "might",
        "must",
        "shall",
        "should",
        "will",
        "would",
        # interrogatives — the scaffolding of a question about a topic
        "how",
        "what",
        "when",
        "where",
        "which",
        "who",
        "whom",
        "whose",
        "why",
        # fillers
        "about",
        "also",
        "else",
        "if",
        "just",
        "only",
        "very",
        # desire / intent verbs — the frame of a career question, not its topic.
        # Live: "I want to learn Python" recalled "I want to become a doctor"
        # on the word "want" alone.
        "want",
        "wants",
        "wanted",
    }
)

#: How the surviving terms are combined: **OR**, not AND.
#:
#: A conjunction between query terms is a claim that every one of them must be
#: in the document, and past questions are not written that way. "software
#: engineer or data analyst" is a user naming two options, not asking for a turn
#: containing the words "software", "engineer", "and" and "analyst"; the
#: conjunction narrows to a set no document is in, and the term that is in no
#: document at all should narrow nothing.
#:
#: Measured on the 30-turn corpus in ``evals.recall``, over 43 queries
#: (36 with a gold turn, 7 deliberately irrelevant), all designs sharing the
#: same tokenisation, dedupe, quoting and SQL:
#:
#: | design                    | zero-hit | p@1  | MRR@5 | irrelevant | widest | MATCHes |
#: | ------------------------- | -------: | ---: | ----: | ---------: | ------: | ------: |
#: | AND only (the bug)        |    13/36 | 0.639| 0.639 |          0 |     7  |      43 |
#: | AND, falling back to OR   |    13/36 | 0.639| 0.639 |          0 |     7  |      63 |
#: | **OR only (this one)**    |   **2/36**| **0.917** | **0.926** | **0** | **12** | **43** |
#:
#: The fallback was the design this could have shipped and it does not earn its
#: second query. It is byte-identical to OR-only in every quality column and
#: costs 63 indexed MATCHes against 43, on a synchronous path that runs before
#: the model is called. So OR-only is not a compromise between the two, it is
#: strictly better: same answers, 31% less work.
#:
#: Widening does not mean "matches everything", but that is a property of the
#: *filter*, not of this constant. Drop the filter, keep the OR, and the seven
#: irrelevant queries return 21 hits between them, the widest result set goes
#: from 12 to 13 of 30, and p@1 falls to 0.833 — see the table on `_STOPWORDS`.
#: Every surviving term still has to be in *some* document, and a query with no
#: term in the index still returns nothing.
#:
#: What OR gives up is precision on a query that names the topic only in
#: passing — "who is the best nurse in antarctica" recalls every nurse turn. That
#: is the right way round: a topic hit is a usable prompt line, and the failure
#: being fixed is silence.
#:
#: A caveat for whoever computes precision@k next, because it is not visible from
#: here: FTS5's `bm25()` uses `idf = ln((N - n + 0.5) / (n + 0.5))` and clamps the
#: result to a magnitude of 1e-6, and that is zero once a term occurs in
#: `n >= N/2` documents. A term in half the corpus therefore contributes nothing
#: to the score, and on a two-document corpus *every* term is in half of it —
#: which is where the `[1e-06, 1e-06]` in the bug report comes from. It is a
#: property of bm25 on a tiny corpus, not of this fix, and nothing in the
#: retriever can repair it. On this 30-turn corpus the most frequent surviving
#: query term is `nurse` at 7/30, so no query term is saturated and the scores
#: are real numbers (0.1 to 10, with 1.2x to 8.2x spread across the hits of a
#: single query). A benchmark run on a handful of turns will be measuring
#: rowid order, not ranking, and should not be used to compare two designs.
_JOIN = " OR "

#: The index row count against the source row count. An index with fewer rows
#: than `episodes` has is *stale*, and a stale index is the silent failure this
#: exists to catch.
#:
#: `record_episode` calls `_ensure_fts`, which **creates** an empty
#: `episodes_fts` on a legacy database that never had one. So the first turn
#: after an upgrade does not raise "no such table", it creates the table — and
#: from then on every query *succeeds*, returns whatever happens to be indexed,
#: and the "run `sync_fts`" warning is unreachable forever. The user's entire
#: recorded history is unrecallable with no signal at all, and the documented
#: remedy can no longer be discovered from the symptom. Counting the rows is
#: what makes the drift visible while there is still something to say about it.
#:
#: Measured at ~0.15x the cost of the `MATCH` it guards (0.011 ms against 0.067
#: ms on 30 turns; 0.65 ms against 4.2 ms on 3000), so it is paid on every
#: search rather than cached, and it is a read: recall still never writes.
_DRIFT_SQL = "SELECT (SELECT COUNT(*) FROM episodes_fts) < (SELECT COUNT(*) FROM episodes)"

#: How long to wait for a lock before giving up on the query. Python's default
#: is five seconds, and every read here is one indexed MATCH on a local file
#: that answers in microseconds. Five seconds of silence on the interactive path
#: is a worse failure than the miss it replaces, so the ceiling is a second.
_BUSY_TIMEOUT_S = 1.0

_SELECT = """
                SELECT run_id, ts, question, node_labels, bm25(episodes_fts) AS rank
                FROM episodes_fts
                WHERE episodes_fts MATCH ?
                ORDER BY rank
                LIMIT ?
                """


def _terms(question: str) -> list[str]:
    """Turn a question into quoted FTS5 terms, or ``[]`` if there are none.

    Quoting each term is what makes this injection-proof by construction: a
    term like ``AND`` or ``"`` becomes a literal string to match, never syntax.
    The FTS5 keywords are matched case-sensitively by its parser, so an
    unquoted ``AND`` in a user question is a syntax error that would fail the
    turn, and ``NEAR(`` runs off the end of the expression. Every term here is
    alphanumeric, so nothing else can be confused for syntax either. The caller
    joins the result with ``_JOIN``, and the operator — not the quoting — is what
    decides whether a query can match anything.

    Three filters, in order of how much each one is worth:

    1. **Length**, then **stopwords** (``_MIN_TERM_CHARS``, ``_STOPWORDS``). The
       second is not redundant with the first: ``or``, ``not``, ``in``, ``to``,
       ``is`` and ``how`` are all two characters or more and all survive it.
    2. **Deduplication.** FTS5 scores each *operand*, not each distinct term:
       ``"nurse" OR "nurse" OR "nurse"`` scores three times ``"nurse"``, which
       inflates every document containing it and reorders the results by how
       often the user happened to repeat themselves. Order of first appearance
       is kept, so the expression stays readable in a log line.
    3. Quoting, applied here and nowhere else.
    """
    seen: dict[str, None] = {}
    for token in _TOKEN.findall(question):
        if len(token) < _MIN_TERM_CHARS or token.lower() in _STOPWORDS:
            continue
        seen.setdefault(token, None)
    return [f'"{token}"' for token in seen]


class Fts5EpisodeRetriever:
    """Keyword search over past questions."""

    def __init__(self, *, db_path: Path | None = None) -> None:
        self._db_path = db_path

    def _connect(self) -> sqlite3.Connection | None:
        """Open the episode database, or ``None`` when it cannot be opened.

        ``None`` for a missing file is not an error: a first run has no
        ``memory.db`` at all, and "no history yet" is the truth rather than a
        failure. Neither is a database that exists but will not open — a
        directory where the file should be, a file this process cannot read.
        `sqlite3.connect` creates what is missing, so the existence check has to
        come first: otherwise a first run is left holding an empty database it
        never wrote anything to, which the next reader cannot tell apart from a
        real one.

        Also warns when the index is *stale* — present, queryable, and holding
        fewer rows than the table it is derived from (``_DRIFT_SQL``). The
        warning is the whole point: the queries do not fail, so nothing else
        would say so, and the state is otherwise indistinguishable from "you
        have never asked anything". Deliberately a read, so "recall never writes"
        survives the addition.
        """
        path = self._db_path or episodes_db_path()
        if not path.exists():
            return None
        try:
            conn = sqlite3.connect(path, timeout=_BUSY_TIMEOUT_S)
        except sqlite3.Error:
            logger.warning("fts5 recall could not open the episode database", exc_info=True)
            return None
        conn.row_factory = sqlite3.Row
        self._warn_if_stale(conn)
        return conn

    def _warn_if_stale(self, conn: sqlite3.Connection) -> None:
        """Say so when the index holds fewer rows than ``episodes``. Never raises.

        Every failure mode here is one ``search`` already handles — no FTS5
        build, a table missing, a locked file, a corrupt header — and this runs
        on a connection that has not been used for anything yet, so an exception
        escaping would break "recall never fails a turn" *earlier* and in a less
        legible place than the query guard. It logs at debug and returns instead.
        """
        try:
            stale = bool(conn.execute(_DRIFT_SQL).fetchone()[0])
        except sqlite3.Error:
            logger.debug("fts5 recall could not compare the index against the episodes table")
            return
        if stale:
            logger.warning(
                "fts5 recall index is out of date; run `sync_fts` "
                "(`python -m talent_angels.cli recall-rebuild`) to rebuild it — "
                "recorded turns are missing from recall until you do"
            )

    def search(self, question: str, *, limit: int = RECALL_LIMIT) -> list[EpisodeHit]:
        """Past turns matching ``question``, best first. Never raises.

        ``limit`` is a ceiling, so zero or below recalls nothing. Flooring it at
        one instead — the obvious way to keep ``LIMIT`` from reading as
        "unlimited" — answers a request for none with one hit, and the seam's
        re-slice hides it.

        "Matching" means matching **any** content word of the question
        (``_terms``, ``_JOIN``), not all of them. A question is not a document:
        "how do I become a nurse" contains four words no stored question would
        contain verbatim, and requiring all four returned nothing for almost
        every real question. The ranking is what recovers the precision, since
        ``bm25()`` sums over the terms a document did match and so puts the
        turn that matched most of the question first.
        """
        if limit <= 0:
            return []
        terms = _terms(question)
        if not terms:
            return []

        conn = self._connect()
        if conn is None:
            return []
        try:
            rows = conn.execute(_SELECT, (_JOIN.join(terms), limit)).fetchall()
        except sqlite3.Error:
            # A database with no fts5 build, an index that was never
            # backfilled, a locked or truncated file, a MATCH expression this
            # build dislikes. All of them are "nothing to recall", and the
            # warning is what tells an operator to run `recall-rebuild`.
            logger.warning("fts5 recall failed; continuing without recall", exc_info=True)
            return []
        finally:
            conn.close()

        return [self._hit(row) for row in rows]

    def _hit(self, row: sqlite3.Row) -> EpisodeHit:
        """One row to one hit. Degrades rather than raising.

        A row whose label list will not parse is still a turn worth quoting; the
        labels are the part that can be dropped. The question and the run_id
        are the part that makes it a hit at all.
        """
        try:
            labels = tuple(json.loads(row["node_labels"]))
        except (TypeError, ValueError):
            logger.warning("episode %s has unreadable labels; omitting them", row["run_id"])
            labels = ()
        # bm25() returns "smaller is better"; flip it so a higher score is a
        # better match, matching every other retriever's convention. It is
        # never NULL for a matched row, and `or 0.0` keeps a build that
        # disagrees from taking the whole recall down.
        return EpisodeHit(
            run_id=row["run_id"],
            ts=row["ts"],
            question=row["question"],
            node_labels=labels,
            score=-float(row["rank"] or 0.0),
        )
