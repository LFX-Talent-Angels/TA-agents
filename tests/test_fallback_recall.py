"""Tests for the relevance floor and the fallback ladder.

Two separate claims are tested here, and they are the two that made the hybrid
worth having over the vector backend alone:

1. **The floor removes the "always answers" behaviour.** A nearest-neighbour
   search has no way to say "nothing here is relevant", which is what produced
   210 irrelevant hits in `evals.recall`. The floor is what gives it that
   ability, so it is tested as a filter rather than as a quality claim — the
   value is measured in `evals.recall` against a live provider, which is not
   what an offline suite is for.
2. **The ladder only reaches for the expensive backend when it has to.** The
   whole argument for putting lexical first is that the common path costs
   nothing extra, so that is asserted directly: on a primary hit the fallback
   is never asked, and therefore never embeds.

Offline throughout. No socket is opened.
"""

from __future__ import annotations

import logging

import pytest

from talent_angels.memory.embeddings import StaticEmbedder
from talent_angels.memory.fallback_retriever import FallbackEpisodeRetriever
from talent_angels.memory.retrieval import RECALL_LIMIT, EpisodeHit, Retriever
from talent_angels.memory.vector_retriever import (
    MIN_RELEVANCE_SCORE,
    VectorEpisodeRetriever,
)
from tests.conftest import MemoryHome

_TS = "2026-01-01T00:00:00Z"


def _hit(run_id: str, score: float) -> EpisodeHit:
    return EpisodeHit(
        run_id=run_id,
        ts=_TS,
        question=f"question {run_id}",
        node_labels=("nurse",),
        score=score,
    )


class _StubIndex:
    """An index that returns a fixed ranking, with `dimensions` to match.

    Scores are supplied rather than computed, because what is under test is the
    floor's *arithmetic* — where it cuts, and what it does to the tail of a
    ranking — not whether a hash-based embedder produces meaningful distances.
    """

    dimensions = 8

    def __init__(self, scores: list[float]) -> None:
        self._scores = scores
        self.calls = 0

    def hits(self, vector: list[float], *, limit: int) -> list[EpisodeHit]:
        self.calls += 1
        return [_hit(f"r{i}", s) for i, s in enumerate(self._scores[:limit], start=1)]


class _StubRetriever(Retriever):
    def __init__(self, hits: list[EpisodeHit] | None = None, *, raises: bool = False) -> None:
        self._hits = hits or []
        self._raises = raises
        self.asked = 0

    def search(self, question: str, *, limit: int = RECALL_LIMIT) -> list[EpisodeHit]:
        self.asked += 1
        if self._raises:
            raise RuntimeError("backend is down")
        return list(self._hits[:limit])


def _build_a_vector_index(memory_home: MemoryHome) -> None:
    """Put a real `episodes_vec` table in the isolated memory home.

    `is_indexed` asks only whether the table is present, not whether its width
    suits the caller, so an index built here with a 32-wide offline embedder is
    enough for the probe to succeed — and building a real one is what makes
    these tests exercise the same code path a configured install takes rather
    than a mock of it.

    An episode is recorded first because `memory.db` is created by the first
    write: with no history there is no file, and a vector index into a file that
    does not exist is not something the product supports either.
    """
    from talent_angels.memory.episodes import record_episode
    from talent_angels.memory.vector_index import SqliteVecIndex
    from talent_angels.runlog.models import ResultSummary, RunLogRecord

    record_episode(
        RunLogRecord(
            run_id="r1",
            ts=_TS,
            suite="esco",
            plan=["locate"],
            question="how do I become a nurse",
            result=ResultSummary(node_ids=["esco:occupation:1"], node_labels=["nurse"]),
        ),
        db_path=memory_home.db,
    )
    SqliteVecIndex(dimensions=32, db_path=memory_home.db).add("r1", [0.0] * 32)


# ---------------------------------------------------------------------------
# The floor
# ---------------------------------------------------------------------------


def test_a_hit_below_the_floor_is_not_recalled() -> None:
    below = MIN_RELEVANCE_SCORE - 0.01
    index = _StubIndex([below])
    retriever = VectorEpisodeRetriever(StaticEmbedder(dimensions=8), index=index)

    assert retriever.search("anything") == []


def test_a_hit_at_the_floor_is_recalled() -> None:
    """The floor is inclusive. A cut that dropped its own boundary would be a
    threshold nobody could reason about, because the documented value and the
    enforced value would differ by a hair."""
    index = _StubIndex([MIN_RELEVANCE_SCORE])
    retriever = VectorEpisodeRetriever(StaticEmbedder(dimensions=8), index=index)

    assert len(retriever.search("anything")) == 1


def test_the_floor_keeps_the_ranked_prefix_and_drops_the_tail() -> None:
    above, below = MIN_RELEVANCE_SCORE + 0.2, MIN_RELEVANCE_SCORE - 0.2
    index = _StubIndex([above, below, below])
    retriever = VectorEpisodeRetriever(StaticEmbedder(dimensions=8), index=index)

    hits = retriever.search("anything")

    assert [h.score for h in hits] == [above], "the tail below the floor survived"


def test_a_good_hit_beyond_the_limit_is_still_found(caplog: pytest.LogCaptureFixture) -> None:
    """The oversample, tested as the bug it prevents.

    Asking the index for exactly `limit` neighbours and filtering afterwards can
    return *fewer* than `limit` hits while a perfectly good one sat just outside
    the window — a ranking decision masquerading as a filtering one. The floor
    sits near the bottom of the score range, so this is the normal case, not an
    edge case.
    """
    # Ranks 1 and 2 are junk; rank 3 is the only good one, and limit is 2.
    junk, good = MIN_RELEVANCE_SCORE - 0.5, MIN_RELEVANCE_SCORE + 0.5
    index = _StubIndex([junk, junk, good, junk])
    retriever = VectorEpisodeRetriever(StaticEmbedder(dimensions=8), index=index)

    with caplog.at_level(logging.DEBUG, logger="talent_angels.memory.vector_retriever"):
        hits = retriever.search("anything", limit=2)

    assert [h.score for h in hits] == [good], (
        "a good hit was lost because the request window was not oversampled"
    )


def test_the_default_floor_is_the_calibrated_value() -> None:
    """Pinned so a casual edit to the constant is a visible test failure.

    -1.10 is a measurement: the 7 questions sharing nothing with any turn scored
    between -1.303 and -1.197, and the 36 answerable ones between -1.018 and
    -0.234. Anything outside (-1.197, -1.018] either leaks irrelevance or
    discards answerable questions, so re-calibrating is a deliberate act that
    should have to be argued for in a commit message.
    """
    assert MIN_RELEVANCE_SCORE == pytest.approx(-1.10)


def test_the_floor_is_a_cosine_similarity_and_not_a_bm25_score() -> None:
    """The lexical backend's scores are unbounded and negative; these are not.

    They share a `score` field and a "higher is better" direction, so a floor
    tuned on one and applied to the other would be silently meaningless. This
    states the two ranges apart so the fields are not later mistaken for one
    comparable quantity.
    """
    # A cosine distance of 0 (identical) is the best possible vector hit.
    assert MIN_RELEVANCE_SCORE < 0.0
    # ...and 2 (opposite) the worst, so every reachable score exceeds -2.
    assert -2.0 < MIN_RELEVANCE_SCORE


def test_a_custom_floor_overrides_the_calibrated_default() -> None:
    index = _StubIndex([-1.5])
    strict = VectorEpisodeRetriever(StaticEmbedder(dimensions=8), index=index, floor=-1.0)

    assert strict.search("anything") == [], "a raised floor must exclude more, not less"


# ---------------------------------------------------------------------------
# The ladder
# ---------------------------------------------------------------------------


def test_the_ladder_answers_from_the_primary_without_asking_the_fallback() -> None:
    """The cost claim, asserted rather than asserted-to.

    This is the entire reason lexical is the first rung: on the common path the
    vector backend must not be reached, because reaching it embeds the question
    and spends a network round trip to produce an answer that is then discarded.
    """
    primary, fallback = _StubRetriever([_hit("r1", -9.0)]), _StubRetriever([_hit("r2", -0.2)])

    hits = FallbackEpisodeRetriever(primary, fallback).search("nursing")

    assert [h.run_id for h in hits] == ["r1"], "the primary's answer must pass through untouched"
    assert fallback.asked == 0, "the fallback ran despite the primary having an answer"


def test_the_ladder_falls_back_only_when_the_primary_finds_nothing() -> None:
    primary, fallback = _StubRetriever([]), _StubRetriever([_hit("r2", -0.2)])

    hits = FallbackEpisodeRetriever(primary, fallback).search("qualify as a nurse")

    assert [h.run_id for h in hits] == ["r2"]
    assert fallback.asked == 1


def test_the_ladder_prefers_a_weak_primary_hit_to_a_strong_fallback_one() -> None:
    """It is a ladder, not a merge.

    Deliberately not the "best" answer by score. Word-match says it found
    something and that is trusted; blending the two rungs is a different design
    whose numbers would have to be measured rather than assumed.
    """
    primary = _StubRetriever([_hit("wordmatch", -30.0)])
    fallback = _StubRetriever([_hit("semantic", -0.1)])

    hits = FallbackEpisodeRetriever(primary, fallback).search("anything")

    assert [h.run_id for h in hits] == ["wordmatch"]


def test_the_ladder_survives_a_failing_primary() -> None:
    primary, fallback = _StubRetriever(raises=True), _StubRetriever([_hit("r2", -0.2)])

    hits = FallbackEpisodeRetriever(primary, fallback).search("anything")

    assert [h.run_id for h in hits] == ["r2"], "a broken first rung must not end recall"


def test_the_ladder_survives_a_failing_fallback() -> None:
    primary, fallback = _StubRetriever([_hit("r1", -9.0)]), _StubRetriever(raises=True)

    hits = FallbackEpisodeRetriever(primary, fallback).search("anything")

    assert [h.run_id for h in hits] == ["r1"]


def test_the_ladder_recalls_nothing_rather_than_raising_when_both_fail() -> None:
    ladder = FallbackEpisodeRetriever(_StubRetriever(raises=True), _StubRetriever(raises=True))

    assert ladder.search("anything") == [], "recall must never be the reason a turn fails"


def test_the_ladder_honours_the_limit() -> None:
    primary = _StubRetriever([_hit(f"r{i}", -9.0) for i in range(9)])

    assert len(FallbackEpisodeRetriever(primary, _StubRetriever()).search("x", limit=2)) == 2


def test_the_ladder_asks_nothing_for_a_non_positive_limit() -> None:
    primary, fallback = _StubRetriever([_hit("r1", -9.0)]), _StubRetriever([_hit("r2", -0.2)])

    assert FallbackEpisodeRetriever(primary, fallback).search("x", limit=0) == []
    assert primary.asked == 0 and fallback.asked == 0


# ---------------------------------------------------------------------------
# Wiring
# ---------------------------------------------------------------------------


def test_hybrid_mode_builds_a_ladder_with_the_floor_underneath(
    memory_home: MemoryHome, monkeypatch: pytest.MonkeyPatch
) -> None:
    from talent_angels import env
    from talent_angels.env import episode_retriever

    monkeypatch.setenv("TA_RECALL", "hybrid")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v-not-a-real-key")
    monkeypatch.setattr(env, "_warned_unavailable", set())
    _build_a_vector_index(memory_home)
    retriever = episode_retriever()

    assert isinstance(retriever, FallbackEpisodeRetriever)
    assert isinstance(retriever._fallback, VectorEpisodeRetriever)
    assert retriever._fallback._floor == MIN_RELEVANCE_SCORE
    assert not isinstance(retriever._primary, VectorEpisodeRetriever), (
        "the two rungs must be different backends, or the ladder is one rung"
    )


def test_vector_mode_is_still_the_bare_vector_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    """Adding the ladder must not have changed what `vector` means."""
    from talent_angels.env import episode_retriever

    monkeypatch.setenv("TA_RECALL", "vector")

    assert isinstance(episode_retriever(), VectorEpisodeRetriever)


def test_building_the_hybrid_embeds_nothing(
    memory_home: MemoryHome, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Construction is free; only a search embeds.

    `episode_retriever` runs on every turn at four prompt sites, so a
    construction that reached the network would tax all of them. The embedder
    only learns its own width on a successful call, so an unqueried one is
    direct evidence that no call was made.
    """
    from talent_angels import env
    from talent_angels.env import episode_retriever

    monkeypatch.setenv("TA_RECALL", "hybrid")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v-not-a-real-key")
    monkeypatch.setattr(env, "_warned_unavailable", set())
    _build_a_vector_index(memory_home)
    retriever = episode_retriever()

    # An embedder that has never been called still does not know its own
    # width, so the width staying unknown is the proof that nothing was asked.
    with pytest.raises(RuntimeError, match="unknown until the first successful call"):
        _ = retriever._fallback._embedder.dimensions


# ---------------------------------------------------------------------------
# Making `hybrid` the default, and keeping that safe
# ---------------------------------------------------------------------------
#
# The default was flipped from `off` to `hybrid` because the measured numbers
# said it was the better default. These tests are the reason that is defensible
# rather than reckless: a default is reached by installs nobody asked, so every
# way it can be unusable has to degrade instead of failing.


def test_the_default_is_hybrid_not_off(monkeypatch: pytest.MonkeyPatch) -> None:
    """The flip itself.

    Pinned because it is a product decision, not an implementation detail, and
    the ADR records it as a deliberate reversal of the previous one.
    """
    from talent_angels.env import recall_mode

    monkeypatch.delenv("TA_RECALL", raising=False)

    assert recall_mode() == "hybrid"


def test_hybrid_without_credentials_degrades_to_keyword(
    memory_home: MemoryHome, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """No key must mean keyword recall, not a failed request per question.

    Without this, the new default would hit the provider on every keyword miss
    in every install that has no embedding key — paying a network round trip to
    be told, repeatedly, what one local check could have established.
    """
    from talent_angels import env
    from talent_angels.env import episode_retriever

    monkeypatch.setenv("TA_RECALL", "hybrid")
    for key in ("OPENAI_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(env, "_warned_unavailable", set())

    with caplog.at_level(logging.WARNING, logger="talent_angels.env"):
        retriever = episode_retriever()

    assert not isinstance(retriever, FallbackEpisodeRetriever)
    assert "keyword recall is active" in caplog.text, (
        "a silent downgrade is indistinguishable from a bug"
    )


def test_hybrid_without_an_index_degrades_to_keyword(
    memory_home: MemoryHome, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Credentials but no built index: same downgrade, different fix.

    The two cases get separate messages because the remedy differs — a key is
    not a rebuild, and a rebuild is not a key. An operator told the wrong one
    would set a key and see no change, then conclude recall is broken.
    """
    from talent_angels import env
    from talent_angels.env import episode_retriever

    monkeypatch.setenv("TA_RECALL", "hybrid")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v-not-a-real-key")
    monkeypatch.setattr(env, "_warned_unavailable", set())

    with caplog.at_level(logging.WARNING, logger="talent_angels.env"):
        retriever = episode_retriever()

    assert not isinstance(retriever, FallbackEpisodeRetriever)
    assert "recall-rebuild --vector" in caplog.text, (
        "the message must name the command that fixes it"
    )


def test_the_downgrade_warning_is_emitted_once(
    memory_home: MemoryHome, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """`episode_retriever` runs on every turn at four prompt sites.

    A warning per turn is a warning nobody reads, which is how the ones that
    matter get ignored too.
    """
    from talent_angels import env
    from talent_angels.env import episode_retriever

    monkeypatch.setenv("TA_RECALL", "hybrid")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(env, "_warned_unavailable", set())

    with caplog.at_level(logging.WARNING, logger="talent_angels.env"):
        for _ in range(4):
            episode_retriever()

    assert caplog.text.count("keyword recall is active") == 1


def test_lexical_stays_silent_when_hybrid_would_warn(
    memory_home: MemoryHome, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Opting out of the meaning backend must also opt out of the complaint.

    `TA_RECALL=lexical` is the documented way to say "I know, keyword only", and
    a warning in response to it would be the app arguing with a decision the
    operator just made.
    """
    from talent_angels.env import episode_retriever

    monkeypatch.setenv("TA_RECALL", "lexical")
    for key in ("OPENAI_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(key, raising=False)

    with caplog.at_level(logging.WARNING, logger="talent_angels.env"):
        retriever = episode_retriever()

    assert not isinstance(retriever, FallbackEpisodeRetriever)
    assert "keyword recall is active" not in caplog.text


def test_vector_still_gets_its_backend_without_a_key(
    memory_home: MemoryHome, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Asking for `vector` explicitly is honoured, degraded or not.

    Deliberately not routed through the same downgrade as `hybrid`: `vector` is
    a specific request, and quietly substituting a different backend would mean
    the configured mode and the running one disagree. The retriever returns
    nothing on its own when it cannot embed, which is the honest outcome.
    """
    from talent_angels.env import episode_retriever

    monkeypatch.setenv("TA_RECALL", "vector")
    for key in ("OPENAI_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(key, raising=False)

    assert isinstance(episode_retriever(), VectorEpisodeRetriever)


def test_the_vector_retriever_skips_embedding_when_it_cannot(
    memory_home: MemoryHome, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The pre-flight check, at the layer that would otherwise pay for it.

    A stub embedder that counts calls and reports no credentials: the assertion
    is that it was never asked to embed, not that the search returned nothing.
    The difference is a network round trip.
    """
    from talent_angels.memory.embeddings import Embedder

    class _Unconfigured(Embedder):
        def __init__(self) -> None:
            self.embed_calls = 0

        @property
        def dimensions(self) -> int:
            return 8

        @property
        def configured(self) -> bool:
            return False

        def embed(self, texts: list[str]) -> list[list[float]]:
            self.embed_calls += 1
            return [[0.0] * 8 for _ in texts]

    embedder = _Unconfigured()
    retriever = VectorEpisodeRetriever(embedder)

    assert retriever.search("anything") == []
    assert embedder.embed_calls == 0, "an unconfigured embedder was asked to work anyway"
