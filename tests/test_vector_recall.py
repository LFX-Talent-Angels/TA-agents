"""Tests for the vector recall path: index, retriever, embedder, and erasure.

The offline suite never opens a socket. The network-touching class
(``LiteLLMEmbedder``) is tested by substituting litellm, and the storage layer
is tested with ``StaticEmbedder`` — a hash-based embedder with no semantic
content, chosen because its job here is to exercise batching, width checking,
ordering and search, which is where the bugs that matter are. Nothing in this
file asserts that recall is *good*, only that it is correct and that it cannot
leak; whether it is good is measured in ``evals.recall`` against a live
provider, which is not what an offline suite is for.
"""

from __future__ import annotations

import logging
import struct
from pathlib import Path

import pytest

from talent_angels.memory.embeddings import (
    DEFAULT_EMBEDDING_MODEL,
    Embedder,
    LiteLLMEmbedder,
    StaticEmbedder,
    configured_model,
)
from talent_angels.memory.episodes import clear_episodes, record_episode
from talent_angels.memory.retrieval import EpisodeHit, Retriever
from talent_angels.memory.vector_index import SqliteVecIndex, sqlite_vec_float32
from talent_angels.memory.vector_retriever import VectorEpisodeRetriever, episode_text
from talent_angels.runlog.models import ResultSummary, RunLogRecord
from tests.conftest import MemoryHome

_TS = "2026-01-01T00:00:00Z"
_DIMS = 32

TURNS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("r1", "how do I become a nurse", ("nurse", "icu")),
    ("r2", "python list comprehension help", ("python",)),
    ("r3", "what training does nursing need", ("nurse", "training")),
)


def _record(db_path: Path, run_id: str, question: str, labels: tuple[str, ...]) -> None:
    record_episode(
        RunLogRecord(
            run_id=run_id,
            ts=_TS,
            suite="esco",
            plan=["locate"],
            question=question,
            result=ResultSummary(node_ids=["esco:occupation:1"], node_labels=list(labels)),
        ),
        db_path=db_path,
    )


@pytest.fixture
def episodes_db(tmp_path: Path) -> Path:
    """A database with :data:`TURNS` recorded, so the index has something to hold."""
    db = tmp_path / "memory.db"
    for run_id, question, labels in TURNS:
        _record(db, run_id, question, labels)
    return db


@pytest.fixture
def embedder() -> StaticEmbedder:
    return StaticEmbedder(dimensions=_DIMS)


@pytest.fixture
def built(episodes_db: Path, embedder: StaticEmbedder) -> SqliteVecIndex:
    """An index holding every turn in :data:`TURNS`."""
    index = SqliteVecIndex(dimensions=_DIMS, db_path=episodes_db)
    for run_id, question, labels in TURNS:
        index.add(run_id, embedder.embed([episode_text(question, labels)])[0])
    return index


def _warnings(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.name.startswith("talent_angels.memory")]


class _BrokenEmbedder:
    """An embedder whose provider is down. The most likely real failure."""

    @property
    def dimensions(self) -> int:
        return _DIMS

    def embed(self, texts: list[str]) -> list[list[float]]:
        raise RuntimeError("429 Model busy, retry later")


class _NarrowEmbedder:
    """An embedder at a different width than the index — a model change."""

    @property
    def dimensions(self) -> int:
        return 8

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] * 8 for _ in texts]


class _UndimensionedEmbedder:
    """An embedder that has never succeeded, so has no dimension to report."""

    @property
    def dimensions(self) -> int:
        raise RuntimeError("no successful embed yet")

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] * _DIMS for _ in texts]


# --------------------------------------------------------------------------
# StaticEmbedder — the offline stand-in's own contract
# --------------------------------------------------------------------------


def test_static_embedder_is_deterministic() -> None:
    a = StaticEmbedder(dimensions=_DIMS)
    b = StaticEmbedder(dimensions=_DIMS)
    assert a.embed(["nursing"]) == b.embed(["nursing"])


def test_static_embedder_normalises_to_unit_length() -> None:
    # Cosine distance in `vec0` assumes unit vectors; a stand-in that skipped
    # this would make every offline search result a statement about magnitude.
    (vector,) = StaticEmbedder(dimensions=_DIMS).embed(["nursing"])
    assert sum(x * x for x in vector) == pytest.approx(1.0)


def test_static_embedder_returns_one_vector_per_input_in_order() -> None:
    vectors = StaticEmbedder(dimensions=_DIMS).embed(["a", "b", "c"])
    assert len(vectors) == 3
    assert len({tuple(v) for v in vectors}) == 3


# --------------------------------------------------------------------------
# The index: storage, widths, and absence
# --------------------------------------------------------------------------


def test_index_holds_every_episode(built: SqliteVecIndex) -> None:
    assert built.count() == len(TURNS)


def test_index_search_returns_nearest_first(
    built: SqliteVecIndex, embedder: StaticEmbedder
) -> None:
    # Hash embedding, so this asserts the *contract* — best first, capped at the
    # limit — not that the nurse turns beat the Python one. That claim is the
    # eval harness's job and needs a real model.
    hits = built.hits(embedder.embed(["nursing career pathway"])[0], limit=3)
    assert len(hits) == 3
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True)


def test_index_search_respects_a_lower_limit(
    built: SqliteVecIndex, embedder: StaticEmbedder
) -> None:
    (vector,) = embedder.embed(["nursing"])
    assert len(built.hits(vector, limit=1)) == 1


@pytest.mark.parametrize("limit", [0, -1])
def test_index_search_treats_limit_as_a_ceiling(
    built: SqliteVecIndex, embedder: StaticEmbedder, limit: int
) -> None:
    (vector,) = embedder.embed(["nursing"])
    assert built.hits(vector, limit=limit) == []


def test_index_refuses_a_vector_of_the_wrong_width(built: SqliteVecIndex) -> None:
    with pytest.raises(ValueError, match="8 dimensions"):
        built.add("r9", [0.0] * 8)


def test_index_refuses_to_reuse_a_table_at_another_width(
    built: SqliteVecIndex, episodes_db: Path
) -> None:
    # A model change must be a rebuild, not a silent redefinition: two indexes
    # over the same rows at different widths cannot be compared with each other.
    with pytest.raises(RuntimeError, match="recall-rebuild"):
        SqliteVecIndex(dimensions=8, db_path=episodes_db).add("r9", [0.0] * 8)


def test_index_search_with_a_wrong_width_vector_finds_nothing(built: SqliteVecIndex) -> None:
    assert built.hits([0.0] * 8, limit=3) == []


def test_index_on_a_missing_database_finds_nothing(tmp_path: Path) -> None:
    index = SqliteVecIndex(dimensions=_DIMS, db_path=tmp_path / "absent.db")
    assert index.hits([0.0] * _DIMS, limit=3) == []
    assert index.count() == 0


def test_search_with_no_index_table_finds_nothing(
    episodes_db: Path, embedder: StaticEmbedder
) -> None:
    # Episodes exist, no vector index does. This is the state of every install
    # that has never run `recall-rebuild`, and it must be quiet rather than fatal.
    index = SqliteVecIndex(dimensions=_DIMS, db_path=episodes_db)
    (vector,) = embedder.embed(["nursing"])
    assert index.hits(vector, limit=3) == []
    assert index.search(vector, limit=3) == []


def test_index_warns_when_behind_the_episodes_table(
    episodes_db: Path, embedder: StaticEmbedder, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.WARNING)
    index = SqliteVecIndex(dimensions=_DIMS, db_path=episodes_db)
    index.add("r1", embedder.embed(["only one"])[0])
    index.search(embedder.embed(["nursing"])[0], limit=3)
    assert any("out of date" in r.message for r in _warnings(caplog))


def test_clear_empties_the_index(built: SqliteVecIndex) -> None:
    built.clear()
    assert built.count() == 0


def test_clear_on_a_database_with_no_index_is_quiet(episodes_db: Path) -> None:
    SqliteVecIndex(dimensions=_DIMS, db_path=episodes_db).clear()  # must not raise


def test_dimensions_must_be_positive() -> None:
    with pytest.raises(ValueError):
        SqliteVecIndex(dimensions=0)


def test_float32_packing_is_little_endian_and_exact() -> None:
    packed = sqlite_vec_float32([1.0, -2.5, 0.25])
    assert struct.unpack("<3f", packed) == (1.0, -2.5, 0.25)


# --------------------------------------------------------------------------
# The retriever: the seam contract, and the network's failure modes
# --------------------------------------------------------------------------


def test_retriever_satisfies_the_seam(built: SqliteVecIndex, embedder: StaticEmbedder) -> None:
    assert isinstance(VectorEpisodeRetriever(embedder, index=built), Retriever)


def test_retriever_returns_episode_hits(built: SqliteVecIndex, embedder: StaticEmbedder) -> None:
    hits = VectorEpisodeRetriever(embedder, index=built).search("nursing", limit=3)
    assert hits and all(isinstance(h, EpisodeHit) for h in hits)
    assert {h.question for h in hits} <= {q for _, q, _ in TURNS}


def test_retriever_honours_the_limit(built: SqliteVecIndex, embedder: StaticEmbedder) -> None:
    assert len(VectorEpisodeRetriever(embedder, index=built).search("nursing", limit=2)) == 2


@pytest.mark.parametrize("limit", [0, -1])
def test_retriever_treats_limit_as_a_ceiling(
    built: SqliteVecIndex, embedder: StaticEmbedder, limit: int
) -> None:
    assert VectorEpisodeRetriever(embedder, index=built).search("nursing", limit=limit) == []


def test_a_failing_embedder_recalls_nothing_and_does_not_raise(built: SqliteVecIndex) -> None:
    assert VectorEpisodeRetriever(_BrokenEmbedder(), index=built).search("nursing") == []


def test_a_failing_embedder_is_reported(
    built: SqliteVecIndex, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.WARNING)
    VectorEpisodeRetriever(_BrokenEmbedder(), index=built).search("nursing")
    assert any("without it" in r.message for r in _warnings(caplog))


def test_a_query_embedding_at_another_width_is_refused_not_truncated(
    built: SqliteVecIndex, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.WARNING)
    assert VectorEpisodeRetriever(_NarrowEmbedder(), index=built).search("nursing") == []
    assert any("recall-rebuild" in r.message for r in _warnings(caplog))


def test_retriever_without_an_embedder_dimensions_recalls_nothing() -> None:
    assert VectorEpisodeRetriever(_UndimensionedEmbedder()).search("nursing") == []


def test_retriever_never_writes_to_the_index(
    built: SqliteVecIndex, embedder: StaticEmbedder
) -> None:
    # "Recall never writes" is what lets a read-path backend be swapped in
    # without re-auditing every caller for side effects.
    before = built.count()
    VectorEpisodeRetriever(embedder, index=built).search("nursing", limit=3)
    assert built.count() == before


def test_episode_text_puts_the_question_first() -> None:
    assert episode_text("how do I become a nurse", ("nurse", "icu")) == (
        "how do I become a nurse nurse icu"
    )


def test_episode_text_with_no_labels_is_just_the_question() -> None:
    assert episode_text("a question", ()) == "a question"


# --------------------------------------------------------------------------
# Erasure — the load-bearing tests
# --------------------------------------------------------------------------


def test_clearing_episodes_removes_the_vector_bytes_from_the_file(
    episodes_db: Path, embedder: StaticEmbedder
) -> None:
    """A cleared episode must leave nothing derived from it in the file.

    This is the test the design would fail without `_clear_vector_index`. The
    question text is erased and the search returns nothing — the JOIN to
    `episodes` finds no row — so every functional check passes while the
    embedding of "I am Priya Raman in Bangalore" sits in the database. An
    embedding is a function of the text, and this repository has twice shipped
    an erase that removed a row and left its contents behind.
    """
    secret = "I am Priya Raman and I live in Bangalore"
    db = episodes_db
    _record(db, "secret", secret, ("nurse",))
    index = SqliteVecIndex(dimensions=_DIMS, db_path=db)
    (vector,) = embedder.embed([episode_text(secret, ("nurse",))])
    index.add("secret", vector)

    packed = sqlite_vec_float32(vector)
    assert packed in db.read_bytes(), "precondition: the vector is in the file"

    clear_episodes(db_path=db)

    raw = db.read_bytes()
    assert packed not in raw, "the cleared episode's vector survived in the file"
    assert secret.encode() not in raw
    assert index.count() == 0


def test_clearing_episodes_on_a_database_with_no_vector_index_succeeds(episodes_db: Path) -> None:
    # Every install that never built a vector index must still be able to erase.
    assert clear_episodes(db_path=episodes_db) == len(TURNS)


# --------------------------------------------------------------------------
# LiteLLMEmbedder — with litellm substituted, never a socket
# --------------------------------------------------------------------------


class _FakeLiteLLM:
    """Stands in for litellm, recording the calls it was asked to make."""

    def __init__(self, dimensions: int = _DIMS, *, fail: bool = False) -> None:
        self.dimensions = dimensions
        self.fail = fail
        self.calls: list[tuple[str, int]] = []
        self.suppress_debug_info = True
        self.set_verbose = True

    def embedding(self, *, model: str, input: list[str], **kwargs: object) -> dict:  # noqa: A002
        if self.fail:
            raise RuntimeError("429 Model busy, retry later")
        self.calls.append((model, len(input)))
        return {
            "data": [
                {"index": i, "embedding": [float(i)] * self.dimensions} for i in range(len(input))
            ]
        }


@pytest.fixture
def fake_litellm(monkeypatch: pytest.MonkeyPatch) -> _FakeLiteLLM:
    fake = _FakeLiteLLM()
    import litellm

    monkeypatch.setattr(litellm, "embedding", fake.embedding)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    return fake


def test_embedder_meets_the_protocol(fake_litellm: _FakeLiteLLM) -> None:
    assert isinstance(LiteLLMEmbedder(), Embedder)


def test_dimension_is_unknown_until_the_first_call(fake_litellm: _FakeLiteLLM) -> None:
    # Guessing here is what produces a table that looks right and searches
    # that quietly return nothing.
    embedder = LiteLLMEmbedder()
    with pytest.raises(RuntimeError, match="until the first successful call"):
        _ = embedder.dimensions
    embedder.embed(["a"])
    assert embedder.dimensions == _DIMS


def test_embedder_batches_and_preserves_order(fake_litellm: _FakeLiteLLM) -> None:
    embedder = LiteLLMEmbedder(batch_size=2)
    vectors = embedder.embed(["a", "b", "c"])
    assert len(vectors) == 3
    # Two requests, 2 then 1 input — indices are per-request, so each batch
    # restarts at 0. What matters is that the *returned* order matches the
    # *input* order, which is what the [0.0, 1.0, 0.0] below pins.
    assert [n for _, n in fake_litellm.calls] == [2, 1]
    assert [v[0] for v in vectors] == [0.0, 1.0, 0.0]


def test_embedder_sorts_by_index_not_arrival(fake_litellm: _FakeLiteLLM) -> None:
    embedder = LiteLLMEmbedder()
    (first,) = embedder.embed(["only"])
    assert first[0] == 0.0


def test_embedder_of_nothing_makes_no_call(fake_litellm: _FakeLiteLLM) -> None:
    assert LiteLLMEmbedder().embed([]) == []
    assert fake_litellm.calls == []


def test_embedder_raises_when_the_provider_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    import litellm

    fake = _FakeLiteLLM(fail=True)
    monkeypatch.setattr(litellm, "embedding", fake.embedding)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    # The build path must be loud: a silent partial index is worse than a
    # failed command. Only the *read* path swallows, and that is
    # `VectorEpisodeRetriever`'s job, not this class's.
    with pytest.raises(RuntimeError, match="Model busy"):
        LiteLLMEmbedder().embed(["a"])


def test_embedder_refuses_a_dimension_change_mid_build(monkeypatch: pytest.MonkeyPatch) -> None:
    import litellm

    class Drifting(_FakeLiteLLM):
        def embedding(self, *, model: str, input: list[str], **kwargs: object) -> dict:  # noqa: A002
            self.calls.append((model, len(input)))
            return {
                "data": [
                    {"index": i, "embedding": [0.0] * (self.dimensions + i * 8)}
                    for i in range(len(input))
                ]
            }

    monkeypatch.setattr(litellm, "embedding", Drifting().embedding)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    # Both inputs in one request: a per-request width change, which is what a
    # provider silently swapping models mid-build would look like. Across
    # batches each request restarts its indices at 0, so a two-batch version of
    # this test would pass for the wrong reason.
    with pytest.raises(RuntimeError, match="dimension changed"):
        LiteLLMEmbedder(batch_size=8).embed(["a", "b"])


def test_embedder_refuses_a_short_response(monkeypatch: pytest.MonkeyPatch) -> None:
    import litellm

    class Short(_FakeLiteLLM):
        def embedding(self, *, model: str, input: list[str], **kwargs: object) -> dict:  # noqa: A002
            return {"data": [{"index": 0, "embedding": [0.0] * self.dimensions}]}

    monkeypatch.setattr(litellm, "embedding", Short().embedding)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    with pytest.raises(RuntimeError, match="refusing to build"):
        LiteLLMEmbedder().embed(["a", "b"])


def test_embedder_names_the_key_it_wants(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        LiteLLMEmbedder().embed(["a"])


def test_default_model_is_the_measured_default() -> None:
    # 3-small tied 3-large on separation (0.385 vs 0.400, n=4) at half the
    # width and a sixth of the price. If this changes, the measurement that
    # justified it is in the PR that changes it.
    assert DEFAULT_EMBEDDING_MODEL == "openai/text-embedding-3-small"


# --------------------------------------------------------------------------
# Which model this install uses
# --------------------------------------------------------------------------


def test_configured_model_is_the_default_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TA_EMBEDDING_MODEL", raising=False)
    assert configured_model() == DEFAULT_EMBEDDING_MODEL


@pytest.mark.parametrize("blank", ["", "   ", "\t"])
def test_a_blank_model_falls_back_rather_than_becoming_empty(
    monkeypatch: pytest.MonkeyPatch, blank: str
) -> None:
    # A present-but-empty value is a configuration slip. Reading it as the model
    # named "" would produce a provider error per turn instead of the default
    # that works.
    monkeypatch.setenv("TA_EMBEDDING_MODEL", blank)
    assert configured_model() == DEFAULT_EMBEDDING_MODEL


def test_a_configured_model_is_used_and_reported(
    monkeypatch: pytest.MonkeyPatch, fake_litellm: _FakeLiteLLM
) -> None:
    # The model has to reach the provider, not just be readable: a documented
    # env var that is never passed on is a lie in `.env.example`.
    monkeypatch.setenv("TA_EMBEDDING_MODEL", "openai/text-embedding-3-large")
    LiteLLMEmbedder().embed(["a"])
    assert fake_litellm.calls[0][0] == "openai/text-embedding-3-large"


def test_an_explicit_model_beats_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TA_EMBEDDING_MODEL", "from/env")
    assert LiteLLMEmbedder(model="explicit/wins")._model == "explicit/wins"  # noqa: SLF001


# --------------------------------------------------------------------------
# The path env.episode_retriever() actually takes
# --------------------------------------------------------------------------
#
# Everything above hands the retriever an `index=`. This section does not, and
# it exists because of a bug the rest of this file was structurally unable to
# catch: `_resolve_index` used to ask the embedder for `dimensions` *before*
# embedding anything, and that property is unknowable until the first successful
# call. So the retriever concluded it had no index and recalled nothing — on
# every query, against a fully built index. Every unit test passed an explicit
# index and never reached it; only the end-to-end path did.


def test_a_retriever_with_no_explicit_index_finds_the_real_one(
    memory_home: MemoryHome, embedder: StaticEmbedder
) -> None:
    """The path `env.episode_retriever()` builds: no `index=` argument at all.

    This is the regression test for a bug the rest of this file could not see.
    `_resolve_index` used to ask the embedder for `dimensions` *before*
    embedding anything, and that property is unknowable until the first
    successful call — so the retriever concluded it had no index and recalled
    nothing, on every query, against a fully built index. Every other test here
    passes an explicit `index=` and never reached that code; only the end-to-end
    path through `env.episode_retriever` did.
    """
    from talent_angels.memory.episodes import record_episode
    from talent_angels.runlog.models import ResultSummary, RunLogRecord

    for run_id, question, labels in TURNS:
        record_episode(
            RunLogRecord(
                run_id=run_id,
                ts=_TS,
                suite="esco",
                plan=["locate"],
                question=question,
                result=ResultSummary(node_ids=["esco:occupation:1"], node_labels=list(labels)),
            ),
            db_path=memory_home.db,
        )
    index = SqliteVecIndex(dimensions=_DIMS, db_path=memory_home.db)
    for run_id, question, labels in TURNS:
        index.add(run_id, embedder.embed([episode_text(question, labels)])[0])

    # A *fresh* embedder, whose dimension is not yet known — exactly what
    # `episode_retriever()` constructs on the first turn.
    retriever = VectorEpisodeRetriever(StaticEmbedder(dimensions=_DIMS))
    hits = retriever.search("nursing", limit=3)
    assert len(hits) == len(TURNS), "a retriever with no injected index recalled nothing"
    assert {h.question for h in hits} <= {q for _, q, _ in TURNS}


def test_the_resolved_index_uses_the_querys_own_width(
    memory_home: MemoryHome, embedder: StaticEmbedder
) -> None:
    """The index is opened at the width the query was just embedded at.

    Opened at any other width it would find a table of the wrong size, so this
    pins the happy path rather than the mismatch check: if the two numbers
    could differ, the check would fire on every normal query.
    """
    from talent_angels.memory.episodes import record_episode
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
    SqliteVecIndex(dimensions=_DIMS, db_path=memory_home.db).add(
        "r1", embedder.embed(["how do I become a nurse nurse"])[0]
    )

    retriever = VectorEpisodeRetriever(StaticEmbedder(dimensions=_DIMS))
    assert retriever.search("nursing", limit=1) != []
