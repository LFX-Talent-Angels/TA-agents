"""Meaning-based episode recall: embed the question, find the nearest past turns.

The second real ``Retriever``, and the one that earns the ``vector`` mode in
``TA_RECALL``.

It is deliberately thin. ``vector_index`` knows how to rank vectors and
``embeddings`` knows how to make them; this file owns exactly two decisions
that belong to neither — **what text gets embedded**, and **what happens when
the network is down** — and it is the only place in the vector path that makes
a network call on a turn.

**What gets embedded is the question and its labels, not the answer.** A past
turn is a thing the user asked, and that is the whole of what makes it
recalled. Embedding the answer would pull in whatever taxonomy nodes the
assistant happened to return, so a turn about nursing would match a later
question about teaching because both answers mentioned a course. The stored
text is the same text the lexical index holds, which keeps the two backends
comparable in ``evals.recall`` — the comparison is only meaningful if the two
are given the same input.

**A failed embedding is a failed turn unless this file prevents it.** The seam
contract is that ``search`` returns ``[]`` rather than raising, because recall
is an enhancement and must never be the reason a turn errors out. Embedding
violates every assumption underneath that: it is a network call, it can
timeout, the provider can be overloaded (which OpenRouter was, twice, while
this was being written), and it costs money. So every one of those becomes
``[]`` here, loudly, exactly as ``Fts5EpisodeRetriever`` degrades on a corrupt
index.

**The cost is one short string per turn.** Building the index embeds the whole
corpus, once, behind an explicit command. Querying embeds the current question
and nothing else. On a personal history that is a fraction of a cent in total
and one round trip in latency — which is the reason this is opt-in rather than
the default, and the number to watch if vector recall ever feels slow.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from talent_angels.memory.embeddings import Embedder
from talent_angels.memory.retrieval import RECALL_LIMIT, EpisodeHit
from talent_angels.memory.vector_index import SqliteVecIndex

logger = logging.getLogger(__name__)

#: How the stored episode is rendered before it is embedded. Kept identical to
#: what the lexical index holds for the same turn, so ``evals.recall`` compares
#: two ranking methods over one input rather than two ranking methods over two
#: different inputs. Labels come after the question so a turn's topic dominates
#: the vector and its node ids cannot.
_SEP = " "

#: The score below which a near neighbour is not evidence of anything.
#:
#: ``EpisodeHit.score`` is ``-distance``, and sqlite-vec's cosine distance is
#: ``1 - similarity``, so this is a cosine similarity of **-0.10**. Chosen from
#: measurement, not taste: over the 43-query eval corpus the 7 questions that
#: share nothing with any turn scored between -1.303 and -1.197, while the 36
#: answerable ones scored between -1.018 and -0.234. That leaves an empty band
#: from -1.197 to -1.018, and this sits in the middle of it.
#:
#: **The margin is thin — 0.09 on either side, and the "nothing is relevant"
#: class is only 7 queries.** Treat this as a starting value to re-calibrate
#: against real usage, not a constant. It is deliberately allowed to be wrong
#: in one direction only: below the floor the retriever returns nothing, which
#: is the failure that measured best. A false positive instead injects an
#: unrelated past turn into the prompt, and that is the 210-irrelevant-hit
#: failure this floor exists to remove.
MIN_RELEVANCE_SCORE = -1.10

#: How many neighbours to fetch per ``limit`` before the floor is applied. The
#: floor sits near the bottom of the range, so asking for exactly ``limit``
#: would let a below-floor hit push a good one out of the window.
_OVERSAMPLE = 4


def episode_text(question: str, node_labels: Sequence[str]) -> str:
    """The text both indexes see for one episode."""
    return _SEP.join([question, *node_labels]).strip()


class VectorEpisodeRetriever:
    """Nearest-neighbour recall over embedded past turns."""

    def __init__(
        self,
        embedder: Embedder,
        *,
        index: SqliteVecIndex | None = None,
        floor: float = MIN_RELEVANCE_SCORE,
    ) -> None:
        self._embedder = embedder
        self._index = index
        self._floor = floor

    def _resolve_index(self, dimensions: int) -> SqliteVecIndex:
        """The index, opened against ``dimensions`` — the embedder's own.

        Takes the dimension as an argument rather than asking the embedder for
        it, and that is not a stylistic choice. ``Embedder.dimensions`` is
        unknowable until the embedder's first successful call, so a retriever
        that asked first would find no dimension, decide it had no index, and
        recall nothing — forever, on every query, while the index sat fully built
        a file away. The order is therefore: embed, *then* open. Every unit
        test here passes an explicit ``index=`` and never noticed; the end-to-end
        path through ``env.episode_retriever`` did.
        """
        if self._index is not None:
            return self._index
        return SqliteVecIndex(dimensions=dimensions)

    def _probe_index(self) -> bool:
        """Whether a default-located index exists, without knowing its width.

        The width is learned from the first embedding, so this has to be asked
        with a placeholder rather than the real thing. A wrong width is
        harmless because ``is_indexed`` only asks whether the table is present,
        not whether it matches — and asking that second question is exactly the
        work this exists to avoid.
        """
        return SqliteVecIndex(dimensions=1).is_indexed

    def search(self, question: str, *, limit: int = RECALL_LIMIT) -> list[EpisodeHit]:
        """Past turns near ``question`` in meaning, nearest first. Never raises.

        ``limit`` is a ceiling on both sides: how many are embedded-and-asked-for
        and how many come back, because a retriever that ignores it is relying
        on the seam to re-slice, and the seam should not have to be the thing
        that makes a backend behave.
        """
        if limit <= 0:
            return []

        # Refuse before embedding, not after. Without a credential there is no
        # request to make, and without an index there is nothing to search, so
        # either way the answer is [] and the only cost of finding out was a
        # network round trip. This is what lets `TA_RECALL=hybrid` be the
        # default: an install that cannot use the meaning index skips it
        # silently instead of paying to rediscover that on every lexical miss.
        #
        # Read defensively. `Embedder` requires `configured`, but an object that
        # does not conform is exactly the kind of thing this method's "never
        # raises" promise is for, and an AttributeError about a duck-typed
        # collaborator is not a useful thing to surface mid-turn. An embedder
        # that cannot answer the question is assumed able to try, and the embed
        # call below reports the failure properly if it was not.
        if not getattr(self._embedder, "configured", True):
            logger.debug("no embedding credentials; skipping vector recall")
            return []
        if self._index is None and not self._probe_index():
            logger.debug(
                "no vector index built; skipping vector recall. Build one with "
                "`python -m talent_angels.cli recall-rebuild --vector`."
            )
            return []

        try:
            vectors = self._embedder.embed([question])
        except Exception:
            # Every failure mode an embedding has: no credentials, provider
            # overloaded, a timeout, a model that was retired. All of them mean
            # "no recall this turn", and none of them may reach the caller.
            logger.warning(
                "could not embed the query for vector recall; continuing without it",
                exc_info=True,
            )
            return []

        if not vectors or not vectors[0]:
            return []
        index = self._resolve_index(len(vectors[0]))

        if len(vectors[0]) != index.dimensions:
            logger.warning(
                "vector recall skipped: the query embedding is %s dimensions and the "
                "index is %s. Rebuild the index for the current model with "
                "`recall-rebuild`.",
                len(vectors[0]),
                index.dimensions,
            )
            return []

        # Ask for more than we may keep, then cut at the floor. Slicing after
        # the fact rather than lowering `limit` is deliberate: the floor is a
        # property of the *ranking*, so it has to be applied to the ranked list.
        # Asking the index for only `limit` and then filtering would silently
        # return fewer than `limit` hits whenever the tail falls below the floor
        # while a better hit was available — a ranking bug wearing a
        # correctness bug's clothes.
        ranked = index.hits(vectors[0], limit=limit * _OVERSAMPLE)
        kept = [hit for hit in ranked if hit.score >= self._floor]
        if len(kept) < len(ranked):
            logger.debug(
                "vector recall dropped %s of %s neighbours below the relevance floor of %.2f",
                len(ranked) - len(kept),
                len(ranked),
                self._floor,
            )
        return kept[:limit]
