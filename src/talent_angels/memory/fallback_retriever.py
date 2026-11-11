"""Recall by word-match first, and by meaning only when words find nothing.

The measured reason this exists
--------------------------------
Two backends were measured against each other on the same corpus (`evals.recall`):

* **word-match (lexical)** — 2 of 36 questions unanswered, precision 0.917,
  and **0** irrelevant hits. It loses only when the user's words change but the
  subject does not ("nursing" asked as "qualify to work as a nurse").
* **meaning (vector)** — 0 of 36 unanswered, precision 0.889, and **210**
  irrelevant hits.

The 210 is the whole problem, and it was structural rather than a bad model: a
nearest-neighbour search has no way to say "nothing here is relevant", so it
answers every question with its nearest neighbours. Given a personal history —
a few dozen turns on a handful of subjects — every stored turn is at least
somewhat near every question, so the "nearest" one is often unrelated.
`VectorEpisodeRetriever`'s relevance floor removes that behaviour, and this
module arranges the two so their strengths do not cancel each other out.

Why *this* order, and not a merged ranking
------------------------------------------
Merging (RRF, weighted fusion, then a reranker) is the better answer in general,
and it is the thing to build next. It is not built here because the measured
failure it would fix is small — the 2 questions above, about 5% — and because a
merge changes *ordering* for every query, whereas a ladder changes nothing at all
for the 34 queries word-match already answers. The ladder is the version whose
worst case is bounded by the backend that measured best.

The cost asymmetry is the point. On the common path this makes exactly the calls
the lexical backend already made: zero embedding calls, zero added latency. It
only reaches for the vector when lexical has already come back empty-handed, and
then only accepts a neighbour that clears ``MIN_RELEVANCE_SCORE``.

The honest cost: when lexical misses, the user is usually asking about something
genuinely new — and a vector backend's best guess at "new" is precisely the case
where it is least reliable. A late, quiet miss is the right failure here. It is
a miss, but it is a silent one.
"""

from __future__ import annotations

import logging

from talent_angels.memory.retrieval import RECALL_LIMIT, EpisodeHit, Retriever

logger = logging.getLogger(__name__)


class FallbackEpisodeRetriever:
    """``primary`` unless it finds nothing, then ``fallback``.

    Both are asked the same question and the primary's answer is returned
    untouched when it has one — no merging, no re-ranking, no deduplication
    against the other backend. A ladder that blended its two rungs would be a
    different design with different numbers, and those numbers would need
    measuring rather than assuming.

    Degradation is per-query and silent: if either backend raises, this returns
    the other's answer, and if both do, nothing. Recall is an enhancement and
    must not be the reason a turn fails.
    """

    def __init__(self, primary: Retriever, fallback: Retriever) -> None:
        self._primary = primary
        self._fallback = fallback

    def _ask(self, retriever: Retriever, question: str, limit: int) -> list[EpisodeHit]:
        try:
            return retriever.search(question, limit=limit)
        except Exception:
            logger.warning(
                "recall backend %s failed; trying the rest of the ladder",
                type(retriever).__name__,
                exc_info=True,
            )
            return []

    def search(self, question: str, *, limit: int = RECALL_LIMIT) -> list[EpisodeHit]:
        """The primary's hits, or the fallback's if the primary found none."""
        if limit <= 0:
            return []

        hits = self._ask(self._primary, question, limit)
        if hits:
            return hits[:limit]

        hits = self._ask(self._fallback, question, limit)
        if hits:
            logger.debug(
                "word-match found nothing for %r; answered from the meaning index",
                question[:60],
            )
        return hits[:limit]
