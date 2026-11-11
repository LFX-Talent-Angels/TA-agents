"""Load a local `.env` without overriding a real shell export.

Also the one place that turns a setting into an object, because "which retriever
is this install using" is a configuration question and every caller that answered
it for itself would answer it differently.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

from dotenv import load_dotenv

if TYPE_CHECKING:
    from talent_angels.memory.retrieval import Retriever

logger = logging.getLogger(__name__)


def _repo_root() -> Path | None:
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    return None


def local_dotenv_candidates() -> tuple[Path, ...]:
    """Prefer cwd `.env`, then the repo-root file next to `pyproject.toml`."""
    seen: set[Path] = set()
    ordered: list[Path] = []
    for path in (Path.cwd() / ".env",):
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            ordered.append(path)
    root = _repo_root()
    if root is not None:
        path = root / ".env"
        resolved = path.resolve()
        if resolved not in seen:
            ordered.append(path)
    return tuple(ordered)


#: The taxonomy suites embed a search query with a local sentence-transformers
#: model. Loading it prints a Hub rate-limit warning and a weights progress bar
#: into the chat; neither is actionable for a user. Defaults only: an export or
#: a `.env` line still wins.
_QUIET_MODEL_LOADING = {
    "HF_HUB_VERBOSITY": "error",
    "HF_HUB_DISABLE_PROGRESS_BARS": "1",
    "TRANSFORMERS_VERBOSITY": "error",
    "TRANSFORMERS_NO_ADVISORY_WARNINGS": "1",
    "TOKENIZERS_PARALLELISM": "false",
}


def load_local_dotenv() -> Path | None:
    """Load the first existing candidate with ``override=False``.

    Shell exports always win. Values are never printed. Also sets quiet
    defaults for local model loading (see ``_QUIET_MODEL_LOADING``).
    """
    loaded: Path | None = None
    for path in local_dotenv_candidates():
        if not path.is_file():
            continue
        load_dotenv(path, override=False)
        if loaded is None:
            loaded = path
    for key, value in _QUIET_MODEL_LOADING.items():
        os.environ.setdefault(key, value)
    return loaded


def recall_mode() -> str:
    """``off`` | ``lexical`` | ``vector`` | ``hybrid``. Default ``hybrid``.

    Defaults to ``hybrid``, which is the best-measured mode: over the
    ``evals.recall`` corpus it answers 0 of 36 questions wrongly-unanswered
    (lexical's 2) at lexical's own 0.917 precision, with 0 irrelevant hits
    against plain ``vector``'s 210. ``vector`` on its own is implemented and is
    *not* recommended — a dense retriever always returns its ``k`` nearest turns
    and so can never return nothing, which is where the 210 came from. See
    ``memory/fts_retriever`` and ``memory/fallback_retriever``.

    **This reverses an earlier decision, deliberately.** Recall used to default
    to ``off`` so a fresh install behaved exactly as it did pre-recall. It now
    defaults on because the measured numbers say it is the better default and
    because a feature shipped dormant is a feature nobody discovers. Two things
    make that safe rather than reckless, and both are load-bearing:

    * ``hybrid`` is the default rather than ``vector``, so the common path costs
      nothing — no network, no embedding, no index — and only consults a
      provider when keyword search has already come back empty.
    * ``episode_retriever`` degrades to keyword-only when there is no embedding
      credential or no built index, warning once. So an install that cannot use
      the meaning index gets lexical, not a failed request per question.

    The value is **normalised, not validated**: lowercased and stripped, and
    unset or blank both become ``hybrid``, while anything unrecognised comes
    back verbatim. That asymmetry is deliberate. A typo must not crash the first
    turn, and it must not silently enable anything either — so an unknown value
    selects no backend at all and recalls nothing, which is the safe direction
    to be wrong in. It also keeps ``episode_retriever`` able to tell a *typo*
    (silent, harmless) from a *documented value this install cannot use*
    (warned once, with the command that would fix it). A normaliser would erase
    the one distinction an operator needs. The cost is that ``TA_RECALL=bogus``
    returns ``"bogus"`` — a truthful answer to "what is this install configured
    for", not a bug.
    """
    return os.environ.get("TA_RECALL", "hybrid").strip().lower() or "hybrid"


#: Documented modes with no implementation yet. **Empty**, and asserted empty
#: by ``test_the_unimplemented_table_is_empty`` so it cannot rot into a set of
#: modes nobody remembers shipping. ``vector`` was here, deferred because
#: meaning-based search needed both a compiled extension (``sqlite-vec``) and a
#: paid embedding provider and neither had a measured payoff; both cleared, it
#: was built, and the comparison that justified building it is in
#: ``evals.recall --vector``.
#:
#: Kept as an empty set rather than deleted, because it is what makes the next
#: deferral cheap to express: a new mode with no backend is a name here, and
#: gets a single named, actionable warning instead of silence.
_UNIMPLEMENTED_MODES: frozenset[str] = frozenset()

#: Warn-once latch, process-global. ``episode_retriever`` is called on every
#: turn, so a warning without this fires forever — and a log line that repeats
#: per turn is a log line nobody reads, which is how the warning that does
#: matter (a stale FTS index, in ``fts_retriever``) gets ignored too.
_warned_unavailable: set[str] = set()


def _warn_once(key: str, message: str) -> None:
    """Say something about this install's configuration exactly once.

    Separate from ``_warn_if_unavailable`` because that one reports a value
    nobody can use, and this one reports a value this install *can* use but has
    not set up — a different situation needing a different message, and both
    latched against the same repetition problem: a line per turn is a line
    nobody reads.
    """
    if key in _warned_unavailable:
        return
    _warned_unavailable.add(key)
    logger.warning(message)


def _warn_if_unavailable(mode: str) -> None:
    """Say once that a configured mode has no backend here. Never raises.

    Warnings, not errors: ``recall_mode``'s contract is that a bad ``.env`` must
    not crash the first turn, and this is that contract holding for the one bad
    value that is a *documented* choice rather than a typo. The message names the
    value the operator set and the one that works, because "recall is off" is
    not actionable and "vector has no backend; use lexical" is.
    """
    if mode not in _UNIMPLEMENTED_MODES or mode in _warned_unavailable:
        return
    _warned_unavailable.add(mode)
    logger.warning(
        "TA_RECALL=%s has no backend yet — recalling nothing. Set TA_RECALL=lexical "
        "for keyword recall over past turns.",
        mode,
    )


def episode_retriever() -> Retriever:
    """The retriever this install is configured to use. Never raises.

    The one place the switch meets an implementation, so all three system-prompt
    sites agree on what recall means today — and so adding the vector backend is
    a branch here rather than a decision repeated per caller. Prompt sites import
    *this*, not a retriever: reaching past the seam for a backend couples them to
    one particular answer, which is what ``memory.retrieval`` exists to prevent.

    Anything that is not ``lexical`` falls through to the null retriever, which
    is the safe direction to be wrong in: a typo recalls nothing, an unimplemented
    mode recalls nothing *and says so once* (``_warn_if_unavailable``). The
    import is deferred for the same reason recall is off by default: a process
    that never recalls should not pay for the lexical backend's module, and an
    import cycle between configuration and implementation is a nuisance to
    unpick later.
    """
    from talent_angels.memory.fts_retriever import Fts5EpisodeRetriever
    from talent_angels.memory.retrieval import NullRetriever

    mode = recall_mode()
    if mode == "lexical":
        return Fts5EpisodeRetriever()
    if mode in ("vector", "hybrid"):
        # Constructed lazily and defensively: this is the one mode whose
        # construction can fail for reasons unrelated to recall — a missing
        # `sqlite-vec` wheel, a build that refuses extension loading. Both mean
        # "no vector recall here", which is the same thing a wrong mode means,
        # so the fallback is shared rather than re-implemented per failure.
        try:
            from talent_angels.memory.embeddings import default_embedder
            from talent_angels.memory.vector_retriever import VectorEpisodeRetriever
        except ImportError:
            logger.warning(
                "TA_RECALL=%s needs sqlite-vec, which is not installed; "
                "recalling nothing. Set TA_RECALL=lexical for keyword recall.",
                mode,
            )
            return NullRetriever()
        embedder = default_embedder()
        if mode == "vector":
            return VectorEpisodeRetriever(embedder)
        from talent_angels.memory.fallback_retriever import FallbackEpisodeRetriever

        # `hybrid` is the default: keyword first, meaning only when keyword
        # finds nothing. With the local embedder (free, on-device) the index is
        # built as turns are recorded (`vector_retriever.index_episode`), so the
        # only install that falls back to keyword-only is one without an
        # embedder at all — said once, with the fix.
        if not embedder.configured:
            _warn_once(
                "no-embedder",
                "TA_RECALL=hybrid has no embedder, so only keyword recall is active. "
                "Install the free local model with `pip install -e '.[local-embed]'` "
                "(or set TA_EMBEDDING_MODEL to a hosted model you have a key for). "
                "Set TA_RECALL=lexical to silence this.",
            )
            return Fts5EpisodeRetriever()
        return FallbackEpisodeRetriever(Fts5EpisodeRetriever(), VectorEpisodeRetriever(embedder))
    _warn_if_unavailable(mode)
    return NullRetriever()
