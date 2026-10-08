"""Turn text into vectors, behind a seam the rest of recall never has to know.

The vector path needs a second thing the lexical path never needed: somebody
else's computer. FTS5 runs in-process against a local file; an embedding is a
network call with a latency budget, a bill, and a failure mode that has nothing
to do with the index. Keeping that behind a two-method protocol is what lets
``vector_index`` and ``vector_retriever`` stay testable offline — the tests
substitute a deterministic embedder and never open a socket.

Three things are decided here that are worth stating plainly, because each of
them is a choice the vector path could quietly get wrong.

**The dimension is discovered, not assumed.** ``vec0`` declares its dimension in
the schema, so picking a number before the model answers means the table is
wrong the first time somebody points ``TA_EMBEDDING_MODEL`` at a different
model. So the first response *is* the dimension, and the index refuses to
accept a vector of any other length rather than storing a table whose contents
cannot be compared with each other.

**A failed embed is a failed index, not a failed turn.** ``embed`` raises.
That is the opposite of the ``Retriever`` contract and deliberate: the caller
here is ``vector-rebuild``, an explicit maintenance command, where a silent
partial index is far worse than a loud failure. The *read* path never calls
``embed`` unguarded — ``vector_retriever`` turns a failure into "no recall",
which is the same bargain ``recall_prefix`` makes.

**Cost is real and is the user's bill.** Querying embeds one string per turn,
on the interactive path, before the model is called. On the 30-turn corpus that
is pennies; on a long history it is still pennies, because the *index* is built
once and the *query* is a single short string. The latency is the part that
matters, and it is why vector recall is opt-in rather than the default.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)

#: The model used when ``TA_EMBEDDING_MODEL`` is unset. ``text-embedding-3-small``
#: is 1536 dimensions and costs roughly a twentieth of a cent per million
#: tokens, which is small enough that a personal history is free and large
#: enough that it is not a toy. ``-3-large`` is 3072 dimensions and measurably
#: better on semantic similarity; it is a one-line ``.env`` change away and is
#: not the default only because the corpus it would be measured against is 30
#: turns of synthetic questions, where the quality gap is not the binding
#: constraint. Change the constant, not the schema: see the module docstring.
DEFAULT_EMBEDDING_MODEL = "openai/text-embedding-3-small"

#: Env var that overrides it. Read here rather than at the call sites so that
#: "which model is this install using" has exactly one answer, and so a
#: ``.env`` that names a model does not silently do nothing — which is the
#: failure this whole module is careful about elsewhere.
_MODEL_ENV = "TA_EMBEDDING_MODEL"


def configured_model() -> str:
    """The embedding model this install is configured to use.

    Blank falls back to the default rather than becoming a model named ``""``,
    for the same reason ``recall_mode`` maps a blank to ``off``: a value that is
    present but empty is a configuration slip, and the safe reading of it is the
    one that works.
    """
    return (os.environ.get(_MODEL_ENV) or "").strip() or DEFAULT_EMBEDDING_MODEL


#: OpenRouter proxies OpenAI's embedding models under its own prefix and is the
#: provider this repository's other models already use. The bare
#: ``openai/text-embedding-3-small`` spelling works when ``OPENAI_API_KEY`` is
#: set, so both are tried before giving up.
_ROUTER_PREFIX = "openrouter/"
_ROUTER_BASE_URL = "https://openrouter.ai/api/v1"

#: How many strings per request. OpenAI-compatible embedding endpoints cap a
#: request at 2048 inputs, and a personal history is thousands of turns at most,
#: so this exists to stay under the cap without asking about it.
_DEFAULT_BATCH = 128


@runtime_checkable
class Embedder(Protocol):
    """Text in, vectors out. The only thing recall needs from a provider."""

    @property
    def dimensions(self) -> int:
        """How many components a vector from this embedder has.

        Known only after the first successful call — it is read off the response
        rather than declared, because a wrong value produces a table that looks
        fine and searches that quietly return nothing.
        """
        ...

    @property
    def configured(self) -> bool:
        """Whether this embedder has somewhere to send a request, and so could
        work at all. False for a provider with no credentials, true for one that
        needs none.

        A pre-flight question rather than a failure: a caller asking this first
        can decline to try, where a caller that only discovers the problem by
        asking has already spent a network round trip finding out. That is what
        makes ``TA_RECALL=hybrid`` a safe default — an install with no key must
        not discover its absence once per question, forever.
        """
        ...

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """One vector per input, in order. Raises when the provider fails."""
        ...


def _batched(items: Sequence[str], size: int) -> Sequence[Sequence[str]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity, for the offline self-check and the tests."""
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


class LiteLLMEmbedder:
    """An ``Embedder`` backed by any litellm embedding model.

    Chosen because litellm is already a dependency for chat, so this adds no
    package — only a network call, and the provider question is answered by
    whichever key the install already has rather than by a new one.
    """

    def __init__(
        self,
        *,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        batch_size: int = _DEFAULT_BATCH,
    ) -> None:
        self._model = model or configured_model()
        self._api_key = api_key
        self._base_url = base_url
        self._batch_size = max(1, batch_size)
        self._dimensions: int | None = None

    @property
    def dimensions(self) -> int:
        if self._dimensions is None:
            raise RuntimeError(
                "embedding dimension is unknown until the first successful call; "
                "embed something first"
            )
        return self._dimensions

    @property
    def configured(self) -> bool:
        """Answered by *running* the credential lookup, not by re-implementing it.

        A second, simpler check of the same question would drift the moment
        someone added a provider, and would drift towards "thinks it is
        configured" — the direction that spends a request to find out otherwise.
        """
        try:
            self._request_kwargs()
        except RuntimeError:
            return False
        return True

    def _request_kwargs(self) -> dict[str, Any]:
        """Pick credentials for ``self._model``, or raise with something useful.

        The failure this replaces is a `401` from the provider with a message
        about authentication, which does not mention that the fix is a key in
        ``.env``.
        """
        model = self._model
        if not model.startswith(_ROUTER_PREFIX) and os.environ.get("OPENAI_API_KEY"):
            return {"api_key": os.environ["OPENAI_API_KEY"]}

        router_key = os.environ.get("OPENROUTER_API_KEY")
        if model.startswith(_ROUTER_PREFIX):
            return {
                "api_key": self._api_key or router_key,
                "base_url": self._base_url or _ROUTER_BASE_URL,
            }

        if router_key:
            return {
                "api_key": router_key,
                "base_url": self._base_url or _ROUTER_BASE_URL,
            }
        if self._api_key:
            return {"api_key": self._api_key}
        raise RuntimeError(
            "no embedding credentials: set OPENAI_API_KEY or OPENROUTER_API_KEY "
            f"in .env, or point TA_EMBEDDING_MODEL at a model you have a key for "
            f"(tried {model!r})"
        )

    def _call(self, batch: Sequence[str]) -> list[list[float]]:
        import litellm

        if hasattr(litellm, "suppress_debug_info"):
            litellm.suppress_debug_info = True
        if hasattr(litellm, "set_verbose"):
            litellm.set_verbose = False

        model = self._model
        if not model.startswith(_ROUTER_PREFIX) and os.environ.get("OPENAI_API_KEY"):
            pass
        elif not model.startswith(_ROUTER_PREFIX) and os.environ.get("OPENROUTER_API_KEY"):
            model = f"{_ROUTER_PREFIX}{model}"

        response = litellm.embedding(model=model, input=list(batch), **self._request_kwargs())
        data = response["data"]
        # OpenAI-compatible responses are indexable but not guaranteed ordered;
        # `index` is in the payload for exactly this, and sorting on it is what
        # keeps a vector attached to the text it was made from.
        rows = sorted(data, key=lambda row: row.get("index", 0))
        return [[float(x) for x in row["embedding"]] for row in rows]

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed every input, in order. Raises when the provider fails.

        Batched rather than one-at-a-time because rebuilding an index is
        thousands of strings and 2048-at-a-time is the difference between one
        request and dozens.
        """
        if not texts:
            return []

        out: list[list[float]] = []
        for batch in _batched(texts, self._batch_size):
            vectors = self._call(batch)
            if len(vectors) != len(batch):
                raise RuntimeError(
                    f"embedding provider returned {len(vectors)} vectors for "
                    f"{len(batch)} inputs; refusing to build an index whose rows "
                    "are not aligned with their text"
                )
            for vector in vectors:
                if self._dimensions is None:
                    self._dimensions = len(vector)
                elif len(vector) != self._dimensions:
                    raise RuntimeError(
                        f"embedding dimension changed mid-build: expected "
                        f"{self._dimensions}, got {len(vector)}"
                    )
            out.extend(vectors)
        return out


class StaticEmbedder:
    """A deterministic offline ``Embedder`` for tests and the offline suite.

    Hashes character n-grams into a fixed-width vector. It has no semantic
    content whatsoever and is not a substitute for a model — "nurse" and
    "nursing" are unrelated to it. What it does provide is a stable,
    offline stand-in that exercises batching, dimension checking, ordering, and
    the index's storage and search paths without a network call, which is the
    part of the vector path that has bugs.
    """

    def __init__(self, *, dimensions: int = 32) -> None:
        self._dimensions = dimensions

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def configured(self) -> bool:
        """Always True: hashing locally needs nothing, unlike a provider."""
        return True

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for text in texts:
            vec = [0.0] * self._dimensions
            for token in text.lower().split():
                for position, char in enumerate(token):
                    vec[(ord(char) * 31 + position) % self._dimensions] += 1.0
            norm = sum(x * x for x in vec) ** 0.5 or 1.0
            out.append([x / norm for x in vec])
        return out
