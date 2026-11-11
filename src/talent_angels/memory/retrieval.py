"""Find past turns that are relevant to the question being asked.

The one module callers import to reach episode history. It deliberately knows
nothing about *how* retrieval happens: it fixes the shape of a hit and says
nothing about where hits come from. A caller that reaches past this module for
an implementation has coupled itself to one particular answer, which is the
thing this file exists to prevent.

Why the seam exists at all: episodes are recorded on every turn
(``record_episode`` from ``assistant.turn`` and ``tui.app``) but until now
nothing read them, so the "we covered this before" behaviour was recorded and
never delivered. The seam lets a real retriever be added — and swapped — without
touching a caller.

The prompt budget is small on purpose. This block shares one system prompt with
``profile_prefix()`` (5 lines) and ``notes_prefix()``, and the profile card is
what makes the agent feel like it knows the user. A recall that grows with the
corpus would quietly evict it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

logger = logging.getLogger(__name__)

#: Hard cap on recalled episodes in the prompt. Not a default — a ceiling.
RECALL_LIMIT = 3

#: Per-line ceiling, so one verbose past question cannot eat the prompt.
#: Applies to the whole line, bullet and label suffix included. Counted in
#: **code points**, not bytes and not tokens: a 120-character line of emoji
#: measures 463 UTF-8 bytes against the same 120, so if this ceiling is ever
#: justified by tokens rather than by prompt shape it is up to 4x looser on
#: non-ASCII text than the number suggests. Revisit before tightening it.
_RECALL_LINE_CHARS = 120

#: Labels shown in a bullet before the rest collapse to a count. Capping the
#: count rather than the joined text is what keeps a long label list ending on a
#: label boundary instead of halfway through one.
_MAX_LABELS = 3

#: Legibility floor on the question, not a budget. It never shrinks a question
#: below this, so a long label suffix cannot reduce the quote to a couple of
#: words. It is **not** what keeps the line inside the ceiling — the clamp in
#: `_hit_line` is, and always was. The two only disagree once the label suffix
#: runs past 78 characters, and there the floor decides *who gives way*: the
#: question keeps its 40 characters and the clamp trims the tail of the label.
#: Semi-dead as a bound — 40 → 10 passes every test here, because the clamp has
#: already fixed the line length and nothing asserts the split. Kept because it
#: is the difference between a clipped question and an illegible one, and
#: because the measurement that says so belongs next to the constant.
_MIN_QUESTION_CHARS = 40

_HEADER = "[Past turns with this user — not taxonomy fact, do not cite as evidence]"
_BULLET = "- "
_ARROW = " → "
_ELLIPSIS = "…"


@dataclass(frozen=True, slots=True)
class EpisodeHit:
    """One past turn that resembles the current question.

    Deliberately narrower than ``memory.episodes.Episode``: recall needs the
    question, the labels for display, and a score. Node ids are excluded because
    nothing in the prompt path should be citing them — a recalled turn is
    something the user asked before, not a taxonomy fact.

    ``score`` is **higher is better**, and that direction is this contract's
    rather than a scorer's: a backend that ranks better-when-lower must negate
    its score when it builds a hit, so nothing downstream has to know which
    backend ran. ``recall_prefix`` never re-sorts — the order is whatever
    ``search`` returned, because ranking is the retriever's job and its cost.
    """

    run_id: str
    ts: str
    question: str
    node_labels: tuple[str, ...]
    #: Relevance — higher is better, as above.
    score: float


@runtime_checkable
class Retriever(Protocol):
    """What callers ask for. Implementations decide how."""

    def search(self, question: str, *, limit: int = RECALL_LIMIT) -> list[EpisodeHit]:
        """Return past turns relevant to ``question``, best first.

        "Best" is the highest ``score`` — see ``EpisodeHit`` for why the
        direction is fixed here. The caller takes the order as given, so a hit
        list is only as ranked as this method makes it.

        Implementations must return ``[]`` rather than raise when they cannot
        answer — see ``recall_prefix``, which also guards.
        """
        ...


class NullRetriever:
    """The default. Retrieves nothing, costs nothing, breaks nothing.

    This is what makes the seam safe to land before any implementation exists:
    with this in place the assistant behaves exactly as it does today, so a later
    task's failures are unambiguously that task's.
    """

    def search(self, question: str, *, limit: int = RECALL_LIMIT) -> list[EpisodeHit]:
        return []


def _one_line(text: str, *, budget: int = _RECALL_LINE_CHARS) -> str:
    """Collapse to a single bounded line. A past question is a human sentence,
    not a paragraph, and a newline here would break the prompt block's shape."""
    flattened = " ".join(text.split())
    if len(flattened) <= budget:
        return flattened
    # One character is reserved for the ellipsis, so the result is never longer
    # than `budget`. `budget >= 1` at both call sites — `_hit_line` floors it at
    # `_MIN_QUESTION_CHARS`, the default is `_RECALL_LINE_CHARS` — which is why
    # the slice below needs no guard of its own: clamping it to 0 would return
    # one character against a zero budget rather than none.
    return flattened[: budget - 1].rstrip() + _ELLIPSIS


def _label_suffix(labels: tuple[str, ...]) -> str:
    """``" → nurse, icu"``, or ``""`` when there are no labels at all.

    Truncating the *joined* text would cut inside a label, so the list is capped
    by count and says how many it dropped. A single label long enough to blow
    the budget on its own still reaches the line clamp in ``_hit_line`` — the
    backstop, not the plan.
    """
    if len(labels) == 0:
        return ""
    if len(labels) > _MAX_LABELS:
        kept = ", ".join(labels[:_MAX_LABELS])
        return f"{_ARROW}{kept} +{len(labels) - _MAX_LABELS}"
    return f"{_ARROW}{', '.join(labels)}"


def _hit_line(hit: EpisodeHit) -> str:
    """One recalled episode as a single bullet, inside ``_RECALL_LINE_CHARS``.

    The budget belongs to the *line*, not to the question: the bullet and the
    label suffix spend it too, so a verbose question gives up characters rather
    than pushing the line over the ceiling the budget exists to enforce.
    ``_label_suffix`` keeps the suffix small, so the trailing clamp only has to
    catch the one case it cannot: a single label long enough to blow the budget
    on its own. The clamp is the ceiling, not the plan.
    """
    suffix = _label_suffix(hit.node_labels)
    room = _RECALL_LINE_CHARS - len(_BULLET) - len(suffix)
    question = _one_line(hit.question, budget=max(room, _MIN_QUESTION_CHARS))
    return _one_line(f"{_BULLET}{question}{suffix}")


def recall_prefix(
    question: str,
    *,
    retriever: Retriever | None = None,
    limit: int = RECALL_LIMIT,
) -> str:
    """A small labelled block for the system prompt, or ``""``.

    Returns ``""`` when there is nothing to recall, when no retriever is
    configured, when the retriever fails, and when ``limit`` — a ceiling, so
    zero or below recalls nothing — leaves no episode to show. Recall is an
    enhancement; it must never be the reason a turn errors out, so everything a
    retriever can do to this call — raise, hand back something only lazily
    consumable, hand back a hit missing a field — is caught here, logged, and
    swallowed rather than propagated.
    """
    if retriever is None:
        return ""

    try:
        hits = retriever.search(question, limit=limit)
        # `limit` is a ceiling, so zero or below recalls nothing. It has to be
        # floored at the slice: a negative index would mean "all but the last
        # N", which is not a cap and is never what a caller meant. Consuming
        # the hits is inside the guard, not after it: re-slicing and rendering
        # are as much part of the seam as the call itself, and a retriever that
        # breaks the protocol must fail here rather than in the caller's turn.
        # Bullets are built before the header so a caller passing limit=0 (or a
        # retriever that ignores `limit`) cannot produce a labelled block with
        # nothing under it.
        bullets = [_hit_line(hit) for hit in hits[: max(limit, 0)]]
    except Exception:
        # Deliberately broad: this is the seam that every later implementation
        # funnels through, and one of them raising must not fail a turn. Ruff's
        # blind-except rule (BLE001) is not enabled in this repo, and the
        # swallow is the documented contract, not an oversight.
        logger.warning("recall failed; continuing without it", exc_info=True)
        return ""

    if not bullets:
        return ""

    return "\n".join([_HEADER, *bullets]) + "\n\n"
