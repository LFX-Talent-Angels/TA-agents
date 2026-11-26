"""A fail-closed filter on model prose: keep a sentence only if its facts are this turn's facts.

The allowlist is the code-written text of the turn (the fact card or the structured fallback),
never the user's words (they would ground a false premise) and never memory. A sentence is
dropped when it carries a number, a name written as a name, an example list, an affirmation or
an ordering/identity claim that the allowlist does not hold. Any exception returns "": the
caller then shows its code-written text. See ADR-0014 for what this does and does not catch.
"""

from __future__ import annotations

import re
import unicodedata

_NUMBER = re.compile(r"\d+(?:[.,]\d+)*")
_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-ZÁÉÍÓÚÑÀ-ÖØ-Þ0-9¿¡\"“'(\[])|\n+")
_SPANS = re.compile(r"\*\*(.+?)\*\*|`([^`]+)`|\"([^\"]+)\"|“([^”]+)”")
_WORD = re.compile(r"[\w'’+#*.-]+")
_CONNECTORS = {"of", "and", "for", "de", "y", "och", "für", "et", "du"}
_AFFIRM = re.compile(
    r"^\W*(?:yes|yeah|correct|right|true|indeed|exactly|that'?s\s+(?:right|correct|true)"
    r"|sí|ja|oui)\b",
    re.I,
)
_ORDER = re.compile(
    r"\b(?:prerequisites?|requires?|required|requiring|depends?\s+on|builds?\s+on|leads?\s+to"
    r"|followed\s+by|first|firstly|before|after|equivalent|maps?\s+to|same\s+as"
    r"|needed\s+(?:for|to))\b",
    re.I,
)
# A spelled-out figure is a number too ("forty thousand dollars"). "one" is ordinary English.
_SPELLED = re.compile(
    r"\b(?:two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen"
    r"|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety"
    r"|hundred|thousand|million|billion)s?\b",
    re.I,
)
# "like" counts only after a plural noun or a comma ("tools like X"), not in "I'd like to help".
# "needs X and Y" lists claims about X and Y; "you need to pick one" does not.
_EXAMPLES = re.compile(
    r"(?:\b(?:such\s+as|including|includes?|e\.g\.|for\s+example)\b|(?:\b\w+s|,)\s+like\b"
    r"|\bneeds\b(?!\s+to\b)(?=[^.;!?\n]*(?:,|\band\b)))"
    r"\s*[:,]?\s*([^.;!?\n]+)",
    re.I,
)
_ITEM_SPLIT = re.compile(r",|\band\b|\bor\b", re.I)
_ARTICLE = re.compile(r"^(?:an?|the)\s+")
_STARTERS = frozenset(
    """the this that these those it its a an you your i we here there they he she in on at for
    with as to if when while both each some most many all no not also however and but or so then
    note one what which who how why where sure great of from by let please hi hello hey thanks
    welcome name ask try pick type say want would could can should will is are was be do does did
    has have had my our their his her like such currently based according below above every
    el la los las un una unos unas este esta estos para con hola en det den de ett jag du vi der
    die das ein eine ich sie le les une il elle o os um uma""".split()
)


def _norm(text: str) -> str:
    """NFKC, casefold, no markup or quotes; punctuation other than `+#*.%-` is a space."""
    text = unicodedata.normalize("NFKC", text).casefold()
    text = re.sub(r"\*\*|[`\"“”'’]", "", text)
    text = re.sub(r"[^\w+#*.%\-]+|_", " ", text)
    # a sentence-final "." or "-" is not part of a word
    return " ".join(w.strip(".-") or w for w in text.split())


def _stem(text: str) -> str:
    """Plural-tolerant form: "data analysts" matches "data analyst" (both sides are stemmed)."""
    return " ".join(w[:-1] if len(w) > 3 and w.endswith("s") else w for w in text.split())


def _numbers(text: str) -> set[str]:
    return {m.replace(",", "") if "," in m and "." not in m else m for m in _NUMBER.findall(text)}


def _holds(candidate: str, facts: str) -> bool:
    """The candidate occurs in the facts on word boundaries (both stemmed)."""
    want = _stem(_ARTICLE.sub("", _norm(candidate)))
    return not want or f" {want} " in f" {facts} "


def _runs(sentence: str) -> list[str]:
    """Runs of capitalised words (connectors allowed inside), leading starters removed."""
    runs: list[list[str]] = []
    current: list[str] = []
    pending: list[str] = []
    for raw in _WORD.findall(sentence):
        word = raw.strip(".,;:!?)(")
        if not word:
            continue
        if word[0].isupper():
            current.extend([*pending, word] if current else [word])
            pending = []
        elif current and word.casefold() in _CONNECTORS:
            pending.append(word)
        else:
            if current:
                runs.append(current)
            current, pending = [], []
    if current:
        runs.append(current)
    cleaned: list[str] = []
    for run in runs:
        while run and re.split(r"['’]", run[0].casefold())[0] in _STARTERS:
            run = run[1:]
        if run:
            cleaned.append(" ".join(run))
    return cleaned


def _sentence_ok(sentence: str, facts: str, numbers: set[str], allowed: str) -> bool:
    if _AFFIRM.match(sentence):
        return False
    if not _numbers(sentence) <= numbers:
        return False
    plain = f" {_norm(allowed)} "
    if any(f" {_norm(m.group(0))} " not in plain for m in _SPELLED.finditer(sentence)):
        return False
    if any(not _holds(next(g for g in m.groups() if g), facts) for m in _SPANS.finditer(sentence)):
        return False
    if any(not _holds(run, facts) for run in _runs(sentence)):
        return False
    for m in _EXAMPLES.finditer(sentence):
        items = (_ARTICLE.sub("", i.strip()) for i in _ITEM_SPLIT.split(m.group(1)))
        if any(item and not _holds(item, facts) for item in items):
            return False
    return all(f" {_norm(m.group(0))} " in plain for m in _ORDER.finditer(sentence))


def grounded(text: str, allowed: str) -> str:
    """The sentences of model text whose every checkable fact occurs in `allowed`.

    Fail-closed: an unknown candidate drops its sentence; any exception returns "".
    """
    try:
        facts = _stem(_norm(allowed))
        numbers = _numbers(allowed)
        sentences = [s.strip() for s in _SPLIT.split(text) if s.strip()]
        kept = [s for s in sentences if _sentence_ok(s, facts, numbers, allowed)]
        # Nothing dropped: the text as written (line breaks and all).
        return text.strip() if len(kept) == len(sentences) else " ".join(kept)
    except Exception:  # noqa: BLE001 - fail closed: the caller shows its code fallback
        return ""
