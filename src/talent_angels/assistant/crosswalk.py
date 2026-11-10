"""Reach a suite whose labels are in another language through its own ESCO matches.

JobTech's labels are Swedish, so "baker" finds nothing there by words. But
JobTech publishes, for each of its records, which ESCO records match it
(exact, close, broad, narrow), and ESCO is searchable in English. So when
JobTech finds nothing, the same words are looked up in ESCO, and JobTech's
own matches to what ESCO found are offered instead, saying so.

Nothing is equated by us: the link is JobTech's published match, the ESCO
record stays an ESCO record, and the answer names both. A suite that offers
no ``search_by_esco`` is never bridged.
"""

from __future__ import annotations

import re
from typing import Any

from talent_angels.contracts import AgentResult, EvidencePointer
from talent_angels.skills.locate import locate
from talent_angels.suites import SuiteRegistry

#: Suites whose records carry their own ESCO matches.
BRIDGED_SUITES = frozenset({"jobtech"})
ESCO = "esco"
#: How many of ESCO's records to follow. Only a clear (not ambiguous) ESCO
#: answer is followed, so in practice this is its one title.
MAX_ESCO_RECORDS = 3
#: Tagging follows ESCO's word matches (the model and the reviewer then choose).
MIN_CONFIDENCE = 0.70
#: Chat follows only an ESCO answer that *is* the words (exact title, alias or
#: casefold, 0.80+). A part-of-a-title hit ("gift" in "teacher of talented and
#: gifted students") must not become a confident Swedish record.
CHAT_MIN_CONFIDENCE = 0.80
#: A suite's own hit at or above this is clear; below, its words only matched part
#: of a title (contains 0.70, meaning 0.75) and a rescue may do better.
CLEAR_CONFIDENCE = 0.85
WARNING_PREFIX = "via_esco:"


def wants_bridge(result: AgentResult) -> bool:
    """The suite found nothing for the words (the bridge's original case)."""
    return (
        result.suite in BRIDGED_SUITES
        and result.capability in ("locate", "connect")
        and not result.nodes
        and "not_found" in result.warnings
    )


def needs_rescue(result: AgentResult) -> bool:
    """A Swedish-labelled suite without a clear answer: nothing, or only part-of-word hits."""
    if result.suite not in BRIDGED_SUITES or result.capability not in ("locate", "connect"):
        return False
    if wants_bridge(result):
        return True
    return bool(result.nodes) and (result.confidence or 0.0) < CLEAR_CONFIDENCE


class _ByEsco:
    """A Locate-shaped view of ``search_by_esco`` so ``locate`` can convert it."""

    def __init__(self, suite: Any, uris: list[str], kind: str) -> None:
        self._suite = suite
        self._uris = uris
        self._kind = kind

    def search_nodes(self, text: str, kind: str | None = None) -> Any:
        return self._suite.search_by_esco(self._uris, kind=self._kind)


def bridge_via_esco(
    registry: SuiteRegistry,
    result: AgentResult,
    text: str,
    *,
    kind: str | None = None,
) -> AgentResult | None:
    """JobTech's own matches to what ESCO finds for ``text``, or None."""
    if not wants_bridge(result) or ESCO not in registry.available or not text.strip():
        return None
    kind = kind if kind in ("occupation", "skill") else "occupation"
    with registry.open(ESCO) as esco_runtime:
        found = locate(esco_runtime.suite, ESCO, text, kind=kind)
    if not found.nodes or (found.confidence or 0.0) < CHAT_MIN_CONFIDENCE:
        return None
    if "ambiguous" in found.warnings:
        # ESCO itself needs the user to choose. Following its first records
        # would hand JobTech a confident answer ESCO never had (and a unique
        # bridged hit binds the occupation and saves a goal).
        return None
    records = found.nodes[:MAX_ESCO_RECORDS]
    uris = [node.source_id for node in records if node.source_id.startswith("http")]
    if not uris:
        return None
    with registry.open(result.suite) as runtime:
        if not hasattr(runtime.suite, "search_by_esco"):
            return None
        bridged = locate(_ByEsco(runtime.suite, uris, kind), result.suite, text, kind=kind)
    if not bridged.nodes:
        return None
    via = [f"{WARNING_PREFIX}{node.pref_label}" for node in records]
    return bridged.model_copy(
        update={
            "warnings": [*bridged.warnings, *via],
            "evidence": [
                *bridged.evidence,
                *(EvidencePointer(suite=ESCO, pointer=p.pointer) for p in found.evidence),
            ],
        }
    )


def esco_records_named(result: AgentResult) -> list[str]:
    """The ESCO titles a bridged result was reached through (for the card)."""
    return [w.removeprefix(WARNING_PREFIX) for w in result.warnings if w.startswith(WARNING_PREFIX)]


class EscoBridgedSuite:
    """A suite whose ``search_nodes`` falls back to its own ESCO matches.

    For batch tagging: rows are English (the rewrite step writes English
    phrases), JobTech labels are Swedish, so JobTech found nothing for almost
    every row. When the suite's own words find nothing, the same words go to
    ESCO and the suite's published matches to ESCO's top records come back,
    each named by its match (``esco_exact_match`` …). Tagging lets the model's
    choose step and the reviewer pick among them, so a list from ESCO is
    followed too (``follow_ambiguous``); chat does not (``bridge_via_esco``).
    Everything else is delegated unchanged.
    """

    def __init__(self, suite: Any, esco: Any, *, follow_ambiguous: bool = True) -> None:
        self._suite = suite
        self._esco = esco
        self._follow_ambiguous = follow_ambiguous

    def __getattr__(self, name: str) -> Any:
        return getattr(self._suite, name)

    @property
    def suite_schema(self) -> Any:
        return self._suite.suite_schema

    def search_nodes(self, text: str, kind: str | None = None) -> Any:
        own = self._suite.search_nodes(text, kind=kind)
        warnings = list(getattr(own, "warnings", None) or [])
        if getattr(own, "candidates", None) or any(w.startswith("unknown_kind:") for w in warnings):
            return own
        bridge_kind = kind or "occupation"
        if bridge_kind not in ("occupation", "skill") or not text.strip():
            return own
        try:
            found = self._esco.search_nodes(text, kind=bridge_kind)
        except Exception:  # noqa: BLE001 — the bridge is a bonus, never a failure
            return own
        candidates = list(getattr(found, "candidates", None) or [])
        if not candidates or candidates[0].confidence < MIN_CONFIDENCE:
            return own
        if "ambiguous" in (getattr(found, "warnings", None) or []) and not self._follow_ambiguous:
            return own
        uris = [
            str(c.node.source_id)
            for c in candidates[:MAX_ESCO_RECORDS]
            if str(c.node.source_id).startswith("http")
        ]
        if not uris:
            return own
        bridged = self._suite.search_by_esco(uris, kind=bridge_kind)
        return bridged if getattr(bridged, "candidates", None) else own


def with_esco_bridge(name: str, suite: Any, esco: Any | None) -> Any:
    """``suite`` wrapped in the ESCO bridge when it is one that offers it."""
    if esco is None or name not in BRIDGED_SUITES or not hasattr(suite, "search_by_esco"):
        return suite
    return EscoBridgedSuite(suite, esco)


_WORD = re.compile(r"[^\W\d_]{5,}", re.UNICODE)
#: Words that name a suite or a question, never a job ("arbetsförmedlingen").
_NOT_TITLES = frozenset(
    "jobtech swedish sweden arbetsförmedlingen taxonomy skills skill which vilka "
    "yrken yrke kompetens behöver needs where there about could would should".split()
)


def own_words_hit(
    registry: SuiteRegistry, suite_name: str, question: str, *, kind: str
) -> AgentResult | None:
    """A word of the user's own message that is exactly a title in the suite.

    The planner rewrites subjects into English ("Elevassistent" became
    "elevator operator"), so the suite's own exact Swedish title was never
    searched. Each longer word of the message is tried; only an exact, alias
    or casefold hit (0.85+) counts.
    """
    words = [w for w in _WORD.findall(question) if w.casefold() not in _NOT_TITLES]
    if not words:
        return None
    with registry.open(suite_name) as runtime:
        for word in dict.fromkeys(words):
            found = locate(runtime.suite, suite_name, word, kind=kind)
            if found.nodes and (found.confidence or 0.0) >= CLEAR_CONFIDENCE:
                return found
    return None


def connect_after_rescue(registry: SuiteRegistry, rescued: AgentResult, draft: Any) -> AgentResult:
    """The skills of a rescued record when the question asked for them."""
    if len(rescued.nodes) != 1 or "ambiguous" in rescued.warnings:
        return rescued
    from talent_angels.assistant.llm_plan import connect_request_from_draft
    from talent_angels.skills.connect import ConnectRequest, connect

    center = rescued.nodes[0]
    with registry.open(rescued.suite) as runtime:
        schema = runtime.suite.suite_schema
        request = (
            connect_request_from_draft(draft, skill_rel_types=schema.skill_rel_types)
            if draft is not None
            else None
        ) or ConnectRequest(subject=center.pref_label, rel_types=schema.skill_rel_types)
        connected = connect(
            runtime.suite,
            rescued.suite,
            center,
            request=request,
            confidence=rescued.confidence,
            locate_evidence=rescued.evidence,
            optional_rel_values=schema.optional_rel_values,
        )
    via = [w for w in rescued.warnings if w.startswith(WARNING_PREFIX)]
    return connected.model_copy(update={"warnings": [*connected.warnings, *via]})


def rescue_jobtech(
    registry: SuiteRegistry,
    result: AgentResult,
    *,
    subject: str,
    question: str,
    kind: str | None,
    language: str | None,
    wants_skills: bool,
    draft: Any = None,
) -> AgentResult | None:
    """A better answer for a Swedish-labelled suite, or None to keep its own.

    1. The user's own words, if one is exactly a title there.
    2. Its published ESCO matches, when its own words found nothing, or found
       only part-of-a-title hits for a question not written in Swedish
       ("chef" in English is a cook, not a Swedish *chef*).
    3. The rescued record's skills, when the question asked for them.
    """
    if not needs_rescue(result):
        return None
    kind = kind if kind in ("occupation", "skill") else "occupation"
    rescued = own_words_hit(registry, result.suite, question, kind=kind)
    if rescued is None and (wants_bridge(result) or (language or "").casefold() != "sv"):
        probe = (
            result
            if wants_bridge(result)
            else AgentResult(capability="locate", suite=result.suite, warnings=["not_found"])
        )
        rescued = bridge_via_esco(registry, probe, subject, kind=kind)
    if rescued is None:
        return None
    return connect_after_rescue(registry, rescued, draft) if wants_skills else rescued
