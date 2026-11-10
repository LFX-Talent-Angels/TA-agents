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
#: ESCO's word matches only; a meaning-only ESCO guess is too weak to follow.
MIN_CONFIDENCE = 0.70
WARNING_PREFIX = "via_esco:"


def wants_bridge(result: AgentResult) -> bool:
    return (
        result.suite in BRIDGED_SUITES
        and result.capability == "locate"
        and not result.nodes
        and "not_found" in result.warnings
    )


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
    if not found.nodes or (found.confidence or 0.0) < MIN_CONFIDENCE:
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
