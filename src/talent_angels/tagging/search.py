"""Per unit x suite search: query building, kind routing, merge, re-rank.

Runs sequentially per suite (embedding search is not thread-safe — see
`assistant/turn_graph.py`'s own sequential-suite comment). Never raises: a
broken suite or an unsupported kind both come back as a typed `SearchOutcome`,
never an exception, so `runner.py` can keep going unit by unit.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from typing import Any

from talent_angels.skills.locate.rank import lexical_rank
from talent_angels.skills.locate.resolve import node_ref
from talent_angels.tagging.models import CandidateRef, QueryLog, SearchOutcome, TagUnit

#: A leading numeric code ("2143", "12.34-5 -") — ESCO's exact-code lookup can
#: misread a client's own numbering (SINCO "2143") as an ISCO code.
_CODE_PREFIX_RE = re.compile(r"^\d[\d.\-]*\s*[-:]?\s*")
#: What remains after stripping the prefix must have a real word in it; a bare
#: number/code ("2143", "12.3-4") is not a query worth sending to a suite.
_CODE_LIKE_RE = re.compile(r"^[\d.\-]*$")
_MAX_QUERY_LEN = 200


def queries_for(
    unit: TagUnit,
    labels: Sequence[str] = (),
    description: str | None = None,
) -> list[str]:
    """Candidate search strings for `unit`: rewritten labels (PR3) or seeds now.

    With `labels` (PR3's rewrite step) the description is appended as one more
    query; without them, falls back to the profile's `seed_phrases` (e.g. an
    "Associated Occupations" row split on `;`) or, failing that, the unit's
    own cleaned text.
    """
    if labels:
        raw = [*labels, *([description] if description else [])]
    else:
        raw = list(unit.seed_phrases) if unit.seed_phrases else [unit.clean_text]

    seen: set[str] = set()
    queries: list[str] = []
    for item in raw:
        text = _CODE_PREFIX_RE.sub("", item, count=1).strip()
        if not text or _CODE_LIKE_RE.fullmatch(text):
            continue
        text = text[:_MAX_QUERY_LEN]
        key = text.casefold()
        if key in seen:
            continue
        seen.add(key)
        queries.append(text)
    return queries


class Router:
    """Caches the suite kinds to search per (suite, unit kind).

    `unknown_kind:<k>` in a probe's warnings means this suite has no such
    concept at all — distinct from `not_found`, which still means the kind is
    supported. Every supported kind is kept, not just the first: O*NET's own
    "knowledge" list is 33 broad areas, so a knowledge row must also reach its
    skills and tasks.
    """

    def __init__(self) -> None:
        self._cache: dict[tuple[str, str], list[str]] = {}

    def route(
        self,
        suite_name: str,
        search: Callable[[str, str | None], Any],
        unit_kind: str,
        kinds: list[str],
        probe_query: str,
    ) -> list[str]:
        """Supported kinds in profile order; empty means `n/a` for this suite."""
        key = (suite_name, unit_kind)
        if key in self._cache:
            return list(self._cache[key])
        routed: list[str] = []
        for kind in kinds:
            # A suite that fails here raises to the caller, which records the
            # unit as `unavailable` — an outage must never be cached as "n/a".
            result = search(probe_query, kind)
            warnings = list(getattr(result, "warnings", None) or [])
            if any(w.startswith("unknown_kind:") for w in warnings):
                continue
            routed.append(kind)
        self._cache[key] = routed
        return list(routed)


def _keep_by_routed_kind(ref_kind: str, routed_kind: str) -> bool:
    has_occupation = "occupation" in ref_kind.casefold()
    return has_occupation if routed_kind.casefold() == "occupation" else not has_occupation


def search_unit(
    search: Callable[[str, str | None], Any],
    suite_name: str,
    unit: TagUnit,
    queries: list[str],
    routed_kind: str | Sequence[str],
    *,
    top: int = 12,
) -> SearchOutcome:
    """Run every query x routed kind against one suite, merge by node id, keep `top`."""
    kinds = [routed_kind] if isinstance(routed_kind, str) else list(routed_kind)
    try:
        query_logs: list[QueryLog] = []
        merged: dict[str, dict[str, Any]] = {}

        for query, kind in ((q, k) for q in queries for k in kinds):
            result = search(query, kind)
            warnings = list(getattr(result, "warnings", None) or [])
            candidates = list(getattr(result, "candidates", None) or [])
            top_method = candidates[0].method if candidates else None
            query_logs.append(
                QueryLog(
                    q=query,
                    kind=kind,
                    n=len(candidates),
                    top_method=top_method,
                    warnings=warnings,
                )
            )

            refs = [
                (node_ref(candidate.node, suite_name), candidate.method) for candidate in candidates
            ]
            refs = [(ref, method) for ref, method in refs if _keep_by_routed_kind(ref.kind, kind)]
            if top_method is not None and top_method != "hybrid_rrf":
                refs = sorted(refs, key=lambda item: lexical_rank(query, item[0]))

            alias_warned = any(w.startswith("alias_unconfirmed") for w in warnings)
            for rank, (ref, method) in enumerate(refs, start=1):
                entry = merged.get(ref.id)
                if entry is None:
                    entry = {
                        "ref": ref,
                        "best_rank": rank,
                        "method": method,
                        "hits": set(),
                        "queries": [],
                        "alias_unconfirmed": False,
                    }
                    merged[ref.id] = entry
                elif rank < entry["best_rank"]:
                    entry["best_rank"] = rank
                    entry["method"] = method
                if rank <= 5:
                    entry["hits"].add(query)
                if query not in entry["queries"]:
                    entry["queries"].append(query)
                if alias_warned:
                    entry["alias_unconfirmed"] = True

        candidates_out = [
            CandidateRef(
                suite=suite_name,
                id=entry["ref"].id,
                source_id=entry["ref"].source_id,
                kind=entry["ref"].kind,
                label=entry["ref"].pref_label,
                method=entry["method"],
                best_rank=entry["best_rank"],
                hits=len(entry["hits"]),
                queries=list(entry["queries"]),
                alias_unconfirmed=bool(entry["alias_unconfirmed"]),
            )
            for entry in merged.values()
        ]
        candidates_out.sort(
            key=lambda candidate: (candidate.best_rank, -candidate.hits, candidate.label)
        )
        return SearchOutcome(
            suite=suite_name,
            status="ok",
            routed_kind="+".join(kinds),
            queries=query_logs,
            candidates=candidates_out[:top],
        )
    except Exception as exc:  # noqa: BLE001 - a broken suite must not crash the run
        return SearchOutcome(
            suite=suite_name,
            status="unavailable",
            routed_kind="+".join(kinds),
            queries=[],
            candidates=[],
            error=type(exc).__name__,
        )
