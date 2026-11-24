"""Official cross-taxonomy links, from codes the sources themselves publish.

Stopgap: the suite contract says nothing cross-suite lives anywhere but
TA-taxonomies ``crosswalks/``. Until that is filled, these two rules live here.
No inference, no label matching: a link exists only when an official code says so.

- BLS <-> O*NET (SOC): ``bls:occupation:15-1252`` <-> ``onet:occupation:15-1252.00``
  (the O*NET code's first 7 characters are the SOC code), or a BLS node's
  ``onet_soc_codes`` property.
- JobTech -> ESCO: the JobTech node's ``esco_exact`` / ``esco_close`` URIs, which
  name an ESCO node ``esco:<kind>:<uuid>``.

ESCO <-> O*NET and SFIA have no official codes, so they are never linked.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass

# Node properties the rules read; the walker keeps only these.
PROP_KEYS = ("esco_exact", "esco_close", "onet_soc_codes")

_ONET_ID = re.compile(r"^onet:occupation:(\d\d-\d{4})\.\d\d$")
_BLS_ID = re.compile(r"^bls:occupation:(\d\d-\d{4})$")
_ESCO_URI = re.compile(r"^http://data\.europa\.eu/esco/([a-z-]+)/([0-9a-f-]{36})$")
_STRENGTH_ORDER = {"exact": 0, "close": 1, "soc": 2}


@dataclass(frozen=True)
class CrosswalkLink:
    from_id: str
    to_id: str
    strength: str  # exact | close | soc
    source: str  # crosswalk:bls-onet-soc | crosswalk:jobtech-esco

    def to_dict(self) -> dict[str, object]:
        return {
            "from_id": self.from_id,
            "to_id": self.to_id,
            "relation": "same_as",
            "graph_edge": "CROSSWALK",
            "strength": self.strength,
            "source": self.source,
        }


def suite_of(node_id: str) -> str:
    return node_id.split(":", 1)[0]


def crosswalk_props(props: Mapping[str, object]) -> dict[str, object]:
    """Only the non-empty properties the rules read."""
    return {k: props[k] for k in PROP_KEYS if props.get(k)}


def links_from(node_id: str, props: Mapping[str, object] | None = None) -> list[CrosswalkLink]:
    """Every link this one node implies; the other end may not be in any map."""
    props = props or {}
    found: list[CrosswalkLink] = []
    bls_onet = "crosswalk:bls-onet-soc"
    if m := _ONET_ID.match(node_id):
        found.append(CrosswalkLink(f"bls:occupation:{m[1]}", node_id, "soc", bls_onet))
    if _BLS_ID.match(node_id):
        codes = props.get("onet_soc_codes")
        for code in codes if isinstance(codes, list) else []:
            if isinstance(code, str) and _ONET_ID.match(f"onet:occupation:{code}"):
                found.append(CrosswalkLink(node_id, f"onet:occupation:{code}", "soc", bls_onet))
    if node_id.startswith("jobtech:"):
        for key, strength in (("esco_exact", "exact"), ("esco_close", "close")):
            uris = props.get(key)
            for uri in uris if isinstance(uris, list) else []:
                if isinstance(uri, str) and (m := _ESCO_URI.match(uri)):
                    found.append(
                        CrosswalkLink(
                            node_id, f"esco:{m[1]}:{m[2]}", strength, "crosswalk:jobtech-esco"
                        )
                    )
    return found


def crosswalk_links(props_by_id: Mapping[str, Mapping[str, object]]) -> list[CrosswalkLink]:
    """Links between nodes that are all in ``props_by_id`` (id -> its crosswalk props)."""
    unique = {
        (link.from_id, link.to_id): link
        for node_id, props in props_by_id.items()
        for link in links_from(node_id, props)
        if link.from_id in props_by_id and link.to_id in props_by_id
    }
    return sorted(
        unique.values(), key=lambda x: (x.source, _STRENGTH_ORDER[x.strength], x.from_id, x.to_id)
    )


__all__ = ["CrosswalkLink", "crosswalk_links", "crosswalk_props", "links_from", "suite_of"]
