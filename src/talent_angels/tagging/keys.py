"""Stable identity for a tag unit, and per-attempt fingerprints for the ledger.

A unit's identity must survive a Neo4j re-export that renumbers every
`identity` — so it is derived from the document key + context path + text,
never from `row_id`.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from importlib import metadata

from talent_angels.tagging.models import TagUnit, Versions

#: `tag-v1` policy versions (verdict.py, PR3); bump when the rule changes.
PIPELINE_VERSION = "tag-1"
POLICY_VERSION = "tag-v1"
PROMPTS_VERSION = "rw-1/ch-1"

_SEP = "\x1f"


def _digest(parts: Sequence[str]) -> str:
    return hashlib.sha256(_SEP.join(parts).encode("utf-8")).hexdigest()


def unit_key(doc_key: str, path_texts: Sequence[str], text: str, occurrence: int) -> str:
    """Stable id for one row: survives Neo4j identity renumbering on re-export."""
    parts = [doc_key, *path_texts, text, str(occurrence)]
    return f"k_{_digest(parts)[:16]}"


def fingerprint(
    unit: TagUnit,
    suite: str,
    versions: Versions,
    *,
    hint: str = "",
    attempt: int = 0,
) -> str:
    """Ledger key: changes whenever anything that could change the answer does."""
    parts = [
        unit.unit_key,
        unit.clean_text,
        *unit.context_path,
        suite,
        versions.pipeline,
        versions.policy,
        versions.prompts,
        versions.ta_taxonomies,
        versions.model,
        hint,
        str(attempt),
    ]
    return f"f_{_digest(parts)[:24]}"


def taxonomies_commit() -> str:
    """The `ta-taxonomies` commit this runtime is pinned to, or `"unknown"`."""
    try:
        text = metadata.distribution("ta-taxonomies").read_text("direct_url.json")
        if text is None:
            return "unknown"
        data = json.loads(text)
        commit = data.get("vcs_info", {}).get("commit_id")
    except (metadata.PackageNotFoundError, json.JSONDecodeError, OSError, AttributeError):
        return "unknown"
    if isinstance(commit, str) and commit:
        return commit[:12]
    return "unknown"


def versions(model: str) -> Versions:
    return Versions(
        pipeline=PIPELINE_VERSION,
        policy=POLICY_VERSION,
        prompts=PROMPTS_VERSION,
        ta_taxonomies=taxonomies_commit(),
        model=model,
    )
