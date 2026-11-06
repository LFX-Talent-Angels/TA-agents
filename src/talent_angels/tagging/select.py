"""Turn a `Document`'s rows into taggable `TagUnit`s (PILOT-PLAN.md `select.py`).

Every non-root row ends up in exactly one of `units` / `skipped`, in document
order — tests lean on that invariant to catch a silently dropped row. A row's
kind comes from its own text (a row rule, or the nearest ancestor/self whose
*text* matches a kind heading) — never its Neo4j label — so the same profile
works whatever a given export happens to label its nodes.
"""

from __future__ import annotations

from typing import Literal

from talent_angels.tagging.keys import unit_key
from talent_angels.tagging.models import Document, SkippedRow, SourceRow, TagUnit
from talent_angels.tagging.profile import Profile


def document_context(doc: Document, profile: Profile) -> list[str]:
    """Cleaned text of every row matching `profile.doc_context`, in order."""
    return [
        profile.clean(row.text)
        for row in doc.rows
        if row.parent_id is not None and profile.matches_doc_context(row.text)
    ]


def select_units(
    doc: Document,
    profile: Profile,
    *,
    rows: set[str] | None = None,
    limit: int | None = None,
) -> tuple[list[TagUnit], list[SkippedRow]]:
    by_id = doc.by_id()
    children = doc.children()
    root_id = next((row.row_id for row in doc.rows if row.parent_id is None), None)

    ancestor_kind: dict[str, str | None] = {}

    def nearest_kind(row: SourceRow) -> str | None:
        """The kind of the nearest ancestor (not `row` itself) whose text matches."""
        if row.row_id in ancestor_kind:
            return ancestor_kind[row.row_id]
        result: str | None = None
        if row.parent_id is not None:
            parent = by_id.get(row.parent_id)
            if parent is not None:
                result = profile.kind_of_heading(parent.text) or nearest_kind(parent)
        ancestor_kind[row.row_id] = result
        return result

    def context_path_for(row: SourceRow) -> list[str]:
        """Ancestor texts from top (excluding the root) down to the parent."""
        path: list[str] = []
        current = by_id.get(row.parent_id) if row.parent_id is not None else None
        while current is not None and current.row_id != root_id:
            path.append(current.text.strip())
            current = by_id.get(current.parent_id) if current.parent_id is not None else None
        path.reverse()
        return path

    occurrence_seen: dict[tuple[str | None, str], int] = {}
    units: list[TagUnit] = []
    skipped: list[SkippedRow] = []

    for row in doc.rows:
        if row.parent_id is None:
            skipped.append(SkippedRow(row_id=row.row_id, reason="root"))
            continue
        parent_id = row.parent_id

        heading_kind = profile.kind_of_heading(row.text)
        if heading_kind is not None:
            skipped.append(SkippedRow(row_id=row.row_id, reason="heading", kind=heading_kind))
            continue

        seed_phrases: list[str] = []
        role: Literal["leaf", "heading", "row_rule"]
        rule = profile.rule_for(row.text)
        if rule is not None:
            kind = rule.kind
            role = "row_rule"
            if rule.split:
                _, _, after = row.text.partition(":")
                seen_phrases: set[str] = set()
                for part in after.split(rule.split):
                    cleaned_part = profile.clean(part)
                    if cleaned_part and cleaned_part not in seen_phrases:
                        seen_phrases.add(cleaned_part)
                        seed_phrases.append(cleaned_part)
        else:
            ancestor = nearest_kind(row)
            if ancestor is None:
                skipped.append(SkippedRow(row_id=row.row_id, reason="metadata"))
                continue
            kind = ancestor
            role = "heading" if children.get(row.row_id) else "leaf"
            tagged_at_this_depth = profile.depth == "all" or (
                (profile.depth == "leaves" and role == "leaf")
                or (profile.depth == "headings" and role == "heading")
            )
            if not tagged_at_this_depth:
                skipped.append(SkippedRow(row_id=row.row_id, reason="heading", kind=kind))
                continue

        if kind not in profile.include_kinds:
            skipped.append(SkippedRow(row_id=row.row_id, reason="excluded_kind", kind=kind))
            continue

        clean_text = profile.clean(row.text)
        if not clean_text:
            skipped.append(SkippedRow(row_id=row.row_id, reason="metadata", kind=kind))
            continue

        context_path = context_path_for(row)
        occ_key = (parent_id, row.text.strip())
        occurrence = occurrence_seen.get(occ_key, 0)
        occurrence_seen[occ_key] = occurrence + 1

        units.append(
            TagUnit(
                unit_key=unit_key(doc.key, context_path, row.text.strip(), occurrence),
                doc_key=doc.key,
                row_id=row.row_id,
                kind=kind,
                role=role,
                text=row.text,
                clean_text=clean_text,
                context_path=context_path,
                group_key=parent_id,
                seed_phrases=seed_phrases,
            )
        )

    # `rows` / `limit` narrow the units to actually tag, applied last so they
    # never affect unit_key/occurrence computation above.
    kept: list[TagUnit] = []
    for unit in units:
        if rows is not None and unit.row_id not in rows:
            skipped.append(SkippedRow(row_id=unit.row_id, reason="filtered", kind=unit.kind))
            continue
        kept.append(unit)
    if limit is not None and len(kept) > limit:
        for unit in kept[limit:]:
            skipped.append(SkippedRow(row_id=unit.row_id, reason="filtered", kind=unit.kind))
        kept = kept[:limit]

    order_of = {row.row_id: row.order for row in doc.rows}
    skipped.sort(key=lambda item: order_of[item.row_id])
    return kept, skipped
