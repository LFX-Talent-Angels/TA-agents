"""`select_units`/`document_context`: rows -> taggable units, by profile rule."""

from __future__ import annotations

from pathlib import Path

from talent_angels.tagging.models import SkippedRow, TagUnit
from talent_angels.tagging.profile import load_profile
from talent_angels.tagging.readers import read_document
from talent_angels.tagging.select import document_context, select_units

FIXTURE = Path(__file__).parent / "fixtures" / "tagging" / "standard_small.json"


def _select(**overrides: object) -> tuple[list[TagUnit], list[SkippedRow]]:
    doc = read_document(FIXTURE)
    profile = load_profile()
    if overrides:
        profile = profile.model_copy(update=overrides)
        profile.model_post_init(None)
    return select_units(doc, profile)


def test_every_non_root_row_appears_exactly_once() -> None:
    doc = read_document(FIXTURE)
    profile = load_profile()
    units, skipped = select_units(doc, profile)

    seen_units = [u.row_id for u in units]
    seen_skipped = [s.row_id for s in skipped]
    assert len(set(seen_units)) == len(seen_units)
    assert len(set(seen_skipped)) == len(seen_skipped)
    assert set(seen_units).isdisjoint(seen_skipped)
    assert set(seen_units) | set(seen_skipped) == {row.row_id for row in doc.rows}


def test_root_row_is_skipped_with_reason_root() -> None:
    doc = read_document(FIXTURE)
    profile = load_profile()
    _, skipped = select_units(doc, profile)
    root_id = next(row.row_id for row in doc.rows if row.parent_id is None)
    root_skip = next(s for s in skipped if s.row_id == root_id)
    assert root_skip.reason == "root"


def test_kind_decided_by_heading_text_not_label() -> None:
    """Section II repeats the Element titles as plain `Item` rows with no kind
    ancestor — they must come out as metadata, not as a `performance`/etc unit,
    even though Section III's real `Element` nodes carry the same text."""
    units, skipped = _select()
    units_by_row = {u.row_id: u for u in units}
    skipped_by_row = {s.row_id: s for s in skipped}

    # Section II's "Element 1 of 2" / "Element 2 of 2" items (ids 11, 12)
    # have no kind ancestor -> metadata, regardless of their `Item` label.
    assert "11" not in units_by_row
    assert skipped_by_row["11"].reason == "metadata"
    assert "12" not in units_by_row
    assert skipped_by_row["12"].reason == "metadata"

    # A leaf performance row is tagged by its ancestor heading's *text*
    # ("PERFORMANCES"), independent of its own `Item`/`Block` label.
    assert units_by_row["17"].kind == "performance"


def test_depth_leaves_only_tags_leaf_rows() -> None:
    units, skipped = _select(depth="leaves")
    skipped_by_row = {s.row_id: s for s in skipped}
    assert all(u.role in ("leaf", "row_rule") for u in units)
    # "1. Cleans the work area:" (16) has children -> heading, skipped at this depth.
    assert skipped_by_row["16"].reason == "heading"
    assert skipped_by_row["16"].kind == "performance"


def test_depth_headings_only_tags_heading_rows() -> None:
    units, _ = _select(depth="headings")
    assert units  # at least the two heading rows (16, 29)
    assert all(u.role in ("heading", "row_rule") for u in units)
    rows = {u.row_id for u in units}
    assert "16" in rows
    assert "17" not in rows  # a leaf, excluded at this depth


def test_depth_all_tags_both_leaves_and_headings() -> None:
    units, _ = _select(depth="all")
    rows = {u.row_id for u in units}
    assert "16" in rows  # heading
    assert "17" in rows  # leaf


def test_occupation_row_rule_splits_seed_phrases() -> None:
    units, _ = _select()
    occupation = next(u for u in units if u.kind == "occupation")
    assert occupation.role == "row_rule"
    assert occupation.seed_phrases == ["Baker", "Pastry Chef", "Bakery Assistant"]


def test_summary_row_rule() -> None:
    units, _ = _select()
    summary = next(u for u in units if u.kind == "summary")
    assert summary.role == "row_rule"
    assert summary.clean_text.startswith("Purpose of the Standard")


def test_excluded_kind_via_project_tag_yaml(tmp_path: Path) -> None:
    (tmp_path / "tag.yaml").write_text(
        "include_kinds: [performance, product, occupation, summary]\n"
    )
    doc = read_document(FIXTURE)
    profile = load_profile(project_dir=tmp_path)
    units, skipped = select_units(doc, profile)

    assert all(u.kind in {"performance", "product", "occupation", "summary"} for u in units)
    excluded = [s for s in skipped if s.reason == "excluded_kind"]
    assert excluded
    assert {s.kind for s in excluded} <= {"knowledge", "attitude", "emergent", "definition"}


def test_knowledge_leaf_cleaned_of_subject_area_and_competency_level() -> None:
    units, _ = _select()
    knowledge = next(u for u in units if u.row_id == "25")
    assert knowledge.kind == "knowledge"
    assert knowledge.clean_text == (
        "Food safety: Cross-contamination prevention, allergen control, and "
        "temperature danger zones"
    )


def test_duplicate_sibling_text_becomes_two_units() -> None:
    units, _ = _select()
    duplicates = [u for u in units if u.row_id in ("22", "23")]
    assert len(duplicates) == 2
    assert duplicates[0].clean_text == duplicates[1].clean_text == "Clean and sanitized work area"
    assert duplicates[0].unit_key != duplicates[1].unit_key


def test_context_path_excludes_document_root() -> None:
    units, _ = _select()
    occupation = next(u for u in units if u.kind == "occupation")
    assert occupation.context_path == [
        "Section I: General Information",
        "Occupations Related to this Standard",
    ]
    assert "Bakery Hygiene and Food Safety Standard.json" not in occupation.context_path


def test_rows_filter_narrows_and_marks_the_rest_filtered() -> None:
    units, skipped = select_units(read_document(FIXTURE), load_profile(), rows={"17"})
    assert [u.row_id for u in units] == ["17"]
    filtered = [s for s in skipped if s.reason == "filtered"]
    assert len(filtered) >= 1
    assert all(s.row_id != "17" for s in filtered)


def test_limit_narrows_and_marks_the_rest_filtered() -> None:
    all_units, _ = _select()
    units, skipped = select_units(read_document(FIXTURE), load_profile(), limit=3)
    assert len(units) == 3
    assert units == all_units[:3]
    filtered = [s for s in skipped if s.reason == "filtered"]
    assert len(filtered) == len(all_units) - 3


def test_document_context_returns_title_and_purpose() -> None:
    doc = read_document(FIXTURE)
    profile = load_profile()
    context = document_context(doc, profile)
    assert context == [
        "Title: Bakery Hygiene and Food Safety Standard",
        "Purpose of the Standard: To certify competency in bakery hygiene and "
        "food safety practices",
    ]
