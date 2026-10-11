"""`Profile`: heading/row-rule/cleaner business rules, loaded as data."""

from __future__ import annotations

from pathlib import Path

import pytest

from talent_angels.tagging.profile import load_profile


def test_builtin_competency_standard_profile_loads() -> None:
    profile = load_profile()
    assert profile.name == "competency_standard"
    assert profile.depth == "leaves"
    assert profile.kind_of_heading("PERFORMANCES") == "performance"
    assert profile.kind_of_heading("Desempeños") == "performance"
    assert profile.kind_of_heading("Glossary of Terms") == "definition"
    assert profile.kind_of_heading("not a heading at all") is None


@pytest.mark.parametrize(
    ("text", "expected_kind"),
    [
        ("Associated Occupations: Baker; Pastry Chef", "occupation"),
        ("Occupations Recognized in the Sector but not in SINCO: Baker", "occupation"),
        ("Purpose of the Standard: do a thing", "summary"),
        ("General Description of the Standard: do a thing", "summary"),
        ("Code: 1234", None),
    ],
)
def test_row_rules_match_case_insensitively(text: str, expected_kind: str | None) -> None:
    profile = load_profile()
    rule = profile.rule_for(text)
    assert (rule.kind if rule is not None else None) == expected_kind


def test_cleaners_strip_trailing_punctuation_and_and_but_keep_real_words() -> None:
    profile = load_profile()
    assert profile.clean("Removing debris from surfaces,") == "Removing debris from surfaces"
    assert profile.clean("Setting criteria for output, and") == "Setting criteria for output"
    assert profile.clean("Documenting the final sheet.") == "Documenting the final sheet"
    # "hand" ends in the letters "and" but must survive the trailing-"and" cleaner.
    assert profile.clean("Disposing of waste by hand.") == "Disposing of waste by hand"
    assert profile.clean("Done by hand") == "Done by hand"


def test_cleaners_strip_subject_area_prefix_and_competency_level_suffix() -> None:
    profile = load_profile()
    cleaned = profile.clean(
        "Subject Area: Food safety: allergen control. | Competency Level: Application"
    )
    assert cleaned == "Food safety: allergen control"


def test_cleaners_strip_emergent_situation_and_expected_response_prefixes() -> None:
    profile = load_profile()
    assert profile.clean("Emergent Situation: a fire starts.") == "a fire starts"
    assert (
        profile.clean("Expected Response: call the fire department.") == "call the fire department"
    )


def test_project_tag_yaml_overrides_a_top_level_key(tmp_path: Path) -> None:
    (tmp_path / "tag.yaml").write_text("include_kinds: [performance, product]\n")
    profile = load_profile(project_dir=tmp_path)
    assert profile.include_kinds == ["performance", "product"]
    # Untouched keys keep the built-in default.
    assert profile.kind_of_heading("KNOWLEDGE") == "knowledge"


def test_missing_tag_yaml_is_not_an_error(tmp_path: Path) -> None:
    profile = load_profile(project_dir=tmp_path)
    assert profile.name == "competency_standard"


def test_bad_tag_yaml_raises_value_error(tmp_path: Path) -> None:
    (tmp_path / "tag.yaml").write_text("not: [valid, yaml, : :\n")
    with pytest.raises(ValueError, match="tag.yaml"):
        load_profile(project_dir=tmp_path)


def test_tag_yaml_must_be_a_mapping(tmp_path: Path) -> None:
    (tmp_path / "tag.yaml").write_text("- just\n- a\n- list\n")
    with pytest.raises(ValueError, match="mapping"):
        load_profile(project_dir=tmp_path)


def test_invalid_regex_raises_value_error(tmp_path: Path) -> None:
    (tmp_path / "tag.yaml").write_text("cleaners: ['[unterminated']\n")
    with pytest.raises(ValueError):
        load_profile(project_dir=tmp_path)
