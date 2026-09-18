"""Offline tests for the USER.md / MEMORY.md long-term memory package."""

from __future__ import annotations

from pathlib import Path

import pytest

from talent_angels.memory import (
    MemoryFullError,
    ProfileRef,
    add_note,
    add_rejected,
    confirm_goal,
    confirm_standing,
    erase_all,
    erase_notes,
    erase_profile,
    load_notes,
    load_profile,
    memory_dir,
    remove_note,
    replace_note,
    set_style_notes,
    set_suite_preference,
)
from talent_angels.memory.models import MAX_REJECTED_ENTRIES
from talent_angels.memory.notes import MAX_NOTES


@pytest.fixture(autouse=True)
def _local_memory_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("TA_MEMORY_DIR", str(tmp_path / "memory"))


def test_load_profile_is_empty_when_nothing_confirmed() -> None:
    profile = load_profile()

    assert profile.is_empty()
    assert not memory_dir().joinpath("USER.md").exists()


def test_confirm_standing_round_trips_with_id_and_date() -> None:
    confirm_standing("Software Developers", node_id="onet:15-1252.00", since="2026-03-10")

    profile = load_profile()
    assert profile.standing == ProfileRef(
        label="Software Developers", node_id="onet:15-1252.00", suite="onet"
    )
    assert profile.standing_since == "2026-03-10"


def test_confirm_standing_round_trips_label_only_for_custom_taxonomies() -> None:
    confirm_standing("Internal Widget Engineer")

    profile = load_profile()
    assert profile.standing == ProfileRef(
        label="Internal Widget Engineer", node_id=None, suite=None
    )


def test_confirm_goal_does_not_clobber_standing() -> None:
    confirm_standing("Software Developers", node_id="onet:15-1252.00")
    confirm_goal("Data Scientists", node_id="onet:15-2051.00")

    profile = load_profile()
    assert profile.standing is not None
    assert profile.goal == ProfileRef(
        label="Data Scientists", node_id="onet:15-2051.00", suite="onet"
    )


def test_rejected_entries_are_capped_and_keep_the_most_recent() -> None:
    for i in range(MAX_REJECTED_ENTRIES + 3):
        add_rejected(f"Candidate {i}", node_id=f"esco:occupation:{i}")

    profile = load_profile()
    assert len(profile.rejected) == MAX_REJECTED_ENTRIES
    assert profile.rejected[-1].label == f"Candidate {MAX_REJECTED_ENTRIES + 2}"
    assert profile.rejected[0].label == "Candidate 3"


def test_suite_preference_and_style_notes_round_trip() -> None:
    set_suite_preference("onet")
    set_style_notes("prefers bullet answers")

    profile = load_profile()
    assert profile.suite_preference == "onet"
    assert profile.style_notes == "prefers bullet answers"


def test_erase_profile_removes_the_file_and_reports_whether_it_existed() -> None:
    assert erase_profile() is False

    confirm_standing("Software Developers")
    assert memory_dir().joinpath("USER.md").exists()

    assert erase_profile() is True
    assert not memory_dir().joinpath("USER.md").exists()
    assert load_profile().is_empty()


def test_notes_add_load_replace_remove_round_trip() -> None:
    assert load_notes() == []

    add_note("This user prefers concise answers")
    add_note("O*NET USES_SOFTWARE edges have relation_type=None")
    assert load_notes() == [
        "This user prefers concise answers",
        "O*NET USES_SOFTWARE edges have relation_type=None",
    ]

    assert replace_note("concise", "This user prefers bullet points") is True
    assert load_notes()[0] == "This user prefers bullet points"

    assert remove_note("USES_SOFTWARE") is True
    assert load_notes() == ["This user prefers bullet points"]


def test_notes_replace_and_remove_are_no_ops_when_nothing_matches() -> None:
    add_note("existing note")

    assert replace_note("nothing-like-this", "new text") is False
    assert remove_note("nothing-like-this") is False
    assert load_notes() == ["existing note"]


def test_add_note_raises_at_the_cap_instead_of_silently_dropping() -> None:
    for i in range(MAX_NOTES):
        add_note(f"note {i}")

    with pytest.raises(MemoryFullError):
        add_note("one too many")

    assert len(load_notes()) == MAX_NOTES


def test_erase_notes_removes_the_file_and_reports_whether_it_existed() -> None:
    assert erase_notes() is False

    add_note("something")
    assert erase_notes() is True
    assert load_notes() == []


def test_erase_all_clears_both_files_in_one_call() -> None:
    confirm_standing("Software Developers")
    add_note("something to remember")

    assert erase_all() is True

    assert load_profile().is_empty()
    assert load_notes() == []
    assert erase_all() is False


def test_memory_files_are_not_written_until_something_is_confirmed() -> None:
    load_profile()
    load_notes()

    assert not memory_dir().exists()
