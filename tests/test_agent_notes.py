"""Tests for talent_angels.memory.agent_notes — bounded, deduplicated MEMORY.md."""

from unittest.mock import patch


def test_append_note_dedupes_exact(tmp_path):
    from talent_angels.memory import agent_notes as notes

    with patch.object(notes, "memory_md", lambda: tmp_path / "MEMORY.md"):
        notes.append_note("entry one")
        notes.append_note("entry one")
        content = (tmp_path / "MEMORY.md").read_text()
        assert content.count("- entry one") == 1


def test_append_note_evicts_oldest_when_over_cap(tmp_path):
    from talent_angels.memory import agent_notes as notes

    with (
        patch.object(notes, "memory_md", lambda: tmp_path / "MEMORY.md"),
        patch.object(notes, "MEMORY_MAX_CHARS", 40),
    ):
        notes.append_note("a" * 25)
        notes.append_note("b" * 25)
        content = (tmp_path / "MEMORY.md").read_text()
        assert len(content) <= 40
        # oldest (a) evicted first, newest (b) retained
        assert "- " + "a" * 25 not in content
        assert "- " + "b" * 25 in content


def test_retired_developer_notes_stay_out_of_prompts(memory_home):
    from talent_angels.memory import agent_notes as notes

    memory_home.memory_md.write_text(
        "- Pathfind not yet implemented — redirect gracefully\n- prefers short answers\n"
    )

    prefix = notes.notes_prefix()

    assert "Pathfind not yet implemented" not in prefix
    assert "- prefers short answers" in prefix


def test_only_retired_notes_means_no_block(memory_home):
    from talent_angels.memory import agent_notes as notes

    memory_home.memory_md.write_text("- Pathfind not yet implemented — redirect gracefully\n")

    assert notes.notes_prefix() == ""


def test_starting_the_tui_does_not_seed_developer_notes(memory_home):
    from talent_angels.tui import app

    assert not hasattr(app, "_seed_memory_if_new")
    assert not memory_home.memory_md.exists()
