"""Tests for talent_angels.memory.agent_notes — bounded, deduplicated MEMORY.md."""

from unittest.mock import patch


def test_append_note_dedupes_exact(tmp_path):
    from talent_angels.memory import agent_notes as notes

    with patch.object(notes, "MEMORY_MD", tmp_path / "MEMORY.md"):
        notes.append_note("entry one")
        notes.append_note("entry one")
        content = (tmp_path / "MEMORY.md").read_text()
        assert content.count("- entry one") == 1


def test_append_note_evicts_oldest_when_over_cap(tmp_path):
    from talent_angels.memory import agent_notes as notes

    with (
        patch.object(notes, "MEMORY_MD", tmp_path / "MEMORY.md"),
        patch.object(notes, "MEMORY_MAX_CHARS", 40),
    ):
        notes.append_note("a" * 25)
        notes.append_note("b" * 25)
        content = (tmp_path / "MEMORY.md").read_text()
        assert len(content) <= 40
        # oldest (a) evicted first, newest (b) retained
        assert "- " + "a" * 25 not in content
        assert "- " + "b" * 25 in content
