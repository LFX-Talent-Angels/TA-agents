"""Erase-person: one call, total, no partial state left behind."""

from __future__ import annotations

from talent_angels.memory.notes import erase_notes
from talent_angels.memory.profile import erase_profile


def erase_all() -> bool:
    """Delete USER.md and MEMORY.md. Returns True if anything was actually removed."""
    profile_removed = erase_profile()
    notes_removed = erase_notes()
    return profile_removed or notes_removed
