"""Long-term personal memory: USER.md profile + MEMORY.md agent notes.

Explicit-confirm writes only; erase is total. Session-scoped state (the
current conversation's bindings) lives in `talent_angels.session` instead.
"""

from talent_angels.memory.erase import erase_all
from talent_angels.memory.models import MAX_REJECTED_ENTRIES, ProfileRef, UserProfile
from talent_angels.memory.notes import (
    MAX_NOTES,
    MemoryFullError,
    add_note,
    erase_notes,
    load_notes,
    remove_note,
    replace_note,
)
from talent_angels.memory.profile import (
    add_rejected,
    confirm_goal,
    confirm_standing,
    erase_profile,
    load_profile,
    memory_dir,
    set_style_notes,
    set_suite_preference,
)

__all__ = [
    "MAX_NOTES",
    "MAX_REJECTED_ENTRIES",
    "MemoryFullError",
    "ProfileRef",
    "UserProfile",
    "add_note",
    "add_rejected",
    "confirm_goal",
    "confirm_standing",
    "erase_all",
    "erase_notes",
    "erase_profile",
    "load_notes",
    "load_profile",
    "memory_dir",
    "remove_note",
    "replace_note",
    "set_style_notes",
    "set_suite_preference",
]
