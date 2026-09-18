"""``MEMORY.md`` — the agent's own operational notes.

Not who the user is (that is ``profile.py``) — how to serve them better:
taxonomy quirks, phrasing preferences, things worth not re-discovering.

Deliberately no auto-compaction. A full file raises instead of silently
truncating or summarizing itself away, so the caller has to make a real
judgment call about what to drop.
"""

from __future__ import annotations

from pathlib import Path

from talent_angels.memory.profile import memory_dir

_FILENAME = "MEMORY.md"

# Provisional cap, not yet a team decision. Keeps the file tiny by
# construction rather than by later compression.
MAX_NOTES = 20


class MemoryFullError(RuntimeError):
    """Raised by add_note when the notes file is already at MAX_NOTES."""


def _notes_path() -> Path:
    return memory_dir() / _FILENAME


def load_notes() -> list[str]:
    path = _notes_path()
    if not path.exists():
        return []
    notes: list[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line.startswith("- "):
            notes.append(line[2:].strip())
        elif line.startswith("-"):
            notes.append(line[1:].strip())
    return notes


def _save(notes: list[str]) -> None:
    path = _notes_path()
    if not notes:
        path.unlink(missing_ok=True)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(f"- {note}" for note in notes) + "\n"
    path.write_text(text, encoding="utf-8")


def add_note(text: str) -> None:
    """Append a new operational note.

    Raises MemoryFullError at MAX_NOTES rather than silently dropping or
    compressing an existing note — the caller must consolidate on purpose.
    """
    notes = load_notes()
    if len(notes) >= MAX_NOTES:
        raise MemoryFullError(
            f"MEMORY.md is at its {MAX_NOTES}-note cap; replace or remove one first"
        )
    notes.append(text.strip())
    _save(notes)


def replace_note(substring: str, new_text: str) -> bool:
    """Replace the first note containing ``substring`` with ``new_text``.

    Returns False (no-op) if nothing matched.
    """
    notes = load_notes()
    for i, note in enumerate(notes):
        if substring in note:
            notes[i] = new_text.strip()
            _save(notes)
            return True
    return False


def remove_note(substring: str) -> bool:
    """Remove the first note containing ``substring``. Returns False if none matched."""
    notes = load_notes()
    for i, note in enumerate(notes):
        if substring in note:
            del notes[i]
            _save(notes)
            return True
    return False


def erase_notes() -> bool:
    """Delete MEMORY.md entirely. Returns True if a file was actually removed."""
    path = _notes_path()
    if not path.exists():
        return False
    path.unlink()
    return True
