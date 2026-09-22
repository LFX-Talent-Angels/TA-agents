"""Read and append agent notes to MEMORY.md.

These are operational facts the agent records for cross-session continuity.
Only bullet-point entries — no prose paragraphs.
"""

from __future__ import annotations

from talent_angels.memory.paths import MEMORY_MD


def read_agent_notes() -> str:
    """Returns raw MEMORY.md content, empty string if not found."""
    if MEMORY_MD.exists():
        return MEMORY_MD.read_text()
    return ""


def append_note(note: str) -> None:
    """Appends a bullet note to MEMORY.md."""
    existing = read_agent_notes()
    updated = existing + f"- {note}\n"
    MEMORY_MD.write_text(updated)
