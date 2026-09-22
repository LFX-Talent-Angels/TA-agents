"""Read and append agent notes to MEMORY.md.

These are operational facts the agent records for cross-session continuity.
Only bullet-point entries — no prose paragraphs.

Capacity mirrors Hermes' persistent-memory model: MEMORY.md is capped (default
2,200 chars). ``append_note`` de-duplicates and consolidates deterministically —
oldest notes are evicted first so the file never exceeds the cap.
"""

from __future__ import annotations

from talent_angels.memory.paths import MEMORY_MD

MEMORY_MAX_CHARS = 2200


def read_agent_notes() -> str:
    """Returns raw MEMORY.md content, empty string if not found."""
    if MEMORY_MD.exists():
        return MEMORY_MD.read_text()
    return ""


def notes_prefix() -> str:
    """Returns MEMORY.md wrapped in a labeled block for system prompts, or empty string."""
    content = read_agent_notes().strip()
    if not content:
        return ""
    return f"[Agent operational notes]\n{content}\n\n"


def append_note(note: str) -> None:
    """Appends a bullet note to MEMORY.md (exact duplicates skipped).

    Consolidation: when the file would exceed MEMORY_MAX_CHARS, the oldest
    notes are dropped first (newest wins), mirroring Hermes' bounded memory.
    """
    existing = read_agent_notes()
    lines = [line for line in existing.splitlines() if line.strip()]
    entry = f"- {note.strip()}"
    if not note.strip() or entry in lines:
        return
    lines.append(entry)
    while len("\n".join(lines)) > MEMORY_MAX_CHARS and len(lines) > 1:
        lines.pop(0)
    MEMORY_MD.write_text("\n".join(lines) + "\n")
