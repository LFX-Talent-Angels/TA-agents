"""Typed shapes for the long-term personal memory.

Node identity is optional per entry: a custom or label-only taxonomy may
have no stable id, so every reference keeps a human-readable label and an
optional ``node_id`` rather than assuming one always exists.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

# Provisional cap, not yet finalized. Keeps the profile card small; raise
# once there's an agreed real number.
MAX_REJECTED_ENTRIES = 5


class ProfileRef(BaseModel):
    """A career-graph reference the user confirmed — a label, optionally an id."""

    model_config = ConfigDict(frozen=True)

    label: str
    node_id: str | None = None  # e.g. "onet:15-1252.00"; None for label-only suites
    suite: str | None = None


class UserProfile(BaseModel):
    """The tiny, human-editable career profile (``USER.md``)."""

    model_config = ConfigDict(frozen=True)

    standing: ProfileRef | None = None
    standing_since: str | None = None  # ISO date, e.g. "2026-03-10"
    goal: ProfileRef | None = None
    rejected: tuple[ProfileRef, ...] = Field(default_factory=tuple)
    suite_preference: str | None = None
    style_notes: str | None = None

    def is_empty(self) -> bool:
        return (
            self.standing is None
            and self.goal is None
            and not self.rejected
            and self.suite_preference is None
            and self.style_notes is None
        )
