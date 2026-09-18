"""``USER.md`` — the tiny, explicit-confirm-only career profile.

Nothing here is ever written from a silent high-confidence Locate. Every
write is a distinct, named function the caller invokes only after the
user has confirmed the fact out loud. Erase is total: there is no
partial-erase path, so "forget me" cannot leave a stale fact behind.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from talent_angels.memory.models import MAX_REJECTED_ENTRIES, ProfileRef, UserProfile

_FILENAME = "USER.md"
_REF_PATTERN = re.compile(r"^(?P<label>.*?)(?:\s*\[(?P<node_id>[^\]]+)\])?$")
_SINCE_PATTERN = re.compile(r"\s*since:\s*(?P<date>\S+)\s*$", re.IGNORECASE)


def memory_dir() -> Path:
    """Resolve the local memory directory (``TA_MEMORY_DIR``, gitignored default)."""
    raw = os.environ.get("TA_MEMORY_DIR", "data/local/memory")
    return Path(raw)


def _profile_path() -> Path:
    return memory_dir() / _FILENAME


def _suite_of(node_id: str | None) -> str | None:
    if node_id is None or ":" not in node_id:
        return None
    return node_id.split(":", 1)[0]


def _parse_ref(text: str) -> ProfileRef:
    match = _REF_PATTERN.match(text.strip())
    label = (match.group("label") if match else text).strip()
    node_id = match.group("node_id").strip() if match and match.group("node_id") else None
    return ProfileRef(label=label, node_id=node_id, suite=_suite_of(node_id))


def _render_ref(ref: ProfileRef) -> str:
    if ref.node_id:
        return f"{ref.label} [{ref.node_id}]"
    return ref.label


def _parse(text: str) -> UserProfile:
    standing: ProfileRef | None = None
    standing_since: str | None = None
    goal: ProfileRef | None = None
    rejected: list[ProfileRef] = []
    suite_preference: str | None = None
    style_notes: str | None = None

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip().upper()
        value = value.strip()
        if key == "STANDING":
            since_match = _SINCE_PATTERN.search(value)
            if since_match:
                standing_since = since_match.group("date")
                value = value[: since_match.start()].strip()
            standing = _parse_ref(value) if value else None
        elif key == "GOAL":
            goal = _parse_ref(value) if value else None
        elif key == "REJECTED":
            rejected = [_parse_ref(part) for part in value.split(";") if part.strip()]
        elif key == "SUITE":
            suite_preference = value or None
        elif key == "STYLE":
            style_notes = value or None

    return UserProfile(
        standing=standing,
        standing_since=standing_since,
        goal=goal,
        rejected=tuple(rejected[-MAX_REJECTED_ENTRIES:]),
        suite_preference=suite_preference,
        style_notes=style_notes,
    )


def _render(profile: UserProfile) -> str:
    lines: list[str] = []
    if profile.standing is not None:
        line = f"STANDING: {_render_ref(profile.standing)}"
        if profile.standing_since:
            line += f" since: {profile.standing_since}"
        lines.append(line)
    if profile.goal is not None:
        lines.append(f"GOAL: {_render_ref(profile.goal)}")
    if profile.rejected:
        rejected_text = "; ".join(_render_ref(ref) for ref in profile.rejected)
        lines.append(f"REJECTED: {rejected_text}")
    if profile.suite_preference:
        lines.append(f"SUITE: {profile.suite_preference}")
    if profile.style_notes:
        lines.append(f"STYLE: {profile.style_notes}")
    return ("\n".join(lines) + "\n") if lines else ""


def load_profile() -> UserProfile:
    """Read the profile, or an empty one if nothing has been confirmed yet."""
    path = _profile_path()
    if not path.exists():
        return UserProfile()
    return _parse(path.read_text(encoding="utf-8"))


def _save(profile: UserProfile) -> None:
    path = _profile_path()
    text = _render(profile)
    if not text:
        path.unlink(missing_ok=True)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def confirm_standing(label: str, *, node_id: str | None = None, since: str | None = None) -> None:
    """Record the user's confirmed current occupation. Caller confirmed this out loud."""
    profile = load_profile()
    suite = _suite_of(node_id)
    _save(
        profile.model_copy(
            update={
                "standing": ProfileRef(label=label, node_id=node_id, suite=suite),
                "standing_since": since,
            }
        )
    )


def confirm_goal(label: str, *, node_id: str | None = None) -> None:
    """Record the user's confirmed target occupation."""
    profile = load_profile()
    suite = _suite_of(node_id)
    _save(
        profile.model_copy(update={"goal": ProfileRef(label=label, node_id=node_id, suite=suite)})
    )


def add_rejected(label: str, *, node_id: str | None = None) -> None:
    """Record a candidate the user explicitly said was not them.

    Keeps only the most recent MAX_REJECTED_ENTRIES — oldest drops first,
    so the file stays tiny without ever needing lossy auto-compaction.
    """
    profile = load_profile()
    suite = _suite_of(node_id)
    new_ref = ProfileRef(label=label, node_id=node_id, suite=suite)
    updated = (*profile.rejected, new_ref)[-MAX_REJECTED_ENTRIES:]
    _save(profile.model_copy(update={"rejected": updated}))


def set_suite_preference(suite: str) -> None:
    profile = load_profile()
    _save(profile.model_copy(update={"suite_preference": suite}))


def set_style_notes(notes: str) -> None:
    profile = load_profile()
    _save(profile.model_copy(update={"style_notes": notes}))


def erase_profile() -> bool:
    """Delete the profile entirely. Returns True if a file was actually removed.

    This is total, not selective — "forget me" must not risk leaving a
    stale fact behind because only some fields were cleared.
    """
    path = _profile_path()
    if not path.exists():
        return False
    path.unlink()
    return True
