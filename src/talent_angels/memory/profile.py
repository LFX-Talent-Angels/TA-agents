"""Read and write structured user-profile facts to USER.md.

Only what the user says about themselves is written: a lookup or a pick is
not a statement about the user. Node IDs + labels only — no prose descriptions.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import date

from talent_angels.contracts.models import NodeRef
from talent_angels.memory.files import MEMORY_FILE_LOCK, atomic_write_text, read_text_or_empty
from talent_angels.memory.paths import user_md


def _node_id_str(node: NodeRef) -> str:
    """Return suite:id_or_slug for display in USER.md.

    Normalises three source_id shapes so the stored id never doubles the suite
    prefix (a graph id like ``onet:occupation:51-4121.00`` was historically
    written back as ``onet:onet:occupation:...``):
    1. ``http...``         → last URI path segment
    2. ``<suite>:<rest>``  → last ``:<code>`` segment (graph-id pipeline fallback)
    3. anything else       → as-is (native id, e.g. O*NET SOC code)
    """
    source_id = (node.source_id or "").strip()
    if source_id:
        if source_id.startswith("http"):
            slug = source_id.rstrip("/").rsplit("/", 1)[-1]
        elif source_id.startswith(f"{node.suite}:"):
            slug = source_id.rsplit(":", 1)[-1]
        else:
            slug = source_id
    else:
        slug = node.pref_label.lower().replace(" ", "-")
    return f"{node.suite}:{slug}"


def read_user_profile() -> str:
    """Returns raw USER.md content, empty string if not found."""
    return read_text_or_empty(user_md())


def _label_of(line: str) -> str:
    """``GOAL: data analyst  [esco:x]`` → ``data analyst``."""
    return line.split(":", 1)[1].split("  [", 1)[0].strip()


def profile_titles() -> tuple[str | None, str | None]:
    """(current job, goal) as stored, ESCO's standing first when there are several."""
    standing: dict[str, str] = {}
    goal: str | None = None
    for line in read_user_profile().splitlines():
        if line.startswith("GOAL:"):
            goal = _label_of(line) or None
        elif line.startswith("STANDING[") and "]:" in line:
            suite = line[len("STANDING[") : line.index("]")]
            standing[suite] = _label_of(line.replace("]:", ":", 1))
    current = standing.get("esco") or next(iter(standing.values()), None)
    return current or None, goal


def profile_line() -> str | None:
    """ "current job: X; goal: Y" for the planner, or None when nothing is saved."""
    current, goal = profile_titles()
    parts = [f"current job: {current}" if current else "", f"goal: {goal}" if goal else ""]
    line = "; ".join(part for part in parts if part)
    return line or None


# Hermes-style caps (docs: Persistent Memory). USER.md holds the user profile.
USER_MAX_CHARS = 1375


def _node_label(node: NodeRef) -> str:
    """Guard: refuse to persist blank labels — they clobber a good standing."""
    label = (node.pref_label or "").strip()
    if not label or label.casefold() == "none":
        return ""
    return label


def _prune_to_limit(content: str, limit: int | None = None) -> str:
    """Deterministic consolidation when USER.md would exceed USER_MAX_CHARS.

    Eviction order (least → most important): oldest STANDING line, then the
    oldest REJECTED entries, then GOAL is kept last. Mirrors Hermes' "make
    room before retrying" — here done in code because writes are code-driven.
    """
    if limit is None:
        limit = USER_MAX_CHARS
    if len(content) <= limit:
        return content
    lines = content.splitlines()
    # 1. Oldest STANDING line (first seen in file = earliest `since`).
    for i, line in enumerate(lines):
        if line.startswith("STANDING["):
            candidate = lines[:i] + lines[i + 1 :]
            if len("\n".join(candidate)) <= limit:
                return "\n".join(candidate)
    # 2. Oldest REJECTED entries (drop from the tail of the list, i.e. earlier).
    for i, line in enumerate(lines):
        if line.startswith("REJECTED:"):
            entries = [e.strip() for e in line[len("REJECTED:") :].split(",") if e.strip()]
            while entries and len("\n".join(lines)) > limit:
                entries.pop(0)  # drop oldest first
                lines[i] = "REJECTED: " + ", ".join(entries)
            break
    if len("\n".join(lines)) <= limit:
        return "\n".join(lines)
    # 3. Last resort: GOAL after everything else failed.
    kept = [line for line in lines if not line.startswith(("STANDING[", "REJECTED:"))]
    return "\n".join(kept) if len("\n".join(kept)) <= limit else kept[-1]


def _rewrite(update: Callable[[list[str]], list[str]]) -> None:
    """Read-modify-write USER.md under the memory-file lock, atomically."""
    with MEMORY_FILE_LOCK:
        lines = update(read_user_profile().splitlines())
        atomic_write_text(user_md(), _prune_to_limit("\n".join(lines)) + "\n")


def write_standing(node: NodeRef) -> None:
    """Called on explicit confirm. Updates STANDING line in USER.md.

    Format: STANDING: <pref_label>  [<suite>:<source_id or slug>]   since: <YYYY-MM-DD>
    If a STANDING line already exists, replace it. If not, insert at the top.
    Blank/no-op labels are refused so a bad node never clobbers a good line.
    """
    label = _node_label(node)
    if not label:
        return
    tag = _node_id_str(node)
    today = date.today().isoformat()
    new_line = f"STANDING[{node.suite}]: {label}  [{tag}]   since: {today}"

    def update(lines: list[str]) -> list[str]:
        for i, line in enumerate(lines):
            if line.startswith(f"STANDING[{node.suite}]:"):
                lines[i] = new_line
                return lines
        return [new_line, *lines]

    _rewrite(update)


def write_rejected(node: NodeRef) -> None:
    """Appends to REJECTED line in USER.md (comma-separated).

    Format: REJECTED: <label> [<suite>:<id_or_slug>], <label> [<suite>:<id_or_slug>]
    """
    label = _node_label(node)
    if not label:
        return
    entry = f"{label} [{_node_id_str(node)}]"

    def update(lines: list[str]) -> list[str]:
        for i, line in enumerate(lines):
            if line.startswith("REJECTED:"):
                existing = line[len("REJECTED:") :].strip()
                lines[i] = f"REJECTED: {existing}, {entry}" if existing else f"REJECTED: {entry}"
                return lines
        return [*lines, f"REJECTED: {entry}"]

    _rewrite(update)


def drop_standing_matching(subject: str) -> list[str]:
    """Remove current-job lines naming ``subject`` and record them as rejected.

    "I am not a teacher" after "dance teacher" was saved: the denial is about
    the saved title, so it is matched on whole words of that title, not searched.
    Returns the removed titles (one per suite line).
    """
    words = subject.casefold().split()
    removed: list[str] = []

    def names_subject(label: str) -> bool:
        tokens = label.casefold().replace("-", " ").split()
        return bool(words) and all(
            any(token == word or token == f"{word}s" for token in tokens) for word in words
        )

    def update(lines: list[str]) -> list[str]:
        kept: list[str] = []
        entries: list[str] = []
        for line in lines:
            if line.startswith("STANDING[") and "]:" in line:
                body = line.split("]:", 1)[1]
                label = body.split("  [", 1)[0].strip()
                if names_subject(label):
                    tag = body.split("  [", 1)[1].split("]", 1)[0] if "  [" in body else ""
                    removed.append(label)
                    entries.append(f"{label} [{tag}]" if tag else label)
                    continue
            kept.append(line)
        if not entries:
            return lines
        for i, line in enumerate(kept):
            if line.startswith("REJECTED:"):
                existing = line[len("REJECTED:") :].strip()
                joined = ", ".join(entries)
                kept[i] = f"REJECTED: {existing}, {joined}" if existing else f"REJECTED: {joined}"
                return kept
        return [*kept, "REJECTED: " + ", ".join(entries)]

    _rewrite(update)
    return removed


def write_goal(node: NodeRef) -> None:
    """Updates GOAL line in USER.md.

    Format: GOAL: <pref_label>  [<suite>:<source_id or slug>]
    """
    label = _node_label(node)
    if not label:
        return
    new_line = f"GOAL: {label}  [{_node_id_str(node)}]"

    def update(lines: list[str]) -> list[str]:
        for i, line in enumerate(lines):
            if line.startswith("GOAL:"):
                lines[i] = new_line
                return lines
        return [*lines, new_line]

    _rewrite(update)


#: Most-important first: a card truncated by line count must never lose the
#: goal or the rejections to a pile of per-suite STANDING lines.
_CARD_ORDER = ("GOAL:", "REJECTED:", "STANDING[")
_CARD_MAX_LINES = 6


def profile_prefix() -> str:
    """Returns a USER.md profile card for LLM system prompts, or empty string."""
    content = read_user_profile().strip()
    if not content:
        return ""
    lines = [line for line in content.splitlines() if line.strip()]

    def rank(line: str) -> int:
        for i, prefix in enumerate(_CARD_ORDER):
            if line.startswith(prefix):
                return i
        return len(_CARD_ORDER)

    card = "\n".join(sorted(lines, key=rank)[:_CARD_MAX_LINES])
    return f"[User profile — confirmed by user, not from taxonomy]\n{card}\n\n"
