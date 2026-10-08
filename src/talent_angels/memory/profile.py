"""Read and write structured user-profile facts to USER.md.

Only ``write_standing`` is called on a confirmed bind (_set_bind gate).
Never called on Locate alone. Node IDs + labels only — no prose descriptions.
"""

from __future__ import annotations

from datetime import date

from talent_angels.contracts.models import NodeRef
from talent_angels.memory.paths import MEMORY_MD, USER_MD


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
    if USER_MD.exists():
        return USER_MD.read_text()
    return ""


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

    content = read_user_profile()
    lines = content.splitlines()
    for i, line in enumerate(lines):
        if line.startswith(f"STANDING[{node.suite}]:"):
            lines[i] = new_line
            break
    else:
        # No existing STANDING line — insert at top
        lines.insert(0, new_line)
    USER_MD.write_text(_prune_to_limit("\n".join(lines)) + "\n")


def write_rejected(node: NodeRef) -> None:
    """Appends to REJECTED line in USER.md (comma-separated).

    Format: REJECTED: <label> [<suite>:<id_or_slug>], <label> [<suite>:<id_or_slug>]
    """
    label = _node_label(node)
    if not label:
        return
    tag = _node_id_str(node)
    entry = f"{label} [{tag}]"

    content = read_user_profile()
    lines = content.splitlines()
    for i, line in enumerate(lines):
        if line.startswith("REJECTED:"):
            existing = line[len("REJECTED:") :].strip()
            lines[i] = f"REJECTED: {existing}, {entry}" if existing else f"REJECTED: {entry}"
            break
    else:
        # No REJECTED line yet — append
        lines.append(f"REJECTED: {entry}")
    USER_MD.write_text(_prune_to_limit("\n".join(lines)) + "\n")


def write_goal(node: NodeRef) -> None:
    """Updates GOAL line in USER.md.

    Format: GOAL: <pref_label>  [<suite>:<source_id or slug>]
    """
    label = _node_label(node)
    if not label:
        return
    tag = _node_id_str(node)
    new_line = f"GOAL: {label}  [{tag}]"

    content = read_user_profile()
    lines = content.splitlines()
    for i, line in enumerate(lines):
        if line.startswith("GOAL:"):
            lines[i] = new_line
            break
    else:
        lines.append(new_line)
    USER_MD.write_text(_prune_to_limit("\n".join(lines)) + "\n")


def profile_prefix() -> str:
    """Returns a USER.md profile card for LLM system prompts, or empty string."""
    content = read_user_profile().strip()
    if not content:
        return ""
    lines = content.splitlines()[:5]
    card = "\n".join(lines)
    return f"[User profile — confirmed by user, not from taxonomy]\n{card}\n\n"


def erase_person() -> None:
    """Deletes USER.md and MEMORY.md. Does not touch memory.db."""
    if USER_MD.exists():
        USER_MD.unlink()
    if MEMORY_MD.exists():
        MEMORY_MD.unlink()
