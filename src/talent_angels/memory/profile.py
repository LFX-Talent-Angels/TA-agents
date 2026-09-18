"""Read and write structured user-profile facts to USER.md.

Only ``write_standing`` is called on a confirmed bind (_set_bind gate).
Never called on Locate alone. Node IDs + labels only — no prose descriptions.
"""

from __future__ import annotations

from datetime import date

from talent_angels.contracts.models import NodeRef
from talent_angels.memory.paths import MEMORY_MD, USER_MD


def _node_id_str(node: NodeRef) -> str:
    """Return suite:id_or_slug for display in USER.md."""
    source_id = (node.source_id or "").strip()
    if source_id:
        slug = source_id
    else:
        slug = node.pref_label.lower().replace(" ", "-")
    return f"{node.suite}:{slug}"


def read_user_profile() -> str:
    """Returns raw USER.md content, empty string if not found."""
    if USER_MD.exists():
        return USER_MD.read_text()
    return ""


def write_standing(node: NodeRef) -> None:
    """Called on explicit confirm. Updates STANDING line in USER.md.

    Format: STANDING: <pref_label>  [<suite>:<source_id or slug>]   since: <YYYY-MM-DD>
    If a STANDING line already exists, replace it. If not, insert at the top.
    """
    tag = _node_id_str(node)
    today = date.today().isoformat()
    new_line = f"STANDING[{node.suite}]: {node.pref_label}  [{tag}]   since: {today}"

    content = read_user_profile()
    lines = content.splitlines()
    for i, line in enumerate(lines):
        if line.startswith(f"STANDING[{node.suite}]:"):
            lines[i] = new_line
            USER_MD.write_text("\n".join(lines) + ("\n" if lines else ""))
            return

    # No existing STANDING line — insert at top
    lines.insert(0, new_line)
    USER_MD.write_text("\n".join(lines) + "\n")


def write_rejected(node: NodeRef) -> None:
    """Appends to REJECTED line in USER.md (comma-separated).

    Format: REJECTED: <label> [<suite>:<id_or_slug>], <label> [<suite>:<id_or_slug>]
    """
    tag = _node_id_str(node)
    entry = f"{node.pref_label} [{tag}]"

    content = read_user_profile()
    lines = content.splitlines()
    for i, line in enumerate(lines):
        if line.startswith("REJECTED:"):
            existing = line[len("REJECTED:"):].strip()
            lines[i] = f"REJECTED: {existing}, {entry}" if existing else f"REJECTED: {entry}"
            USER_MD.write_text("\n".join(lines) + "\n")
            return

    # No REJECTED line yet — append
    lines.append(f"REJECTED: {entry}")
    USER_MD.write_text("\n".join(lines) + "\n")


def write_goal(node: NodeRef) -> None:
    """Updates GOAL line in USER.md.

    Format: GOAL: <pref_label>  [<suite>:<source_id or slug>]
    """
    tag = _node_id_str(node)
    new_line = f"GOAL: {node.pref_label}  [{tag}]"

    content = read_user_profile()
    lines = content.splitlines()
    for i, line in enumerate(lines):
        if line.startswith("GOAL:"):
            lines[i] = new_line
            USER_MD.write_text("\n".join(lines) + "\n")
            return

    lines.append(new_line)
    USER_MD.write_text("\n".join(lines) + "\n")


def erase_person() -> None:
    """Deletes USER.md and MEMORY.md. Does not touch memory.db."""
    if USER_MD.exists():
        USER_MD.unlink()
    if MEMORY_MD.exists():
        MEMORY_MD.unlink()
