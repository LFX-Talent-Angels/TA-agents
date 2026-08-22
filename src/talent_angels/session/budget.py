"""Model-facing context budget (not the on-disk transcript)."""

from __future__ import annotations

from talent_angels.session.models import SessionState


def model_view(state: SessionState, current: str) -> str:
    """Compact prompt slice: current line plus optional binding.

    v1 map-work lock: never dump transcript skill lists into the model view.
    """
    parts = [current]
    if state.binding is not None:
        node = state.binding.node
        parts.append(f"Bound occupation: {node.pref_label} ({node.id})")
    for line in reversed(state.transcript):
        if line.role == "user" and line.text != current:
            parts.append(f"Previous user: {line.text[:200]}")
            break
    return "\n".join(parts)
