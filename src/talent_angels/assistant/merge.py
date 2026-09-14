"""Combine per-suite AgentResults into one user-facing answer.

Each result stays suite-scoped. IDs are never equated. Similar labels across
suites are shown side by side; the caller may label that resemblance as
model inference, never as a graph fact.
"""

from __future__ import annotations

from collections.abc import Sequence

from talent_angels.contracts import AgentResult

_HEADINGS = {
    "esco": "ESCO",
    "onet": "O*NET",
    "sfia": "SFIA",
    "bls": "BLS",
}


def suite_heading(name: str) -> str:
    """Stable display label for a registry suite name."""
    return _HEADINGS.get(name.casefold(), name.upper())


def merge_answers(
    results: Sequence[AgentResult],
    *,
    extra_warnings: Sequence[str] = (),
) -> str:
    """One user-facing answer citing every attached suite."""
    from talent_angels.assistant.synthesize import synthesize_structured

    return synthesize_structured(results, extra_warnings=extra_warnings)
