"""Combine per-suite AgentResults into one user-facing answer.

Each result stays suite-scoped. IDs are never equated. Similar labels across
suites are shown side by side; the caller may label that resemblance as
model inference, never as a graph fact.
"""

from __future__ import annotations

from collections.abc import Sequence

from talent_angels.assistant.answer import summarize_result
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
    """One labeled block per suite, separated by a blank line."""
    if not results:
        if extra_warnings:
            return "No attached taxonomy was reachable. [" + ", ".join(extra_warnings) + "]"
        return "No attached taxonomy was reachable."

    blocks = [f"{suite_heading(result.suite)} · {summarize_result(result)}" for result in results]
    if extra_warnings:
        blocks.append("[" + ", ".join(extra_warnings) + "]")
    return "\n\n".join(blocks)
