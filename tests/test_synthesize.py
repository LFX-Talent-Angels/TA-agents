"""One answer from N attached suites; a third suite appears without new branches."""

from __future__ import annotations

from talent_angels.assistant.synthesize import sources_line, synthesize_structured
from talent_angels.contracts import AgentResult, NodeRef


def _occ(suite: str, node_id: str, label: str) -> AgentResult:
    return AgentResult(
        capability="locate",
        suite=suite,
        nodes=[
            NodeRef(
                id=node_id,
                suite=suite,
                source=suite,
                source_id=node_id,
                kind="Occupation",
                pref_label=label,
            )
        ],
        confidence=0.95,
    )


def test_two_suites_one_answer_and_sources() -> None:
    answer = synthesize_structured(
        (
            _occ("esco", "esco:occupation:dev", "software developer"),
            _occ("onet", "onet:occupation:15-1252.00", "Software Developers"),
        )
    )
    assert "software developer" in answer
    assert "Software Developers" in answer
    assert "Sources used: ESCO · O*NET" in answer
    assert "not one shared id" in answer


def test_third_suite_appears_in_sources_without_hardcoding() -> None:
    answer = synthesize_structured(
        (
            _occ("esco", "esco:occupation:dev", "software developer"),
            _occ("onet", "onet:occupation:15-1252.00", "Software Developers"),
            _occ("sfia", "sfia:skill:prog", "Programming"),
        )
    )
    line = sources_line(
        (
            _occ("esco", "esco:occupation:dev", "software developer"),
            _occ("onet", "onet:occupation:15-1252.00", "Software Developers"),
            _occ("sfia", "sfia:skill:prog", "Programming"),
        )
    )
    assert "SFIA" in line
    assert "Programming" in answer
    assert "Sources used: ESCO · O*NET · SFIA" in answer
