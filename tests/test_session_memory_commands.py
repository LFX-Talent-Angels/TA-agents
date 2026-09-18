"""Offline tests for /standing, /goal, /reject, /whoami, /forget."""

from __future__ import annotations

from pathlib import Path

import pytest

from talent_angels.contracts import NodeRef
from talent_angels.session.kernel import handle_line
from talent_angels.session.models import LastBinding
from talent_angels.session.store import new_session


def _boom(question: str, **kwargs: object):
    raise AssertionError(f"runner must not be called, got {question!r}")


def _bound_state(label: str = "software developer", node_id: str = "esco:occupation:1"):
    state = new_session()
    state.binding = LastBinding(
        node=NodeRef(
            id=node_id,
            suite="esco",
            source="esco",
            source_id="src-1",
            kind="Occupation",
            pref_label=label,
        )
    )
    return state


@pytest.fixture(autouse=True)
def _local_dirs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("TA_SESSIONS_DIR", str(tmp_path / "sessions"))
    monkeypatch.setenv("TA_MEMORY_DIR", str(tmp_path / "memory"))


def test_standing_requires_a_binding_first() -> None:
    state = new_session()

    reply = handle_line(state, "/standing", runner=_boom)

    assert "Nothing is bound yet" in reply.text


def test_standing_confirms_the_bound_node() -> None:
    from talent_angels.memory import load_profile

    state = _bound_state()

    reply = handle_line(state, "/standing", runner=_boom)

    assert "software developer" in reply.text
    profile = load_profile()
    assert profile.standing is not None
    assert profile.standing.label == "software developer"
    assert profile.standing.node_id == "esco:occupation:1"
    assert profile.standing_since is not None


def test_goal_confirms_the_bound_node_without_touching_standing() -> None:
    from talent_angels.memory import load_profile

    state = _bound_state()
    handle_line(state, "/standing", runner=_boom)

    state = _bound_state(label="data scientist", node_id="esco:occupation:2")
    handle_line(state, "/goal", runner=_boom)

    profile = load_profile()
    assert profile.standing is not None and profile.standing.label == "software developer"
    assert profile.goal is not None and profile.goal.label == "data scientist"


def test_reject_adds_to_the_rejected_list() -> None:
    from talent_angels.memory import load_profile

    state = _bound_state(label="web developer", node_id="esco:occupation:3")

    reply = handle_line(state, "/reject", runner=_boom)

    assert "not web developer" in reply.text
    profile = load_profile()
    assert [ref.label for ref in profile.rejected] == ["web developer"]


def test_whoami_reports_nothing_saved_before_any_confirmation() -> None:
    state = new_session()

    reply = handle_line(state, "/whoami", runner=_boom)

    assert "Nothing saved yet" in reply.text


def test_whoami_reports_what_was_confirmed() -> None:
    state = _bound_state()
    handle_line(state, "/standing", runner=_boom)

    reply = handle_line(state, "/profile", runner=_boom)

    assert "Standing: software developer" in reply.text


def test_forget_erases_everything_and_reports_when_nothing_was_there() -> None:
    state = _bound_state()
    handle_line(state, "/standing", runner=_boom)

    reply = handle_line(state, "/forget", runner=_boom)
    assert "Cleared" in reply.text

    reply_again = handle_line(state, "/forget", runner=_boom)
    assert "Nothing was saved" in reply_again.text

    from talent_angels.memory import load_profile

    assert load_profile().is_empty()
