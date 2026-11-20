"""/reset must not erase a session the user saved by name."""

from __future__ import annotations

from talent_angels.contracts import NodeRef
from talent_angels.session.kernel import handle_line
from talent_angels.session.models import LastBinding
from talent_angels.session.store import new_session, save_session, sessions_dir


def _boom(*_args: object, **_kwargs: object) -> object:
    raise AssertionError("commands must not run a turn")


def _chef() -> NodeRef:
    return NodeRef(
        id="esco:occupation:chef",
        suite="esco",
        source="esco",
        source_id="chef",
        kind="Occupation",
        pref_label="chef",
    )


def test_reset_after_save_keeps_the_saved_session_resumable() -> None:
    state = new_session()
    state.bindings["esco"] = _chef()
    state.binding = LastBinding(node=_chef())
    handle_line(state, "/save chefwork", runner=_boom)

    reply = handle_line(state, "/reset", runner=_boom)

    assert state.name is None
    assert state.bindings == {}
    assert (sessions_dir() / "chefwork").is_dir()
    assert "/resume chefwork" in reply.text

    resumed = handle_line(state, "/resume chefwork", runner=_boom)
    assert "Resumed" in resumed.text
    assert state.bindings["esco"].pref_label == "chef"


def test_reset_of_an_unnamed_session_still_erases_it() -> None:
    state = new_session()
    save_session(state)
    old_dir = sessions_dir() / state.session_id
    assert old_dir.is_dir()

    handle_line(state, "/reset", runner=_boom)

    assert not old_dir.exists()
