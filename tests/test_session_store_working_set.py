"""A saved session brings back its working set and a pick still owed a profile write."""

from __future__ import annotations

import json

from talent_angels.session.store import load_session, new_session, save_session, sessions_dir


def test_recent_titles_and_pending_intent_survive_save_and_resume() -> None:
    state = new_session()
    state.recent = ["plumber", "electrician"]
    state.pending_profile_intent = "standing"

    save_session(state, name="work")
    loaded = load_session("work")

    assert loaded.recent == ["plumber", "electrician"]
    assert loaded.pending_profile_intent == "standing"


def test_a_session_saved_before_the_working_set_still_loads() -> None:
    state = new_session()
    save_session(state, name="old")
    binding = sessions_dir() / "old" / "binding.json"
    payload = json.loads(binding.read_text(encoding="utf-8"))
    del payload["recent"], payload["pending_profile_intent"]
    binding.write_text(json.dumps(payload), encoding="utf-8")

    loaded = load_session("old")

    assert loaded.recent == []
    assert loaded.pending_profile_intent is None
