import os
from pathlib import Path

from talent_angels.tui.app import main


def test_ta_agent_script_is_configured() -> None:
    text = Path("pyproject.toml").read_text(encoding="utf-8")
    assert 'ta-agent = "talent_angels.tui.app:main"' in text
    assert "rich" in text


def test_tui_main_quit_on_stream(monkeypatch, tmp_path, capsys) -> None:
    from contextlib import contextmanager

    from talent_angels.suites import SuiteRegistry, SuiteRuntime
    from tests.test_edges_offline import FakeSuite

    monkeypatch.setenv("TA_SESSIONS_DIR", str(tmp_path / "sessions"))
    monkeypatch.setenv("RUNLOG_PATH", str(tmp_path / "runlog.jsonl"))
    monkeypatch.setenv("QUERY_DETAILS_DIR", str(tmp_path / "details"))
    monkeypatch.setenv("LLM_PROVIDER", "none")
    monkeypatch.setenv("LLM_MODEL", "stub")
    monkeypatch.setenv("ANSWER_MODE", "structured")

    @contextmanager
    def factory():
        yield SuiteRuntime(name="test", suite=FakeSuite(), health_check=lambda: True)

    registry = SuiteRegistry({"test": factory}, default="test")
    answers = iter(["hi", "/quit"])
    monkeypatch.setattr(
        "rich.console.Console.input",
        lambda self, *args, **kwargs: next(answers),
    )
    code = main([], registry=registry)
    captured = capsys.readouterr().out
    assert code == 0
    assert "ESCO desk" not in captured
    assert "Talent Angels" in captured or "I'm here" in captured or "here" in captured.lower()


def test_tui_records_an_episode_for_a_real_turn(monkeypatch, tmp_path, capsys) -> None:
    """Regression, found via a live run against real Neo4j: run_turn(persist=False)
    (deliberately, so a cancelled/Esc'd turn is never written — see tui/app.py) was
    also silently skipping record_episode, because that call lived inside the same
    persist-gated block in assistant/turn.py. The result: the one interface a
    person actually types into never recorded a single episode, ever, regardless
    of how much real usage it saw. Confirmed live: persist=True recorded an
    episode; persist=False (unpatched, this exact code path) recorded zero.
    """
    from contextlib import contextmanager

    from talent_angels.memory.episodes import recent_episodes
    from talent_angels.suites import SuiteRegistry, SuiteRuntime
    from tests.test_edges_offline import FakeSuite

    memory_db = tmp_path / "memory.db"
    monkeypatch.setattr("talent_angels.memory.episodes.DB_PATH", memory_db)
    monkeypatch.setenv("TA_SESSIONS_DIR", str(tmp_path / "sessions"))
    monkeypatch.setenv("RUNLOG_PATH", str(tmp_path / "runlog.jsonl"))
    monkeypatch.setenv("QUERY_DETAILS_DIR", str(tmp_path / "details"))
    monkeypatch.setenv("LLM_PROVIDER", "none")
    monkeypatch.setenv("LLM_MODEL", "stub")
    monkeypatch.setenv("ANSWER_MODE", "structured")

    @contextmanager
    def factory():
        yield SuiteRuntime(name="test", suite=FakeSuite(), health_check=lambda: True)

    registry = SuiteRegistry({"test": factory}, default="test")
    answers = iter(["software developer", "/quit"])
    monkeypatch.setattr(
        "rich.console.Console.input",
        lambda self, *args, **kwargs: next(answers),
    )

    assert recent_episodes() == []
    code = main([], registry=registry)
    capsys.readouterr()

    assert code == 0
    episodes = recent_episodes()
    assert len(episodes) == 1
    assert episodes[0].question == "software developer"


def test_runlog_path_follows_the_session_after_a_reset(monkeypatch, tmp_path) -> None:
    """The TUI must not keep logging into a session directory a reset deleted.

    `RUNLOG_PATH` is read from the environment on every write, and `/reset`
    hands back a fresh `session_id` while deleting the old directory. Without
    the re-point, the next turn recreates the erased directory and logs a
    conversation the user threw away — the erase quietly undoing itself.
    """
    from talent_angels.session.store import new_session, sessions_dir
    from talent_angels.tui.app import _sync_runlog_path

    monkeypatch.setenv("TA_SESSIONS_DIR", str(tmp_path / "sessions"))
    state = new_session()
    first_key = state.name or state.session_id
    monkeypatch.setenv("RUNLOG_PATH", str(sessions_dir() / first_key / "runlog.jsonl"))

    # A reset returns a different session, as `handle_line` does.
    fresh = new_session()
    fresh_key = _sync_runlog_path(fresh, first_key)

    assert fresh_key == fresh.session_id
    assert fresh_key != first_key
    assert os.environ["RUNLOG_PATH"] == str(sessions_dir() / fresh.session_id / "runlog.jsonl")
    assert first_key not in os.environ["RUNLOG_PATH"], (
        "the erased session's directory must not be written to again"
    )


def test_runlog_path_is_left_alone_when_the_session_has_not_moved(monkeypatch, tmp_path) -> None:
    """An ordinary turn must not churn the path — no needless rewrite."""
    from talent_angels.session.store import new_session, sessions_dir
    from talent_angels.tui.app import _sync_runlog_path

    monkeypatch.setenv("TA_SESSIONS_DIR", str(tmp_path / "sessions"))
    state = new_session()
    key = state.name or state.session_id
    target = str(sessions_dir() / key / "runlog.jsonl")
    monkeypatch.setenv("RUNLOG_PATH", target)

    assert _sync_runlog_path(state, key) == key
    assert os.environ["RUNLOG_PATH"] == target
