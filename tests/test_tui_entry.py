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
