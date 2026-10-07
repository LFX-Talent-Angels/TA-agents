"""The memory home is resolved per call, never inside site-packages, never at import."""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from talent_angels.contracts import NodeRef
from talent_angels.memory import paths, profile


def test_home_honours_an_override_set_after_import(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`.env` is loaded after the package is imported; the override must still win."""
    monkeypatch.setenv("TA_AGENTS_HOME", str(tmp_path / "late"))
    assert paths.home() == tmp_path / "late"
    assert paths.db_path() == tmp_path / "late" / "memory.db"
    assert paths.default_runlog_path() == tmp_path / "late" / "runlog.jsonl"


def test_installed_package_uses_the_user_home_not_site_packages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("TA_AGENTS_HOME", raising=False)
    fake_site = tmp_path / "lib" / "python3.11" / "site-packages"
    fake_site.mkdir(parents=True)
    monkeypatch.setattr(paths, "_CANDIDATE_ROOT", fake_site.parent)
    monkeypatch.setattr(paths.Path, "home", classmethod(lambda cls: tmp_path / "user"))

    assert paths.project_root() is None
    assert paths.home() == tmp_path / "user" / ".ta-agents"


def test_source_checkout_uses_the_project_local_home(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TA_AGENTS_HOME", raising=False)
    root = paths.project_root()
    assert root is not None and (root / "pyproject.toml").is_file()
    assert paths.home() == root / ".ta-agents"


def test_resolving_the_home_creates_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "not-yet"
    monkeypatch.setenv("TA_AGENTS_HOME", str(target))
    paths.home(), paths.user_md(), paths.db_path()
    assert not target.exists()
    assert paths.ensure_home() == target and target.is_dir()


def _node(suite: str, label: str) -> NodeRef:
    return NodeRef(
        id=f"{suite}:occupation:{label}",
        suite=suite,
        source=suite,
        source_id=label,
        kind="Occupation",
        pref_label=label,
    )


def test_concurrent_profile_writes_do_not_lose_updates() -> None:
    """8 writers on the API threadpool used to keep ~6 of 8 lines."""
    suites = [f"s{i}" for i in range(8)]
    barrier = threading.Barrier(len(suites))

    def write(suite: str) -> None:
        barrier.wait()
        profile.write_standing(_node(suite, f"job-{suite}"))

    threads = [threading.Thread(target=write, args=(s,)) for s in suites]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    content = profile.read_user_profile()
    assert all(f"STANDING[{s}]:" in content for s in suites), content


def test_profile_card_keeps_goal_and_rejections_ahead_of_standings() -> None:
    profile.write_goal(_node("esco", "data scientist"))
    profile.write_rejected(_node("esco", "chef"))
    for i in range(6):
        profile.write_standing(_node(f"s{i}", f"job{i}"))

    card = profile.profile_prefix()
    assert "Goal: data scientist" in card
    assert "Not their job (they said so): chef" in card
    assert "esco:" not in card
