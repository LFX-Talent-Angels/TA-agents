"""Which taxonomies a tagging run uses when --suites is not given."""

from __future__ import annotations

import pytest

from talent_angels.tagging.runner import TAG_SUITES_ENV, default_tag_suites

ALL = ("bls", "esco", "jobtech", "onet", "sfia")


def test_default_is_esco_and_onet(monkeypatch: pytest.MonkeyPatch) -> None:
    # Each taxonomy costs one model call per row: new ones are opt-in.
    monkeypatch.delenv(TAG_SUITES_ENV, raising=False)
    assert default_tag_suites(ALL) == ["esco", "onet"]


def test_other_suites_stay_attached(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(TAG_SUITES_ENV, raising=False)
    assert default_tag_suites(("down", "esco", "sfia")) == ["down", "esco"]


def test_a_registry_of_only_opt_in_suites_uses_them(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(TAG_SUITES_ENV, raising=False)
    assert default_tag_suites(("bls",)) == ["bls"]


def test_env_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(TAG_SUITES_ENV, " esco , bls,, jobtech ")
    assert default_tag_suites(ALL) == ["esco", "bls", "jobtech"]
