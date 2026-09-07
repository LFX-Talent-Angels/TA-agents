"""Smoke tests — confirm the package and its skeleton import cleanly."""

import pytest

import talent_angels


def test_version() -> None:
    assert talent_angels.__version__


def test_runtime_packages_import() -> None:
    # These have no dependency on the (unpublished) TA-taxonomies suite
    # library and must always import cleanly.
    import talent_angels.contracts  # noqa: F401
    import talent_angels.memory  # noqa: F401
    import talent_angels.runlog  # noqa: F401
    import talent_angels.skills.connect  # noqa: F401
    import talent_angels.skills.evaluate  # noqa: F401
    import talent_angels.skills.pathfind  # noqa: F401


def test_esco_dependent_packages_import() -> None:
    # api/assistant/skills.locate call into ta_taxonomies (TA-taxonomies) —
    # skip cleanly where it isn't installed rather than failing collection
    # (CONTRIBUTING.md: fixture-backed tests should skip, not hard-fail).
    pytest.importorskip("ta_taxonomies")
    import talent_angels.api  # noqa: F401
    import talent_angels.assistant  # noqa: F401
    import talent_angels.skills.locate  # noqa: F401
