"""Smoke tests — confirm the package and its skeleton import cleanly."""

import talent_angels


def test_version() -> None:
    assert talent_angels.__version__


def test_runtime_packages_import() -> None:
    # The assistant runtime + all skill packages should import without error.
    import talent_angels.api  # noqa: F401
    import talent_angels.assistant  # noqa: F401
    import talent_angels.contracts  # noqa: F401
    import talent_angels.runlog  # noqa: F401
    import talent_angels.skills.connect  # noqa: F401
    import talent_angels.skills.evaluate  # noqa: F401
    import talent_angels.skills.locate  # noqa: F401
    import talent_angels.skills.pathfind  # noqa: F401
