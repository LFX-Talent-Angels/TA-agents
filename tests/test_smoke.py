"""Smoke tests — confirm the generic runtime imports without concrete suites."""

import talent_angels


def test_version() -> None:
    assert talent_angels.__version__


def test_runtime_packages_import() -> None:
    import talent_angels.assistant  # noqa: F401
    import talent_angels.contracts  # noqa: F401
    import talent_angels.runlog  # noqa: F401
    import talent_angels.skills.connect  # noqa: F401
    import talent_angels.skills.evaluate  # noqa: F401
    import talent_angels.skills.locate  # noqa: F401
    import talent_angels.skills.pathfind  # noqa: F401


def test_esco_context_manager_is_exported_without_loading_concrete_suite() -> None:
    from talent_angels.skills.locate import open_esco_suite

    assert callable(open_esco_suite)
