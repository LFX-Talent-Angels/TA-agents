"""Smoke tests — confirm the package and its skeleton import cleanly."""

import talent_angels
from talent_angels.taxonomies import SUPPORTED_TAXONOMIES


def test_version() -> None:
    assert talent_angels.__version__


def test_agent_packages_import() -> None:
    # All agent + support packages should import without error.
    import talent_angels.connector  # noqa: F401
    import talent_angels.graph  # noqa: F401
    import talent_angels.locator  # noqa: F401
    import talent_angels.pathfinder  # noqa: F401
    import talent_angels.taxonomies  # noqa: F401


def test_supported_taxonomies() -> None:
    assert {"ESCO", "ONET", "SFIA", "BLS", "Lightcast"} == set(SUPPORTED_TAXONOMIES)
