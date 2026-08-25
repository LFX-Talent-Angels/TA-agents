"""Smoke tests — confirm the generic runtime imports without concrete suites."""

import subprocess
import sys

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
    import talent_angels.suites  # noqa: F401


def test_esco_context_manager_is_exported_without_loading_concrete_suite() -> None:
    from talent_angels.skills.locate import open_esco_suite

    assert callable(open_esco_suite)


def test_default_registry_describes_every_suite_without_opening_one() -> None:
    from talent_angels.suites import default_suite_registry

    registry = default_suite_registry()

    assert registry.default == "esco"
    assert registry.available == ("bls", "esco", "onet", "sfia")


def test_cli_and_api_import_without_concrete_taxonomy_package() -> None:
    script = """
import builtins

real_import = builtins.__import__

def import_without_taxonomies(name, *args, **kwargs):
    if name == "ta_taxonomies" or name.startswith("ta_taxonomies."):
        raise ModuleNotFoundError("blocked for import-safety smoke test")
    return real_import(name, *args, **kwargs)

builtins.__import__ = import_without_taxonomies
import talent_angels.api.app
import talent_angels.cli
import talent_angels.tui.app
from talent_angels.suites import default_suite_registry
assert default_suite_registry().available == ("bls", "esco", "onet", "sfia")
"""

    subprocess.run([sys.executable, "-c", script], check=True)
