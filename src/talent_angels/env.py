"""Load a local `.env` without overriding a real shell export."""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv


def _repo_root() -> Path | None:
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    return None


def local_dotenv_candidates() -> tuple[Path, ...]:
    """Prefer cwd `.env`, then the repo-root file next to `pyproject.toml`."""
    seen: set[Path] = set()
    ordered: list[Path] = []
    for path in (Path.cwd() / ".env",):
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            ordered.append(path)
    root = _repo_root()
    if root is not None:
        path = root / ".env"
        resolved = path.resolve()
        if resolved not in seen:
            ordered.append(path)
    return tuple(ordered)


def load_local_dotenv() -> Path | None:
    """Load the first existing candidate with ``override=False``.

    Shell exports always win. Values are never printed.
    """
    loaded: Path | None = None
    for path in local_dotenv_candidates():
        if not path.is_file():
            continue
        load_dotenv(path, override=False)
        if loaded is None:
            loaded = path
    return loaded
