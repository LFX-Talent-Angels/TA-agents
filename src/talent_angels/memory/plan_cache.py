"""The planner's reading of a question, kept so the same words read the same way.

A model's output varies between runs even for identical input (temperature 0
does not make Claude deterministic), and "swe" read once with suggested titles
and once without gave a pick list one time and an auto-picked title the next.
The plan is cached by the normalised question, the profile line and a hash of
the planner prompt and model, so a repeat question reuses the first reading and
skips a model call. A prompt or model change starts afresh.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import time
from pathlib import Path

from talent_angels.memory.paths import db_path

#: A reading is reused for this long; the taxonomy and prompt rarely change.
PLAN_TTL_SECONDS = 30 * 24 * 3600

_SPACES = re.compile(r"\s+")


def _connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS plan_cache (
            key TEXT PRIMARY KEY,
            plan TEXT NOT NULL,
            created_at REAL NOT NULL
        )
        """
    )
    return conn


def plan_key(question: str, profile: str | None, *, prompt: str, model: str) -> str:
    """Same words (case, spacing and end punctuation aside), same profile, same prompt."""
    words = _SPACES.sub(" ", question.casefold()).strip().rstrip("?.!")
    raw = json.dumps([words, profile or "", hashlib.sha256(prompt.encode()).hexdigest(), model])
    return hashlib.sha256(raw.encode()).hexdigest()


def get_plan(key: str, *, path: Path | None = None) -> str | None:
    """The stored planner JSON for ``key``, or None. Never raises."""
    try:
        with _connect(path or db_path()) as conn:
            row = conn.execute(
                "SELECT plan, created_at FROM plan_cache WHERE key = ?", (key,)
            ).fetchone()
    except (sqlite3.Error, OSError):
        return None
    if row is None or time.time() - float(row[1]) > PLAN_TTL_SECONDS:
        return None
    return str(row[0])


def set_plan(key: str, plan_json: str, *, path: Path | None = None) -> None:
    """Store a validated planner JSON. A write failure is invisible by design."""
    try:
        with _connect(path or db_path()) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO plan_cache (key, plan, created_at) VALUES (?, ?, ?)",
                (key, plan_json, time.time()),
            )
    except (sqlite3.Error, OSError):
        pass


def clear_plans(*, path: Path | None = None) -> int:
    """Forget every stored reading (they hold the user's words). Returns the count."""
    try:
        with _connect(path or db_path()) as conn:
            return int(conn.execute("DELETE FROM plan_cache").rowcount or 0)
    except (sqlite3.Error, OSError):
        return 0
