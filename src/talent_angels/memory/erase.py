"""Erase memory at three scopes, so "forget me" is a choice with a name.

One owner for what forgetting means, so the TUI commands, the API edge, and any
future surface cannot disagree about what a given command actually deletes.
Before this module the logic was inlined in ``session.kernel`` and covered only
the two human-readable files — the episode index in ``memory.db`` survived a
reset, so the questions a person asked and the node ids cited on their behalf
outlived the "Memory and session cleared" message.

Three scopes, because "forget this conversation" and "forget me" are different
requests and collapsing them into one command is how people end up surprised:

- ``erase_session`` — this conversation only. Transcript, bindings, pending
  choices, the session's own run-log and query details. The profile and the
  long-term history survive, so the agent still knows the user afterwards.
- ``erase_person`` — everything tied to *who the user is*. ``USER.md``,
  ``MEMORY.md``, and every recorded episode.
- ``erase_all`` — both of the above, plus any run-log or query-details store
  configured outside the session directory, and a ``VACUUM`` so the bytes are
  actually gone from the file rather than merely unlinked.

Stores, and why each is in or out:

- ``USER.md``   — the confirmed user profile. Deleted by ``erase_person``.
- ``MEMORY.md`` — agent notes about how to serve this person. Deleted by
  ``erase_person``.
- ``episodes``  — one typed record per past turn. Deleted by ``erase_person``;
  this is personal history, not telemetry.
- ``transcript.jsonl`` / ``binding.json`` / per-session ``runlog.jsonl`` and
  query details — the raw conversation. Deleted by ``erase_session``. This is
  the largest store of the user's own words and the easiest one to forget: a
  plain ``DELETE FROM episodes`` leaves the text sitting in SQLite's free
  pages, readable with ``strings``, which is why the byte-level clean matters.
- ``checkpoints.db`` — LangGraph's per-thread state for the API, holding the
  question, the plan and the cited nodes of every turn on the thread. Deleted
  by ``erase_all``, and **not** by ``erase_session``: one file holds every
  thread, so unlinking it for one conversation would silently take the others'
  state with it. A session-scoped delete would have to be a per-thread row
  delete, which is a different change with its own free-page problem to solve.
- ``neighbor_cache`` — **always kept**. Rows are keyed by ``(node_id,
  rel_types)`` and carry no user dimension: derived taxonomy answers, identical
  for everyone, with a 24h TTL. There is nothing personal to erase, and
  dropping it would only make the next turn slower.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path

from talent_angels.memory.episodes import clear_episodes, vacuum_db
from talent_angels.memory.paths import default_runlog_path, memory_md, user_md

# Files a session directory is made of. Listed so a count can be reported
# honestly rather than "deleted something".
_SESSION_FILES = ("meta.json", "transcript.jsonl", "binding.json", "runlog.jsonl")


@dataclass(frozen=True, slots=True)
class EraseResult:
    """What an erase actually removed, so no caller can claim more than it did."""

    files_deleted: int = 0
    episodes_deleted: int = 0
    session_files_deleted: int = 0
    extra_stores_removed: int = 0
    checkpoint_files_removed: int = 0
    vacuumed: bool = False
    kept: tuple[str, ...] = field(default_factory=tuple)


def _unlink(path: Path) -> int:
    """Delete one file if present. Returns 1 if it went, 0 if it wasn't there."""
    if path.is_file():
        path.unlink()
        return 1
    return 0


def _purge_dir(path: Path) -> int:
    """Delete a whole directory tree if present. Returns files removed."""
    if not path.is_dir():
        return 0
    removed = sum(1 for entry in path.rglob("*") if entry.is_file())
    shutil.rmtree(path)
    return removed


def _purge_checkpoint_store() -> int:
    """Delete LangGraph's checkpoint database and its SQLite sidecars.

    **The sidecars are the point.** ``SqliteSaver`` opens the database in WAL
    mode, so a checkpoint written since the last fold-back has not reached
    ``checkpoints.db`` at all — it is sitting in ``checkpoints.db-wal``, which
    is a separate file with the same name plus a suffix. Measured while writing
    this: after one turn, none of the user's question was in ``checkpoints.db``
    and all of it was in ``checkpoints.db-wal``. Unlinking only the database
    would report "Forgotten." over a file ``strings`` still reads the name out
    of, which is the exact defect ``test_erase_scopes.py`` exists to prevent,
    one store over.

    The path comes from ``assistant.graph.checkpoint_db_path`` — the writer's
    own resolver — rather than being rebuilt here, so the eraser and the saver
    cannot disagree about where the file is. See the module docstring for what
    that disagreement cost the last time.

    Returns the number of files removed, so the receipt agrees with the disk.
    """
    from talent_angels.assistant.checkpoint import release
    from talent_angels.assistant.graph import checkpoint_db_path

    db = checkpoint_db_path()
    removed = sum(_unlink(path) for path in (db, _sidecar(db, "-wal"), _sidecar(db, "-shm")))
    # The process-wide saver still holds a connection to the unlinked inode.
    # Close it so the next conversation opens a fresh file instead of writing
    # into one nobody can reach (and nobody can erase).
    release(db)
    return removed


def _sidecar(db: Path, suffix: str) -> Path:
    return db.with_name(db.name + suffix)


def _purge_session_dir(session_dir: Path) -> int:
    """Delete every file in one session directory. Returns files removed.

    Everything under the directory goes, not only the four known files: a
    session accumulates query details and other nested stores, and leaving any
    of them behind would mean reporting a fresh conversation while a fragment
    of the old one is still on disk.

    The directory itself goes last, so `/resume` stops offering a session with
    nothing in it.
    """
    if not session_dir.is_dir():
        return 0
    removed = 0
    for name in _SESSION_FILES:
        removed += _unlink(session_dir / name)
    # Anything else the session accumulated (query details, nested dirs).
    for leftover in sorted(session_dir.rglob("*"), reverse=True):
        if leftover.is_file():
            removed += _unlink(leftover)
        elif leftover.is_dir():
            shutil.rmtree(leftover, ignore_errors=True)
    for leftover_dir in sorted(session_dir.rglob("*"), reverse=True):
        if leftover_dir.is_dir():
            leftover_dir.rmdir()
    session_dir.rmdir()
    return removed


def erase_session(session_dir: Path | None = None) -> EraseResult:
    """Forget this conversation. The profile and episode history survive.

    Deletes the session's own files, so the raw transcript stops existing. The
    caller's in-memory state is reset by the command layer; this only handles
    what is on disk.
    """
    if session_dir is None:
        from talent_angels.session.store import sessions_dir

        session_dir = sessions_dir() / "current"

    removed = _purge_session_dir(session_dir)
    return EraseResult(
        session_files_deleted=removed,
        kept=("your profile", "episode history"),
    )


def erase_person(*, vacuum: bool = True) -> EraseResult:
    """Forget who the user is. The transcript is untouched; ``erase_all`` does both.

    ``vacuum=True`` rewrites the database so deleted episode text is gone from
    the file, not merely unlinked. Without it the strings stay readable in
    free pages — verified, not assumed.
    """
    files_deleted = _unlink(user_md()) + _unlink(memory_md())
    episodes_deleted = clear_episodes()
    vacuumed = vacuum and vacuum_db()
    return EraseResult(
        files_deleted=files_deleted,
        episodes_deleted=episodes_deleted,
        vacuumed=vacuumed,
        kept=("this conversation",),
    )


def erase_all(*, session_dir: Path | None = None) -> EraseResult:
    """Forget everything: every conversation, the profile, the history, the bytes.

    Sweeps **all** session directories, not just the live one. An earlier
    ``/save my-journey`` is a full transcript of the same person's questions, and
    a command that answers "Forgotten." while leaving it on disk is worse than
    the bug this module was written to fix. Verified by probe, not assumed.

    Also removes a run-log or query-details store that lives outside the session
    directory, because those hold the same raw questions and would otherwise
    outlive the promise this command makes. Same for the LangGraph checkpoint
    database, which holds the per-thread question, plan and cited nodes.

    The two paths come from ``runlog.writer.runlog_path()`` and
    ``query_details.details_dir()`` — the writers' own resolvers — and *not* from
    re-reading the environment here. That distinction is the bug this call site
    shipped with: the writers fall back to a default (``./runlog.jsonl`` and
    ``<cwd>/data/local/query-details``) and so, a reader of this function would
    have said, does nothing when the variable is unset — which is the state of
    almost every install. So the writer and the eraser disagreed about where the
    files were, the eraser lost, and ``/reset-all`` answered "Forgotten." with 656
    query dumps of real questions still on disk. One resolver per store, reached
    through the module that owns it; an eraser that re-derives a path is an eraser
    that can be wrong about it silently. ``assistant.graph.checkpoint_db_path``
    is reached the same way, and for the same reason.
    """
    from talent_angels.query_details import details_dir
    from talent_angels.runlog.writer import runlog_path
    from talent_angels.session.store import sessions_dir

    root = sessions_dir()
    swept: set[Path] = set()
    session_removed = _purge_session_dir(session_dir) if session_dir else 0
    if session_dir:
        swept.add(session_dir.resolve())
    if root.is_dir():
        for candidate in sorted(root.iterdir()):
            if candidate.is_dir() and candidate.resolve() not in swept:
                session_removed += _purge_session_dir(candidate)
                swept.add(candidate.resolve())

    person = erase_person()

    # Telemetry pointed at outside any session dir is still the user's words.
    # `runlog_path()` may be relative (`./runlog.jsonl` is the default), so it is
    # resolved against the cwd the writer would have used before it is compared
    # with the swept set — an unresolved `.` is not "inside a swept dir" even when
    # the file is.
    extra = 0
    # The TUI re-points RUNLOG_PATH at the live session, so the home run-log
    # (written by the API and CLI) must be named explicitly as well.
    for runlog in {runlog_path().resolve(), default_runlog_path().resolve()}:
        if runlog.parent not in swept:
            extra += _unlink(runlog)
    details = details_dir().resolve()
    if details not in swept:
        extra += _purge_dir(details)
    # The checkpoint store, plus the SQLite sidecars. Unlinking the database
    # alone is the same defect one store over: `SqliteSaver` opens in WAL mode,
    # so a checkpoint written since the last fold is still sitting in the `-wal`
    # file, and that is where the user's words are. Measured while building this
    # — after one turn, `checkpoints.db` held none of the question and
    # `checkpoints.db-wal` held all of it. The sidecars go with the database.
    #
    # Counted on its own field rather than folded into `extra_stores_removed`,
    # which is reported as "log file(s)" — a database is not a log, and a receipt
    # that misnames what it deleted is not a receipt anyone can check.
    checkpoints = _purge_checkpoint_store()

    return EraseResult(
        files_deleted=person.files_deleted,
        episodes_deleted=person.episodes_deleted,
        session_files_deleted=session_removed,
        extra_stores_removed=extra,
        checkpoint_files_removed=checkpoints,
        vacuumed=person.vacuumed,
        kept=("cached taxonomy answers (identical for everyone)",),
    )


def erase_summary(result: EraseResult) -> str:
    """One honest sentence about what was erased — and what was deliberately not.

    Naming what survived matters as much as naming what went: a mentee who asks
    to be forgotten needs to know whether they actually were.
    """
    removed: list[str] = []
    if result.session_files_deleted:
        removed.append(f"{result.session_files_deleted} session file(s)")
    if result.files_deleted:
        removed.append(f"{result.files_deleted} profile file(s)")
    if result.episodes_deleted:
        removed.append(f"{result.episodes_deleted} recorded turn(s)")
    if result.extra_stores_removed:
        removed.append(f"{result.extra_stores_removed} log file(s)")
    if result.checkpoint_files_removed:
        removed.append(f"{result.checkpoint_files_removed} checkpoint file(s)")

    if not removed:
        head = "Nothing to erase — nothing stored."
    else:
        head = "Erased " + " and ".join(removed) + "."
    if result.vacuumed:
        head += " Storage rewritten, so the text is gone from the file too."
    if result.kept:
        head += f" Kept: {'; '.join(result.kept)}."
    return head
