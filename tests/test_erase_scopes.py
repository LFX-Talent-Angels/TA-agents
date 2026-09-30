"""Forgetting has to actually forget — verified against the bytes on disk.

Found by probing, not by reading: a `DELETE FROM episodes` empties the table
while leaving the user's question text readable in SQLite's free pages. A
`/reset` that reports "cleared" while `strings memory.db` still shows the
person's email is not erasure, it is a reassuring message.

These tests assert at the byte level, because that is the only level at which
the original bug was visible.

**And they assert on tokenised residue, not on the verbatim string** — which is
the one lesson the first version of this file needed. Searching for
`b"Priya Raman"` or `b"priya.raman@example.com"` cannot see a leak that has been
tokenised: FTS5's `unicode61` tokenizer lowercases and de-punctuates whatever it
stores, so `priya` and `raman` survive a `DELETE` even when the capitalised
phrase and the `@`-joined address do not. The markers below are therefore
compared **case-insensitively against the lowercased file**, and the lowercase
token forms come first, because those are the ones that actually leaked.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

import talent_angels.memory.agent_notes as notes
import talent_angels.memory.profile as profile
from talent_angels.memory.episodes import clear_episodes, record_episode, vacuum_db
from talent_angels.runlog.models import ResultSummary, RunLogRecord
from talent_angels.session.kernel import handle_line
from talent_angels.session.models import TranscriptLine
from talent_angels.session.store import new_session, save_session, sessions_dir

PII = "I am Priya Raman, priya.raman@example.com, based in Bangalore"

# The tokenised forms first, and these are the ones that matter. A leak through
# an FTS5 index is lowercased and de-punctuated by the tokenizer, so `priya`
# survived a clear while the capitalised phrase and the address did not — which
# is exactly the shape of marker that makes the suite green over a file that
# still holds the user's name. Matched case-insensitively against the lowercased
# bytes, so `Priya` in the question cannot satisfy the `priya` marker.
TOKEN_MARKERS = (b"bangalore", b"example", b"priya", b"raman")

# The verbatim forms, kept because the plain-table free-page leak is a *different*
# defect and it leaves these behind: it is worth knowing both are covered, and
# worth keeping them distinguishable.
PII_MARKERS = TOKEN_MARKERS + (b"priya raman", b"priya.raman@example.com")


def _boom(*_args: object, **_kwargs: object):  # noqa: ANN202
    raise AssertionError("a reset command must not reach the graph or the model")


def _seed_personal_stores() -> None:
    """Put the same personal details in every store that holds user speech."""
    profile.user_md().write_text(f"STANDING: Priya Raman  [esco:nurse]\nGOAL: {PII}\n")
    notes.memory_md().write_text("- Priya Raman asked about Bangalore roles\n")
    record_episode(
        RunLogRecord(
            run_id="run-pii",
            ts="2026-01-01T00:00:00+00:00",
            suite="esco",
            plan=["locate"],
            question=PII,
            result=ResultSummary(node_ids=["esco:occupation:nurse"], node_labels=["nurse"]),
        )
    )


def _all_bytes_under(root: Path) -> bytes:
    blob = b""
    for path in sorted(root.rglob("*")):
        if path.is_file():
            blob += path.read_bytes()
    return blob


def _roots(memory_home) -> tuple[Path, ...]:
    """Both isolated roots: the memory home and wherever sessions now live."""
    return (memory_home.root, sessions_dir().parent)


def _leaks(roots: Path | tuple[Path, ...]) -> set[str]:
    """Which PII markers are readable in any file under the given roots.

    Case-folded on the file side, so a marker is a marker in any casing. The
    markers are all lowercase, so this catches the capitalised form too and
    cannot miss a tokenised leak behind it.
    """
    if isinstance(roots, Path):
        roots = (roots,)
    blob = b"".join(_all_bytes_under(root) for root in roots).lower()
    return {marker.decode() for marker in PII_MARKERS if marker in blob}


def test_pii_is_present_before_the_reset(memory_home) -> None:
    """Control: the fixtures really do put the PII on disk.

    Without this, every assertion below would also pass on a test that never
    wrote anything — a green suite proving nothing.
    """
    _seed_personal_stores()
    state = new_session()
    state.transcript.append(TranscriptLine(role="user", text=PII, ts="t0"))
    save_session(state)

    assert _leaks(_roots(memory_home)), "fixture failed to write the PII to disk"


def test_reset_all_removes_the_pii_from_the_bytes_on_disk(memory_home) -> None:
    """The headline guarantee: after `/reset-all`, the text is in no file at all.

    Checks raw bytes rather than table contents, which is the whole point — the
    table was already empty in the version of this bug that shipped.
    """
    _seed_personal_stores()
    state = new_session()
    state.transcript.append(TranscriptLine(role="user", text=PII, ts="t0"))
    save_session(state)
    assert _leaks(_roots(memory_home))

    handle_line(state, "/reset-all", runner=_boom)

    still_there = _leaks(_roots(memory_home))
    assert not still_there, (
        f"{sorted(still_there)} still readable in a file after /reset-all — "
        f"erasure is cosmetic at the byte level"
    )


def _seed_default_location_stores() -> tuple[Path, Path]:
    """Write the two stores where the *writers* put them when nothing overrides them.

    `query_details.details_dir()` falls back to `<home>/query-details`
    and `runlog.writer.runlog_path()` to `<home>/runlog.jsonl`, so those are the
    paths to seed — not paths this test invented. That is the whole point: the
    bug was an eraser resolving a *different* location than the writer did, and
    a test that seeds its own invented path cannot see that.
    """
    from talent_angels.query_details import details_dir
    from talent_angels.runlog.writer import runlog_path

    details = details_dir()
    details.mkdir(parents=True, exist_ok=True)
    # One turn is a `.md` dump plus a `.json` sidecar, so both are seeded: an
    # eraser that unlinks one shape and misses the other is half-fixed.
    (details / "run-pii.md").write_text(f"# run-pii\n\n- **question:** {PII}\n")
    (details / "run-pii.json").write_text(f'{{"question": "{PII}"}}')
    runlog = runlog_path()
    runlog.write_text(
        RunLogRecord(
            run_id="run-pii",
            ts="2026-01-01T00:00:00+00:00",
            suite="esco",
            plan=["locate"],
            question=PII,
            result=ResultSummary(node_ids=["esco:occupation:nurse"], node_labels=["nurse"]),
        ).model_dump_json()
        + "\n"
    )
    return details, runlog


def _unconfigured_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A cwd that behaves like a fresh install: no `RUNLOG_PATH`, no details dir.

    `tests/conftest.py` sets both variables for every test, which is what made
    the broken branch untestable — the env-override path is the *only* path the
    suite ever exercised, and it is the path production never takes. Deleting
    them here is the missing half of that fixture, and `chdir` is what makes the
    defaults land in `tmp_path` instead of in the developer's repo, so a
    regression here cannot reach the real `data/local/`.
    """
    monkeypatch.delenv("RUNLOG_PATH", raising=False)
    monkeypatch.delenv("QUERY_DETAILS_DIR", raising=False)
    # The defaults now live under the memory home (not the cwd), so the home is
    # the sandbox; chdir as well so a cwd-relative regression lands here too.
    monkeypatch.setenv("TA_AGENTS_HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _file_count(root: Path) -> int:
    return sum(1 for entry in root.rglob("*") if entry.is_file()) if root.is_dir() else 0


def test_reset_all_removes_the_stores_at_their_default_locations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No env override configured — which is how almost every install runs.

    The eraser used to read `RUNLOG_PATH`/`QUERY_DETAILS_DIR` directly and treat
    "unset" as "there is nothing there". The writers default instead, so the two
    disagreed about where the files were and the eraser lost: 656 query-detail
    files with real questions sat in `data/local/query-details` while the command
    answered "Forgotten." Asserted through `erase_all`, because that is where the
    path is resolved and the count is returned.
    """
    from talent_angels.memory.erase import erase_all

    cwd = _unconfigured_home(tmp_path, monkeypatch)
    details, runlog = _seed_default_location_stores()
    assert details == cwd / "query-details", "seeded the wrong place"

    result = erase_all()

    assert not details.exists(), "the default query-details store survived a full forget"
    assert not runlog.exists(), "the default run log survived a full forget"
    assert result.extra_stores_removed == 3, (
        f"erased the files but did not count them: {result!r} — the message a user "
        f"is told has to agree with the disk"
    )
    assert not _leaks(cwd), "the user's own words are still readable under the temp cwd"


def test_the_reset_all_message_reports_the_default_logs_it_deleted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Through the real command, so the sentence the user reads is the one pinned.

    A purge that deletes three files and reports only the session and the
    episodes is the same defect one layer up: the user cannot tell what the
    promise covered.
    """
    _unconfigured_home(tmp_path, monkeypatch)
    _seed_default_location_stores()

    reply = handle_line(new_session(), "/reset-all", runner=_boom)

    assert "3 log file(s)" in reply.text, f"the message hid the deleted logs: {reply.text!r}"
    assert not _leaks(tmp_path)


def test_a_store_the_session_sweep_already_took_is_not_counted_twice(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The TUI's own configuration: a store configured *inside* the session dir.

    `tui/app.py` repoints `RUNLOG_PATH` at the live session directory on every
    `/reset`, so the run log is usually already covered by the session sweep. This
    pins the observable contract for that shape — everything under the session dir
    goes, and the receipt counts it once, as session files.

    The `swept` guard in `erase_all` is **not** what makes this pass, and this
    test does not pretend otherwise. The sweep runs first, so by the time the
    extra-store branch is reached the files are already gone and `_unlink` /
    `_purge_dir` return 0 with or without the guard — verified by running the
    module with the guard deleted, which produces a byte-identical
    `EraseResult`. The guard is defence in depth against a future ordering
    change, not a live behaviour, and there is no assertion that could tell the
    two apart. Kept as-is for that reason; a test written to "pin" it would be
    pinning nothing.
    """
    from talent_angels.memory.erase import erase_all, erase_summary

    _unconfigured_home(tmp_path, monkeypatch)
    state = new_session()
    save_session(state, name="my-journey")
    session = sessions_dir() / "my-journey"
    runlog = session / "runlog.jsonl"
    monkeypatch.setenv("RUNLOG_PATH", str(runlog))
    monkeypatch.setenv("QUERY_DETAILS_DIR", str(session / "query-details"))
    runlog.write_text('{"run_id": "run-pii", "question": "' + PII + '"}\n')
    (session / "query-details").mkdir()
    (session / "query-details" / "turn-1.json").write_text(f'{{"question": "{PII}"}}')
    # Counted, not asserted as a literal: `save_session` adds its own files, and
    # what matters is that the sweep's receipt covers the session *and* the
    # nothing-extra is claimed on top of it.
    in_session = _file_count(session)

    result = erase_all(session_dir=session)

    assert not session.exists()
    assert result.session_files_deleted == in_session, f"the sweep undercounted: {result!r}"
    assert result.extra_stores_removed == 0, (
        f"a store the sweep already removed was counted again: {result!r}"
    )
    assert "log file(s)" not in erase_summary(result), (
        f"the message claims a log delete that never happened: {erase_summary(result)!r}"
    )


def test_reset_all_from_a_temp_cwd_erases_nothing_outside_it(
    tmp_path: Path,
    tmp_path_factory: pytest.TempPathFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The safety interlock: this suite must not be able to touch the real repo.

    A forget command pointed at a developer's real `data/local/` would destroy 656
    question dumps the first time anyone ran the suite. Asserted by recording
    every path the eraser actually reaches rather than by inspecting the repo's
    directory afterwards — the second version passes vacuously anywhere the
    checkout has no `data/`, which is most CI runners and every `/tmp` copy of
    this tree.

    The bound is pytest's basetemp, not this test's own `tmp_path`. `erase_all`
    legitimately reaches three separate roots — the cwd-derived default stores
    (under `tmp_path`), the profile files (under the `memory_home` fixture) and
    the session directories (under the `TA_SESSIONS_DIR` fixture) — and only the
    first is reachable from `tmp_path`. Asserting against `tmp_path` alone would
    fail on `USER.md` and quietly train a reader to weaken the check; the thing
    being protected is the *repository*, and basetemp is the smallest root that
    provably contains every path the fixtures hand out.
    """
    from talent_angels.memory import erase

    touched: list[Path] = []
    real_unlink, real_purge = erase._unlink, erase._purge_dir
    monkeypatch.setattr(erase, "_unlink", lambda path: touched.append(path) or real_unlink(path))
    monkeypatch.setattr(erase, "_purge_dir", lambda path: touched.append(path) or real_purge(path))

    repo = Path(__file__).resolve().parents[1]
    repo_details = repo / "data" / "local" / "query-details"
    _unconfigured_home(tmp_path, monkeypatch)
    _seed_default_location_stores()
    before = _file_count(repo_details)

    erase.erase_all()

    sandbox = tmp_path_factory.getbasetemp().resolve()
    assert touched, "nothing was erased, so the interlock proved nothing"
    for path in touched:
        assert path.is_relative_to(sandbox), (
            f"the erase reached {path}, which is outside pytest's temp root {sandbox}"
        )
        assert not path.resolve().is_relative_to(repo), (
            f"the erase reached into the checkout itself: {path}"
        )
    # The cwd-derived defaults are what the bug was about, so they must be among
    # the paths reached — otherwise the bound above passes without the fix.
    assert tmp_path.resolve() in {p.resolve().parents[0] for p in touched}, (
        "the default-location stores were never reached, so this test no longer "
        "covers the path it was written for"
    )
    assert _file_count(repo_details) == before, "a test erased the real query-details store"


def test_reset_all_also_erases_earlier_saved_sessions(memory_home) -> None:
    """A `/save`d session is the same person's questions, so a full purge takes it too.

    Found by probe: `/reset-all` originally swept only the live session, and
    answered "Forgotten." while a `/save my-journey` transcript sat untouched.
    """
    _seed_personal_stores()
    old = new_session()
    old.transcript.append(TranscriptLine(role="user", text=PII, ts="t0"))
    save_session(old, name="my-journey")
    assert (sessions_dir() / "my-journey" / "transcript.jsonl").is_file()

    handle_line(new_session(), "/reset-all", runner=_boom)

    assert not (sessions_dir() / "my-journey" / "transcript.jsonl").exists()
    assert not _leaks(_roots(memory_home))


def test_reset_keeps_the_profile_and_the_history(memory_home) -> None:
    """`/reset` is session-scoped. It must not quietly become `/reset-all`."""
    _seed_personal_stores()
    from talent_angels.memory.episodes import recent_episodes

    state = new_session()
    state.transcript.append(TranscriptLine(role="user", text=PII, ts="t0"))
    save_session(state)
    old_key = state.name or state.session_id

    reply = handle_line(state, "/reset", runner=_boom)

    assert profile.user_md().exists(), "/reset must not forget who the user is"
    assert notes.memory_md().exists()
    assert len(recent_episodes()) == 1, "/reset must not erase the long-term history"
    # The conversation itself did go.
    assert not (sessions_dir() / old_key / "transcript.jsonl").exists()
    assert "kept" in reply.text.lower(), (
        "the message must say what survived, or a mentee asking to be forgotten "
        "will believe they were"
    )


def test_the_two_scopes_differ_in_exactly_one_way(memory_home) -> None:
    """The split is the feature; pin both halves of it in one place."""
    from talent_angels.memory.episodes import recent_episodes

    _seed_personal_stores()
    state = new_session()
    save_session(state, name="prior")

    handle_line(state, "/reset", runner=_boom)
    assert profile.user_md().exists() and len(recent_episodes()) == 1
    assert not (sessions_dir() / "prior" / "transcript.jsonl").exists()

    handle_line(state, "/reset-all", runner=_boom)
    assert not profile.user_md().exists()
    assert recent_episodes() == []


def test_both_commands_are_idempotent_and_honest_when_there_is_nothing_to_erase(
    memory_home,
) -> None:
    """Erasing an empty home must not raise, and must not claim a deletion."""
    state = new_session()
    first = handle_line(state, "/reset-all", runner=_boom)
    second = handle_line(state, "/reset-all", runner=_boom)

    assert "nothing to erase" in first.text.lower()
    assert "nothing to erase" in second.text.lower(), "a second erase must not invent work"
    assert "1 recorded turn" not in second.text
    assert "1 profile file" not in second.text


def test_clearing_episodes_does_not_leave_the_text_in_free_pages(memory_home) -> None:
    """`PRAGMA secure_delete=ON`, tested by its effect rather than by its spelling.

    A plain `DELETE` leaves the row's bytes in the file; the pragma makes the
    engine overwrite them. Only the observable difference matters, so the
    assertion reads ``memory.db`` itself rather than any store's contents.
    """
    _seed_personal_stores()
    db = memory_home.db
    assert PII.encode() in db.read_bytes(), "precondition: the question is in the file"

    assert clear_episodes() == 1
    assert PII.encode() not in db.read_bytes(), (
        "the question text survived a DELETE in SQLite's free pages"
    )
    # The table really is empty too — the byte check is not passing by accident.
    conn = sqlite3.connect(db)
    try:
        assert conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0] == 0
    finally:
        conn.close()


def test_vacuum_db_cleans_a_file_written_before_secure_delete_existed(memory_home) -> None:
    """`PRAGMA secure_delete=ON` only protects future writes.

    A database that already holds freed pages keeps them until the file is
    rewritten, so `vacuum_db` has to exist for anyone upgrading in place.
    Simulated by writing a file the old way, then asking the new code to clean it.
    """
    db = memory_home.db
    conn = sqlite3.connect(db)  # deliberately without the pragma
    try:
        conn.execute("CREATE TABLE legacy (question TEXT)")
        conn.execute("INSERT INTO legacy VALUES (?)", (PII,))
        conn.commit()
        conn.execute("DELETE FROM legacy")
        conn.commit()
    finally:
        conn.close()
    assert PII.encode() in db.read_bytes(), "precondition: the text is still in the file"

    assert vacuum_db() is True
    assert PII.encode() not in db.read_bytes()

    assert vacuum_db() is True  # idempotent


def test_vacuum_db_reports_honestly_when_there_is_no_database(memory_home) -> None:
    """A vacuum with nothing to vacuum returns False rather than raising or lying."""
    missing = memory_home.db.parent / "nope.db"
    assert vacuum_db(db_path=missing) is False, "no file means nothing was rewritten"

    record_episode(
        RunLogRecord(
            run_id="run-1",
            ts="2026-01-01T00:00:00+00:00",
            suite="esco",
            plan=["locate"],
            question="what skills does a nurse need?",
            result=ResultSummary(node_ids=["esco:occupation:nurse"], node_labels=["nurse"]),
        )
    )
    assert vacuum_db() is True


def test_forgetting_clears_the_recall_index_too(memory_home) -> None:
    """The lexical index is a second copy of the user's words.

    `/reset-all` that empties `episodes` but leaves `episodes_fts` behind would
    report success while every past question is still retrievable — the same
    class of bug as the SQLite free pages this suite already covers. Asserted
    through the real command rather than through `clear_episodes`, because the
    command is what a user is actually told worked.
    """
    from talent_angels.memory.fts_retriever import Fts5EpisodeRetriever

    record_episode(
        RunLogRecord(
            run_id="run-1",
            ts="2026-01-01T00:00:00+00:00",
            suite="esco",
            plan=["locate"],
            question=PII,
            result=ResultSummary(node_ids=["esco:occupation:1"], node_labels=["nurse"]),
        )
    )
    assert Fts5EpisodeRetriever().search("Priya") != [], "precondition: indexed"

    handle_line(new_session(), "/reset-all", runner=_boom)

    assert Fts5EpisodeRetriever().search("Priya") == []
    assert PII.encode() not in memory_home.db.read_bytes()


def test_an_erased_session_directory_does_not_linger(memory_home) -> None:
    """The empty directory must go, or `/resume` keeps offering a blank session."""
    _seed_personal_stores()
    state = new_session()
    state.transcript.append(TranscriptLine(role="user", text=PII, ts="t0"))
    save_session(state, name="my-journey")
    assert (sessions_dir() / "my-journey").is_dir()

    handle_line(state, "/reset", runner=_boom)

    assert not (sessions_dir() / "my-journey").exists(), (
        "an empty session directory left behind will be offered by /resume"
    )


def test_the_whole_session_directory_goes_including_nested_stores(memory_home) -> None:
    """Not just the four known files — the entire session directory.

    A session accumulates query details and other nested files. Leaving any of
    them behind would mean `/reset` reports a fresh conversation while a
    fragment of the old one is still on disk, which is the same class of bug as
    the episode rows that used to survive.
    """
    _seed_personal_stores()
    state = new_session()
    state.transcript.append(TranscriptLine(role="user", text=PII, ts="t0"))
    save_session(state, name="my-journey")
    nested = sessions_dir() / "my-journey" / "query-details" / "turn-1.json"
    nested.parent.mkdir(parents=True, exist_ok=True)
    nested.write_text(f'{{"question": "{PII}"}}')

    handle_line(state, "/reset", runner=_boom)

    assert not (sessions_dir() / "my-journey").exists(), (
        "the session directory and everything under it must be gone"
    )
    # Scoped to the session tree: `/reset` deliberately leaves the profile and
    # the episode history in place, so they are expected to still hold the PII.
    assert not _leaks(sessions_dir().parent)
