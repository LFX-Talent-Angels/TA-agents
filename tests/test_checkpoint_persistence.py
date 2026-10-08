"""The checkpointer has to outlive the process that wrote it.

`MemorySaver` — what the graph used before this — keeps LangGraph's internal
state in a dict inside the running process. The graph is rebuilt on every turn
and the process is restarted on every deploy, so that state was unreachable a
second after it was written: `api/app.py` passes a `thread_id` on every
request, and every one of those threads started blank.

So the interesting assertion is not "a checkpointer exists". It is that state
written through one saver is readable through a **different** saver object
against the same file, which is what a process restart looks like. A test that
writes and reads back through one in-memory instance passes against
`MemorySaver` and proves nothing; these do not.

Erasure is here too, and it is here at the *byte* level for the reason the
rest of `test_erase_scopes.py` is: the store this adds holds the user's
questions, and a sweep that reports "Forgotten." over a file that still spells
out their name is not a sweep. The `-wal` sidecar in particular is where the
text actually lands — measured, not assumed — so an erase that unlinks only
`checkpoints.db` leaves the conversation on disk.
"""

from __future__ import annotations

import gc
from pathlib import Path

import pytest
from langgraph.checkpoint.sqlite import SqliteSaver

from talent_angels.assistant.graph import build_graph, checkpoint_db_path
from talent_angels.llm.stub_client import StubLLMClient
from talent_angels.session.store import new_session
from tests.fakes.taxonomy import FakeCandidate, FakeToolResult
from tests.test_assistant import FakeSuite, _occupation_node

PII = "I am Priya Raman, priya.raman@example.com, based in Bangalore"

# Matched case-insensitively against the lowercased file bytes. The tokenised
# forms come first for the same reason as in test_erase_scopes: a marker that
# only matches the capitalised phrase cannot see a leak that has been
# lowercased on its way to disk, and a green suite over a file that still holds
# the user's name is the exact failure this file exists to prevent.
TOKEN_MARKERS = (b"bangalore", b"example", b"priya", b"raman")
PII_MARKERS = TOKEN_MARKERS + (b"priya raman", b"priya.raman@example.com")


def _suite() -> FakeSuite:
    node = _occupation_node()
    return FakeSuite(
        FakeToolResult(
            candidates=[FakeCandidate(node=node, confidence=0.95, method="exact_pref")],
            nodes=[node],
            evidence=["esco:search:exact_pref:software developer"],
        )
    )


def _leaks(root: Path) -> set[str]:
    """Which PII markers are readable in any file under ``root``."""
    if not root.is_dir():
        return set()
    blob = b"".join(path.read_bytes() for path in sorted(root.rglob("*")) if path.is_file())
    blob = blob.lower()
    return {marker.decode() for marker in PII_MARKERS if marker in blob}


def _thread(thread_id: str = "thread-1") -> dict[str, dict[str, str]]:
    return {"configurable": {"thread_id": thread_id}}


# --- the defect this file exists for ---------------------------------------


def test_state_written_by_one_graph_is_readable_by_a_later_one() -> None:
    """The headline: a second graph, built after the first is gone, sees the turn.

    Two separate `build_graph` calls means two separate savers means two
    separate SQLite connections — the closest a test gets to a process
    restart. `gc.collect()` in between drops the first graph (and with it the
    connection it held) so the read cannot be answered out of an object that
    is still in memory. Against `MemorySaver` this fails, which is the point.
    """
    first = build_graph(
        suite=_suite(),
        llm_client=StubLLMClient(),
        suite_name="esco",
        answer_mode="structured",
        thread_id="thread-1",
    )
    assert isinstance(first.checkpointer, SqliteSaver), (
        "the graph is still holding its state in process memory, so nothing "
        "below this line can mean anything"
    )
    first.invoke({"question": PII}, _thread())

    del first
    gc.collect()

    restarted = build_graph(
        suite=_suite(),
        llm_client=StubLLMClient(),
        suite_name="esco",
        answer_mode="structured",
        thread_id="thread-1",
    )
    snapshot = restarted.get_state(_thread())

    assert snapshot.values.get("question") == PII, (
        "the thread came back empty after a restart — the checkpoint was never "
        "written to disk, only held in the process that wrote it"
    )


def test_the_state_is_on_disk_and_not_only_in_the_saver(memory_home) -> None:
    """Proves the mechanism rather than the outcome.

    The graph-level test above would also pass against a saver that keeps state
    somewhere durable but wrong (another user's home, a path the erase sweep
    never reaches). This one pins *where* the bytes go, which is what
    `test_erase_scopes` and `/reset-all` depend on, and it asks the store's own
    resolver rather than a path this test invented — a test that seeds its own
    invented path is how the original erase bug stayed green.
    """
    graph = build_graph(
        suite=_suite(),
        llm_client=StubLLMClient(),
        suite_name="esco",
        answer_mode="structured",
        thread_id="thread-1",
    )
    graph.invoke({"question": PII}, _thread())

    db = checkpoint_db_path()
    assert db.exists(), f"no checkpoint database at {db} — nothing survives a restart"
    assert db.parent == memory_home.root, (
        f"the checkpoint store landed in {db.parent}, outside the personal data "
        f"home at {memory_home.root} — it would not be isolated by the test "
        f"fixture or swept by /reset-all"
    )
    assert _leaks(memory_home.root), "precondition: the turn is not readable in the store"


def test_a_different_thread_does_not_inherit_another_thread_state() -> None:
    """Threads are separate conversations; sharing a file must not merge them.

    One database file, many threads. If the thread id were not honoured the
    second session would inherit the first session's question, which is a
    stranger's conversation surfacing in a different conversation — the kind of
    leak that is invisible in a smoke test and obvious to a user.
    """
    first = build_graph(
        suite=_suite(),
        llm_client=StubLLMClient(),
        suite_name="esco",
        answer_mode="structured",
        thread_id="thread-1",
    )
    first.invoke({"question": PII}, _thread("thread-1"))
    del first
    gc.collect()

    other = build_graph(
        suite=_suite(),
        llm_client=StubLLMClient(),
        suite_name="esco",
        answer_mode="structured",
        thread_id="thread-2",
    )

    assert other.get_state(_thread("thread-2")).values.get("question") is None
    assert other.get_state(_thread("thread-1")).values.get("question") == PII, (
        "the original thread lost its state — the second thread read and "
        "overwrote it rather than reading nothing"
    )


# --- the callers that must stay unaffected ----------------------------------


def test_no_thread_id_means_no_checkpointer() -> None:
    """The TUI and the CLI pass no `thread_id`, and must keep not passing one.

    Unchanged behaviour, asserted so it stays unchanged: no checkpointer means
    no SQLite file is created for those callers, and a run that never opted into
    session continuity does not silently acquire a store on disk.
    """
    graph = build_graph(
        suite=_suite(),
        llm_client=StubLLMClient(),
        suite_name="esco",
        answer_mode="structured",
    )

    assert graph.checkpointer is None


def test_a_turn_without_a_thread_still_answers() -> None:
    """The ordinary offline path: a plain turn, answered as before.

    Asserted on the *result*, not on the absence of an exception, so that a
    change which broke the common path in exchange for durable state would fail
    here. The disk check matters too: the TUI and the CLI pass no `thread_id`,
    and must not silently acquire a store on disk.
    """
    graph = build_graph(
        suite=_suite(),
        llm_client=StubLLMClient(),
        suite_name="esco",
        answer_mode="structured",
    )

    final_state = graph.invoke({"question": "software developer"})

    assert final_state["result"].confidence == 0.95
    assert "software developer" in final_state["answer"]
    assert not checkpoint_db_path().exists(), (
        "an unthreaded turn created a checkpoint store — the TUI and CLI never "
        "asked for session continuity and should not be paying for it"
    )


def test_a_second_turn_on_a_thread_accumulates_on_the_first() -> None:
    """What the durable state is actually *for*: continuity within a thread.

    A restarted process that has forgotten the first turn is not a session, it
    is two unrelated turns. The saved state has to come back and be usable, not
    merely present.
    """
    first = build_graph(
        suite=_suite(),
        llm_client=StubLLMClient(),
        suite_name="esco",
        answer_mode="structured",
        thread_id="thread-1",
    )
    first.invoke({"question": "software developer"}, _thread())
    del first
    gc.collect()

    resumed = build_graph(
        suite=_suite(),
        llm_client=StubLLMClient(),
        suite_name="esco",
        answer_mode="structured",
        thread_id="thread-1",
    )
    snapshot = resumed.get_state(_thread())

    assert snapshot.values.get("result") is not None
    assert snapshot.values["result"].confidence == 0.95


# --- erasure ---------------------------------------------------------------


def _unconfigured_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A cwd that behaves like a fresh install: no path overrides set.

    `tests/conftest.py` sets `TA_SESSIONS_DIR`, `RUNLOG_PATH` and
    `QUERY_DETAILS_DIR` for every test, which is what made the original erase
    bug untestable: the override branch is the only branch the suite ever
    exercised, and it is the branch production never takes. Deleting them is
    the missing half of that fixture. `chdir` keeps the resulting defaults
    inside `tmp_path` so a regression cannot reach the real checkout.
    """
    for name in ("TA_SESSIONS_DIR", "RUNLOG_PATH", "QUERY_DETAILS_DIR"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_reset_all_takes_the_checkpoint_store_with_it(
    tmp_path: Path,
    tmp_path_factory: pytest.TempPathFactory,
    monkeypatch: pytest.MonkeyPatch,
    memory_home,
) -> None:
    """/reset-all must not answer "Forgotten." over a live conversation.

    Driven end to end through the real writer rather than by seeding an
    invented path: the file is created by `build_graph(thread_id=...)` doing an
    actual turn, so a test cannot pass while the writer and the eraser disagree
    about where the store lives. That disagreement is the bug this module
    already shipped once, when the eraser resolved an env var and the writer
    resolved a default.
    """
    from talent_angels.memory.erase import erase_all, erase_summary

    _unconfigured_home(tmp_path, monkeypatch)

    graph = build_graph(
        suite=_suite(),
        llm_client=StubLLMClient(),
        suite_name="esco",
        answer_mode="structured",
        thread_id="thread-1",
    )
    graph.invoke({"question": PII}, _thread())
    del graph
    gc.collect()

    assert _leaks(memory_home.root), (
        "precondition: the checkpoint store holds nothing to erase, so this "
        "test would pass over a sweep that missed it entirely"
    )

    result = erase_all()

    assert not _leaks(memory_home.root), (
        "the conversation is still readable in the memory home after a full "
        "forget — the checkpoint database is out of the erase path"
    )
    assert result.checkpoint_files_removed == 1, (
        f"the receipt does not name the store it deleted: {erase_summary(result)!r} — "
        f"a user asking to be forgotten has to be able to tell what the promise covered"
    )
    # The sweep must not have reached outside pytest's temp root, or the test
    # above proves erasure by destroying the developer's real stores.
    assert memory_home.root.resolve().is_relative_to(tmp_path_factory.getbasetemp().resolve())


def test_the_checkpoint_erasure_is_not_just_the_main_database(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    memory_home,
) -> None:
    """The sidecars are where the text is, and unlinking the `.db` misses it.

    `SqliteSaver` opens its database in WAL mode, so a checkpoint written since
    the last fold-back has not reached `checkpoints.db` at all — it is sitting
    in `checkpoints.db-wal`, a separate file. Measured while writing this: with
    the connection open, as it is on any running server, `checkpoints.db` holds
    none of the question and `checkpoints.db-wal` holds all of it. An erase that
    unlinks the database and not the sidecar reports success over a file
    `strings` still reads — byte-for-byte the defect `test_erase_scopes.py` was
    written for, one store over.

    The graph is deliberately **held open** for the duration. Dropping it closes
    the connection, which folds the WAL back into the database and deletes the
    sidecar — so a test that tidies up first would be asserting against a state
    the running server is never in, and would pass over a sweep that only ever
    worked because nothing was open.
    """
    from talent_angels.memory.erase import erase_all, erase_summary

    _unconfigured_home(tmp_path, monkeypatch)
    graph = build_graph(
        suite=_suite(),
        llm_client=StubLLMClient(),
        suite_name="esco",
        answer_mode="structured",
        thread_id="thread-1",
    )
    graph.invoke({"question": PII}, _thread())

    # Preconditions, stated as the facts they are: the text is not in the .db,
    # so unlinking that alone provably leaves it behind.
    db = memory_home.root / "checkpoints.db"
    wal = memory_home.root / "checkpoints.db-wal"
    assert wal.exists(), (
        "precondition: no WAL sidecar on disk, so this test no longer covers "
        "the case it was written for"
    )
    assert PII.split(",")[0].encode().lower() not in db.read_bytes().lower(), (
        "precondition changed: the main database now holds the question, so "
        "unlinking it alone would erase the text and this test would prove nothing"
    )
    assert b"priya" in wal.read_bytes().lower(), "precondition: the sidecar holds nothing"

    result = erase_all()

    assert not _leaks(memory_home.root), (
        "a WAL sidecar survived the sweep — the store is not byte-level erased, "
        "and a live server holds its checkpoint connection open"
    )
    # The connection is still open, so all three files existed: the database and
    # both sidecars. A receipt that claimed one here would be under-reporting.
    assert result.checkpoint_files_removed == 3, (
        f"expected the database and both sidecars, got {result.checkpoint_files_removed}: "
        f"{erase_summary(result)!r}"
    )


def test_a_forget_with_no_checkpoint_store_is_still_honest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Most installs never pass a `thread_id`, so most erases find no store.

    The sweep must not claim a deletion it did not make. An erase that counted
    a checkpoint database that was never there would teach a mentee to
    distrust the receipt, which is the same failure as claiming one too few.
    """
    from talent_angels.memory.erase import erase_all, erase_summary

    _unconfigured_home(tmp_path, monkeypatch)

    result = erase_all()

    assert "nothing to erase" in erase_summary(result).lower()
    assert result.extra_stores_removed == 0
    assert result.checkpoint_files_removed == 0


def test_erase_session_scope_leaves_the_thread_store_alone(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    memory_home,
) -> None:
    """`/reset` is this-conversation-only; it must not become `/reset-all`.

    The checkpoint database holds every thread in one file, so erasing one
    session by unlinking it would silently destroy the other conversations'
    state. The scope split in `memory/erase.py` is the feature; this pins the
    half that is easy to get wrong when a new store is added.
    """
    from talent_angels.memory.erase import erase_session
    from talent_angels.session.store import save_session

    _unconfigured_home(tmp_path, monkeypatch)
    graph = build_graph(
        suite=_suite(),
        llm_client=StubLLMClient(),
        suite_name="esco",
        answer_mode="structured",
        thread_id="thread-1",
    )
    graph.invoke({"question": PII}, _thread())
    del graph
    gc.collect()
    # A real session on disk, so `erase_session` runs against the shape it sees
    # in production rather than an empty directory this test made up.
    session_dir = save_session(new_session(), name="current")
    assert session_dir.is_dir(), "precondition: no session directory to erase"

    erase_session(session_dir)

    assert (memory_home.root / "checkpoints.db").exists(), (
        "/reset deleted the shared checkpoint store, taking every other thread's state with it"
    )
