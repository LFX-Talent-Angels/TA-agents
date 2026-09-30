"""Conversation kernel: rails, numbered bind, bound Connect — no search on pick."""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path

import pytest

from talent_angels.assistant.turn import run_turn
from talent_angels.llm.stub_client import StubLLMClient
from talent_angels.session.models import LastBinding, PendingChoice
from talent_angels.session.store import new_session, save_session, sessions_dir
from talent_angels.suites import SuiteRegistry, SuiteRuntime
from talent_angels.suites.schema import SuiteSchema
from tests.fakes.taxonomy import FakeCandidate, FakeEdge, FakeNode, FakeToolResult

_DEFAULT_SCHEMA = SuiteSchema(
    skill_rel_types=("HAS_SKILL", "USES_SOFTWARE"),
    optional_rel_values=frozenset({"optional", "transferable"}),
    group_rel_type="CLASSIFIED_UNDER",
    group_node_kinds=frozenset({"ISCOGroup", "isco group"}),
)


class RecordingFakeSuite:
    """Returns ambiguous multi-node locate hits and records search texts."""

    @property
    def suite_schema(self) -> SuiteSchema:
        return _DEFAULT_SCHEMA

    def __init__(
        self,
        *,
        ambiguous_nodes: list[FakeNode] | None = None,
        unique_nodes: list[FakeNode] | None = None,
    ) -> None:
        self.searches: list[str] = []
        self.neighbor_ids: list[str] = []
        self.ambiguous_nodes = list(ambiguous_nodes or [])
        self.unique_nodes = list(unique_nodes or [])
        self._nodes_by_id: dict[str, FakeNode] = {
            node.id: node for node in (*self.ambiguous_nodes, *self.unique_nodes)
        }

    def search_nodes(self, text: str, kind: str | None = None) -> FakeToolResult:
        self.searches.append(text)
        key = text.strip().casefold()
        for node in self.unique_nodes:
            if node.label.casefold() == key:
                return self._unique(node)
        if self.ambiguous_nodes:
            return self._ambiguous()
        if self.unique_nodes:
            return self._unique(self.unique_nodes[0])
        node = FakeNode(
            id="test:occupation:1",
            kind="Occupation",
            label=text,
            source="test",
            source_id="test-source-1",
            properties={},
        )
        return self._unique(node)

    def get_neighbors(self, node_id: str, rel_types: list[str] | None = None) -> FakeToolResult:
        self.neighbor_ids.append(node_id)
        center = self._nodes_by_id.get(node_id) or FakeNode(
            node_id, "Occupation", "unknown", "test", "src"
        )
        skill = FakeNode(
            id="test:skill:1",
            kind="Skill",
            label="computer programming",
            source="test",
            source_id="test-skill-1",
            properties={},
        )
        return FakeToolResult(
            nodes=[center, skill],
            edges=[
                FakeEdge(
                    type="HAS_SKILL",
                    from_id=node_id,
                    to_id=skill.id,
                    properties={"relation_type": "essential"},
                )
            ],
            evidence=[f"test:neighbors:{node_id}"],
        )

    def _unique(self, node: FakeNode) -> FakeToolResult:
        self._nodes_by_id[node.id] = node
        return FakeToolResult(
            nodes=[node],
            candidates=[FakeCandidate(node=node, confidence=0.9, method="exact")],
            evidence=["test:search:exact"],
        )

    def _ambiguous(self) -> FakeToolResult:
        nodes = self.ambiguous_nodes
        return FakeToolResult(
            nodes=nodes,
            candidates=[
                FakeCandidate(node=node, confidence=0.7 - 0.1 * i, method="contains")
                for i, node in enumerate(nodes)
            ],
            warnings=["ambiguous"],
            evidence=["test:search:contains"],
        )


def _occ(i: int, label: str) -> FakeNode:
    return FakeNode(
        id=f"test:occupation:{i}",
        kind="Occupation",
        label=label,
        source="test",
        source_id=f"src-{i}",
        properties={},
    )


def _node_ref_occ(label: str, n: int = 1):
    from talent_angels.contracts import NodeRef

    return NodeRef(
        id=f"esco:occupation:{n}",
        suite="esco",
        source="esco",
        source_id=f"src-{n}",
        kind="Occupation",
        pref_label=label,
    )


def _runner_for(suite: RecordingFakeSuite):
    @contextmanager
    def factory():
        yield SuiteRuntime(name="test", suite=suite, health_check=lambda: True)

    registry = SuiteRegistry({"test": factory}, default="test")

    def runner(question: str, *, bound_node=None, bound_nodes=None, force_capability=None):
        with registry.open() as runtime:
            return run_turn(
                suite=runtime.suite,
                suite_name=runtime.name,
                llm_client=StubLLMClient(),
                question=question,
                answer_mode="structured",
                bound_node=bound_node,
                bound_nodes=bound_nodes,
                force_capability=force_capability,
            )

    return runner


def _boom(question: str, **kwargs: object):
    raise AssertionError(f"runner must not be called, got {question!r}")


@pytest.fixture(autouse=True)
def _local_runtime(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("RUNLOG_PATH", str(tmp_path / "runlog.jsonl"))
    monkeypatch.setenv("QUERY_DETAILS_DIR", str(tmp_path / "query-details"))
    monkeypatch.setenv("TA_SESSIONS_DIR", str(tmp_path / "sessions"))
    monkeypatch.setenv("LLM_PROVIDER", "none")
    monkeypatch.setenv("LLM_MODEL", "stub")
    monkeypatch.setenv("ANSWER_MODE", "structured")
    # USER.md / MEMORY.md / memory.db isolation is owned by tests/conftest.py.


def test_numeric_pick_does_not_call_search_nodes(tmp_path: Path) -> None:
    from talent_angels.session.kernel import handle_line

    suite = RecordingFakeSuite(
        ambiguous_nodes=[
            _occ(1, "software developer"),
            _occ(2, "web developer"),
        ]
    )
    runner = _runner_for(suite)
    state = new_session()

    listed = handle_line(state, "developer", runner=runner)
    assert suite.searches == ["developer"]
    assert listed.pending_count == 2
    assert "software developer" in listed.text
    assert "1." in listed.text
    assert listed.source_note == "TEST"
    assert state.binding is None
    software = next(c for c in state.pending if c.node.pref_label == "software developer")

    picked = handle_line(state, str(software.number), runner=runner)
    assert suite.searches == ["developer"]
    assert state.binding is not None
    assert state.binding.node.id == software.node.id
    assert picked.bound_label == "software developer"
    assert picked.pending_count == 2
    assert picked.source_note is None

    follow = handle_line(state, "essential skills", runner=runner)
    assert suite.searches == ["developer"]
    assert software.node.id in suite.neighbor_ids
    assert "computer programming" in follow.text
    assert follow.source_note == "TEST"

    events_path = tmp_path / "sessions" / (state.name or state.session_id) / "picker-events.jsonl"
    assert events_path.is_file()
    event = json.loads(events_path.read_text(encoding="utf-8").splitlines()[0])
    assert event["query"] == "developer"
    assert software.node.id in event["candidates"]
    assert event["chosen_id"] == software.node.id
    assert "timestamp" in event


def test_login_argument_is_refused_without_recording_the_secret() -> None:
    from talent_angels.session.kernel import handle_line

    secret = "sk-or-v1-" + "a" * 40
    state = new_session()

    reply = handle_line(state, f"/login {secret}", runner=_boom)

    assert reply.request_key is False
    assert secret not in reply.text
    assert all(secret not in line.text for line in state.transcript)
    assert state.transcript[0].text == "/login"


def test_model_none_does_not_fetch_the_catalogue(monkeypatch: pytest.MonkeyPatch) -> None:
    import talent_angels.session.kernel as kernel

    def catalogue_must_not_be_called() -> tuple[list[object], str | None]:
        raise AssertionError("stub switching must not need the network catalogue")

    monkeypatch.setattr(kernel, "load_catalogue", catalogue_must_not_be_called)
    state = new_session()

    reply = kernel.handle_line(state, "/model none", runner=_boom)

    assert reply.new_llm_client is not None
    assert reply.new_llm_client.provider == "none"


def test_explicit_model_slug_does_not_fetch_the_catalogue(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import talent_angels.session.kernel as kernel

    def catalogue_must_not_be_called() -> tuple[list[object], str | None]:
        raise AssertionError("explicit model switching must not need the network catalogue")

    selected: list[object] = []
    monkeypatch.setattr(kernel, "load_catalogue", catalogue_must_not_be_called)
    monkeypatch.setattr(
        kernel,
        "apply",
        lambda choice: selected.append(choice) or StubLLMClient(),
    )

    reply = kernel.handle_line(new_session(), "/model google/gemma-4-31b-it:free", runner=_boom)

    assert reply.new_llm_client is not None
    assert len(selected) == 1


def test_hi_does_not_open_the_suite() -> None:
    from talent_angels.session.kernel import handle_line

    runner_calls: list[str] = []

    def runner(question: str, **kwargs: object):
        runner_calls.append(question)
        raise AssertionError("greetings must not run a turn")

    state = new_session()
    reply = handle_line(state, "hi", runner=runner)
    assert runner_calls == []
    assert "here" in reply.text.lower()
    assert reply.source_note is None
    assert [line.role for line in state.transcript] == ["user", "assistant"]


def test_resume_does_not_replay_tools(tmp_path: Path) -> None:
    from talent_angels.session.kernel import handle_line

    saved = new_session()
    saved.binding = LastBinding(node=_node_ref_occ("software developer"))
    save_session(saved, name="demo")

    state = new_session()
    reply = handle_line(state, "/resume demo", runner=_boom)
    assert state.binding is not None
    assert state.binding.node.pref_label == "software developer"
    assert state.session_id == saved.session_id
    assert "demo" in reply.text.lower() or "resumed" in reply.text.lower()
    assert reply.source_note is None


def test_help_advice_quit_and_unknown_skip_runner() -> None:
    from talent_angels.session.copy import ADVICE_REFUSE, HELP_TEXT
    from talent_angels.session.kernel import handle_line

    state = new_session()
    help_reply = handle_line(state, "/help", runner=_boom)
    assert help_reply.text == HELP_TEXT
    assert help_reply.source_note is None

    plain = handle_line(state, "what can you do", runner=_boom)
    assert plain.text == HELP_TEXT

    advice = handle_line(state, "should I learn Python first?", runner=_boom)
    assert advice.text == ADVICE_REFUSE

    unknown = handle_line(state, "/foo", runner=_boom)
    assert "/foo" in unknown.text
    assert unknown.source_note is None

    quit_reply = handle_line(state, "/quit", runner=_boom)
    assert quit_reply.quit is True
    exit_reply = handle_line(state, "/exit", runner=_boom)
    assert exit_reply.quit is True


def test_pick_without_pending_does_not_search() -> None:
    from talent_angels.session.kernel import handle_line

    state = new_session()
    reply = handle_line(state, "1", runner=_boom)
    assert "list" in reply.text.lower()
    assert state.binding is None


def test_unique_locate_sets_binding() -> None:
    from talent_angels.session.kernel import handle_line

    suite = RecordingFakeSuite(unique_nodes=[_occ(9, "nurse")])
    state = new_session()
    reply = handle_line(state, "nurse", runner=_runner_for(suite))
    assert suite.searches == ["nurse"]
    assert state.binding is not None
    assert state.binding.node.id == "test:occupation:9"
    assert reply.bound_label == "nurse"
    assert reply.source_note == "TEST"
    assert reply.pending_count == 0


def test_ambiguous_locate_clears_binding() -> None:
    from talent_angels.session.kernel import handle_line

    suite = RecordingFakeSuite(
        unique_nodes=[_occ(9, "nurse")],
        ambiguous_nodes=[_occ(1, "software developer"), _occ(2, "web developer")],
    )
    runner = _runner_for(suite)
    state = new_session()
    handle_line(state, "nurse", runner=runner)
    assert state.binding is not None

    listed = handle_line(state, "developer", runner=runner)
    assert state.binding is None
    assert listed.pending_count == 2
    assert "I won't pick" in listed.text


def test_save_and_clear_mutate_state(tmp_path: Path) -> None:
    from talent_angels.session.kernel import handle_line

    state = new_session()
    saved = handle_line(state, "/save demo-nurse", runner=_boom)
    assert state.name == "demo-nurse"
    assert "demo-nurse" in saved.text
    assert (tmp_path / "sessions" / "demo-nurse").is_dir()

    from talent_angels.session.models import TranscriptLine

    state.transcript.append(TranscriptLine(role="user", text="keep?", ts="t0"))
    state.binding = LastBinding(node=_node_ref_occ("nurse"))
    # `/clear` is an alias for the session-scoped `/reset`.
    cleared = handle_line(state, "/clear", runner=_boom)
    assert state.binding is None
    assert state.pending == []
    assert state.session_id
    assert "fresh conversation" in cleared.text.lower()


def test_reset_keeps_the_profile_and_erases_the_conversation(memory_home) -> None:
    """`/reset` is session-scoped: forget the conversation, keep who they are.

    The scope split is deliberate. Most turns of `/reset` want a clean slate
    without losing the agent's knowledge of the user, so wiping USER.md and the
    episode index here would be a regression. The full purge moved to
    `/reset-all` (covered separately). What `/reset` *does* own is the
    conversation — including the transcript on disk, which previously survived
    every reset because erase_person() only covered the profile and the table.
    """
    from talent_angels.memory.episodes import recent_episodes, record_episode
    from talent_angels.runlog.models import ResultSummary, RunLogRecord
    from talent_angels.session.kernel import handle_line
    from talent_angels.session.models import TranscriptLine

    memory_home.user_md.write_text("STANDING[esco]: Nurse  [esco:nurse]\n")
    memory_home.memory_md.write_text("- asks short questions\n")
    record_episode(
        RunLogRecord(
            run_id="run-private",
            ts="2026-01-01T00:00:00+00:00",
            suite="esco",
            plan=["locate", "connect"],
            question="what skills does a nurse need?",
            result=ResultSummary(
                node_ids=["esco:occupation:nurse"],
                node_labels=["nurse"],
                warnings=[],
            ),
        )
    )

    state = new_session()
    state.transcript.append(TranscriptLine(role="user", text="?", ts="t0"))
    state.binding = LastBinding(node=_node_ref_occ("nurse"))
    save_session(state)

    reply = handle_line(state, "/reset", runner=_boom)

    # Session state is fresh.
    assert state.binding is None
    assert state.pending == []
    assert "?" not in [line.text for line in state.transcript]

    # The profile and the long-term history deliberately survive.
    assert memory_home.user_md.exists(), "/reset must not forget who the user is"
    assert memory_home.memory_md.exists()
    assert len(recent_episodes()) == 1

    # The transcript is gone from disk, which is the part that used to leak.
    session_dir = sessions_dir() / "current"
    assert not (session_dir / "transcript.jsonl").exists()

    # And the message says what was kept, so nobody is misled about being forgotten.
    assert "kept" in reply.text.lower()
    assert "profile" in reply.text.lower()


def test_reset_all_erases_profile_notes_and_episode_history(memory_home) -> None:
    """`/reset-all` is the full purge: conversation, profile, and history.

    Regression for the original bug: the command inlined the unlink of
    USER.md/MEMORY.md and left memory.db alone, so the person's questions and
    the node ids cited for them outlived the "Memory and session cleared"
    message.
    """
    from talent_angels.memory.episodes import recent_episodes, record_episode
    from talent_angels.runlog.models import ResultSummary, RunLogRecord
    from talent_angels.session.kernel import handle_line
    from talent_angels.session.models import TranscriptLine

    memory_home.user_md.write_text("STANDING[esco]: Nurse  [esco:nurse]\n")
    memory_home.memory_md.write_text("- asks short questions\n")
    record_episode(
        RunLogRecord(
            run_id="run-private",
            ts="2026-01-01T00:00:00+00:00",
            suite="esco",
            plan=["locate", "connect"],
            question="what skills does a nurse need?",
            result=ResultSummary(
                node_ids=["esco:occupation:nurse"],
                node_labels=["nurse"],
                warnings=[],
            ),
        )
    )

    state = new_session()
    state.transcript.append(TranscriptLine(role="user", text="?", ts="t0"))
    state.binding = LastBinding(node=_node_ref_occ("nurse"))
    save_session(state)

    reply = handle_line(state, "/reset-all", runner=_boom)

    assert not memory_home.user_md.exists()
    assert not memory_home.memory_md.exists()
    assert recent_episodes() == []
    assert state.binding is None
    assert state.pending == []
    assert "?" not in [line.text for line in state.transcript]
    assert not (sessions_dir() / "current" / "transcript.jsonl").exists()
    # The message must not claim more than it erased.
    assert "1 recorded turn" in reply.text


def test_picker_main_markdown_omits_node_ids() -> None:
    from talent_angels.session.kernel import handle_line

    suite = RecordingFakeSuite(
        ambiguous_nodes=[
            FakeNode(
                id="esco:occupation:1",
                kind="Occupation",
                label="software developer",
                source="esco",
                source_id="src-1",
                properties={},
            ),
            FakeNode(
                id="esco:occupation:2",
                kind="Occupation",
                label="web developer",
                source="esco",
                source_id="src-2",
                properties={},
            ),
        ]
    )
    listed = handle_line(new_session(), "developer", runner=_runner_for(suite))
    assert "1." in listed.text and "software developer" in listed.text
    assert "id=esco:occupation:" not in listed.text
    assert "id=" not in listed.text.split("source:")[0]


def test_not_found_is_honest_miss_not_suite_identity() -> None:
    from talent_angels.session.kernel import handle_line

    class NotFoundSuite(RecordingFakeSuite):
        def search_nodes(self, text: str, kind: str | None = None) -> FakeToolResult:
            self.searches.append(text)
            return FakeToolResult(
                warnings=["not_found"],
                evidence=[f"test:search:not_found:{text}"],
            )

    suite = NotFoundSuite()
    reply = handle_line(new_session(), "xyzzy-no-such-job", runner=_runner_for(suite))
    assert reply.text == (
        "No node for that phrase with today's search. That's a miss, not a maybe."
    )
    assert "not in ESCO" not in reply.text
    assert "No match found" not in reply.text
    assert reply.source_note == "TEST"


def test_unique_locate_appends_next_step_when_bound() -> None:
    from talent_angels.session.kernel import handle_line

    suite = RecordingFakeSuite(unique_nodes=[_occ(9, "nurse")])
    reply = handle_line(new_session(), "nurse", runner=_runner_for(suite))
    assert "nurse" in reply.text.lower()
    assert "You can ask for essential skills, optional skills, or pick another number." in (
        reply.text
    )


def test_unique_locate_phrasing_omits_raw_id_line() -> None:
    from talent_angels.llm.protocol import LLMResult, LLMUsage
    from talent_angels.session.kernel import handle_line

    class Voice:
        provider = "litellm"

        def complete(self, messages: object, **_kwargs: object) -> LLMResult:
            return LLMResult(
                text="Nurse responsible for general care implements nursing care.",
                provider=self.provider,
                model="test",
                usage=LLMUsage(),
            )

    suite = RecordingFakeSuite(unique_nodes=[_occ(9, "nurse")])
    reply = handle_line(
        new_session(),
        "nurse",
        runner=_runner_for(suite),
        llm_client=Voice(),
    )
    assert "implements nursing care" in reply.text
    assert "id=" not in reply.text
    assert "You can ask for essential skills, optional skills, or pick another number." in (
        reply.text
    )


def test_connect_truncation_points_at_query_details() -> None:
    """Truncation now kicks in past CONNECT_PREVIEW_CAP (GAP F raised it from 5)."""
    from talent_angels.session.kernel import handle_line

    class ManyNeighborSuite(RecordingFakeSuite):
        def get_neighbors(self, node_id: str, rel_types: list[str] | None = None) -> FakeToolResult:
            self.neighbor_ids.append(node_id)
            center = self._nodes_by_id.get(node_id) or FakeNode(
                node_id, "Occupation", "unknown", "test", "src"
            )
            skills = [
                FakeNode(
                    id=f"test:skill:{i}",
                    kind="Skill",
                    label=f"skill {i}",
                    source="test",
                    source_id=f"test-skill-{i}",
                    properties={},
                )
                for i in range(1, 22)
            ]
            return FakeToolResult(
                nodes=[center, *skills],
                edges=[
                    FakeEdge(
                        type="HAS_SKILL",
                        from_id=node_id,
                        to_id=skill.id,
                        properties={"relation_type": "essential"},
                    )
                    for skill in skills
                ],
                evidence=[f"test:neighbors:{node_id}"],
            )

    suite = ManyNeighborSuite(unique_nodes=[_occ(1, "software developer")])
    state = new_session()
    runner = _runner_for(suite)
    handle_line(state, "software developer", runner=runner)
    follow = handle_line(state, "essential skills", runner=runner)
    assert "full list is in query details" in follow.text
    assert "result payload" not in follow.text
    assert "You can ask for essential skills, optional skills, or pick another number." in (
        follow.text
    )


def _skills_suite(count: int) -> RecordingFakeSuite:
    class ManyNeighborSuite(RecordingFakeSuite):
        def get_neighbors(self, node_id: str, rel_types: list[str] | None = None) -> FakeToolResult:
            self.neighbor_ids.append(node_id)
            center = self._nodes_by_id.get(node_id) or FakeNode(
                node_id, "Occupation", "unknown", "test", "src"
            )
            skills = [
                FakeNode(
                    id=f"test:skill:{i}",
                    kind="Skill",
                    label=f"skill {i}",
                    source="test",
                    source_id=f"test-skill-{i}",
                    properties={},
                )
                for i in range(1, count + 1)
            ]
            return FakeToolResult(
                nodes=[center, *skills],
                edges=[
                    FakeEdge(
                        type="HAS_SKILL",
                        from_id=node_id,
                        to_id=skill.id,
                        properties={"relation_type": "essential"},
                    )
                    for skill in skills
                ],
                evidence=[f"test:neighbors:{node_id}"],
            )

    return ManyNeighborSuite(unique_nodes=[_occ(1, "software developer")])


def test_connect_answer_names_show_more_skills_when_truncated() -> None:
    """GAP F: the hint must name the exact follow-up words that already work."""
    from talent_angels.session.kernel import handle_line

    suite = _skills_suite(21)  # past CONNECT_PREVIEW_CAP (15)
    state = new_session()
    runner = _runner_for(suite)
    handle_line(state, "software developer", runner=runner)
    follow = handle_line(state, "essential skills", runner=runner)

    assert 'Ask "show more skills" to see the rest.' in follow.text
    # And the words it names must actually work, end to end.
    listed = handle_line(state, "show more skills", runner=runner)
    assert "skill 21" in listed.text


def test_connect_answer_has_no_more_hint_when_everything_is_already_shown() -> None:
    """Fewer skills than the preview cap: nothing truncated, nothing to hint at."""
    from talent_angels.session.kernel import handle_line

    suite = _skills_suite(4)  # under CONNECT_PREVIEW_CAP (15)
    state = new_session()
    runner = _runner_for(suite)
    handle_line(state, "software developer", runner=runner)
    follow = handle_line(state, "essential skills", runner=runner)

    assert "show more skills" not in follow.text.lower()


def test_complete_list_followup_uses_last_connect_without_search() -> None:
    from talent_angels.session.kernel import handle_line

    class ManyNeighborSuite(RecordingFakeSuite):
        def get_neighbors(self, node_id: str, rel_types: list[str] | None = None) -> FakeToolResult:
            self.neighbor_ids.append(node_id)
            center = self._nodes_by_id.get(node_id) or FakeNode(
                node_id, "Occupation", "unknown", "test", "src"
            )
            skills = [
                FakeNode(
                    id=f"test:skill:{i}",
                    kind="Skill",
                    label=f"skill {i}",
                    source="test",
                    source_id=f"test-skill-{i}",
                    properties={},
                )
                for i in range(1, 9)
            ]
            return FakeToolResult(
                nodes=[center, *skills],
                edges=[
                    FakeEdge(
                        type="HAS_SKILL",
                        from_id=node_id,
                        to_id=skill.id,
                        properties={"relation_type": "essential"},
                    )
                    for skill in skills
                ],
                evidence=[f"test:neighbors:{node_id}"],
            )

    suite = ManyNeighborSuite(unique_nodes=[_occ(1, "web developer")])
    state = new_session()
    runner = _runner_for(suite)
    handle_line(state, "web developer", runner=runner)
    handle_line(state, "essential skills", runner=runner)
    searches_before = list(suite.searches)
    listed = handle_line(state, "yes list down complete list", runner=runner)
    assert suite.searches == searches_before
    assert "skill 1" in listed.text
    assert "skill 8" in listed.text
    assert listed.source_note == "TEST"


def test_bare_yes_expands_truncated_connect() -> None:
    from talent_angels.session.followup import render_connect_list
    from talent_angels.session.kernel import handle_line

    suite = RecordingFakeSuite(unique_nodes=[_occ(1, "web developer")])
    state = new_session()
    runner = _runner_for(suite)
    handle_line(state, "web developer", runner=runner)
    handle_line(state, "essential skills", runner=runner)
    # one skill from default fake — expand still lists stored neighbors
    listed = handle_line(state, "yes", runner=_boom)
    assert "computer programming" in listed.text
    assert render_connect_list(state.last_result) == listed.text


class TwelveSkillSuite(RecordingFakeSuite):
    def get_neighbors(self, node_id: str, rel_types: list[str] | None = None) -> FakeToolResult:
        self.neighbor_ids.append(node_id)
        center = self._nodes_by_id.get(node_id) or FakeNode(
            node_id, "Occupation", "unknown", "test", "src"
        )
        skills = [
            FakeNode(
                id=f"test:skill:{i}",
                kind="Skill",
                label=f"skill {i}",
                source="test",
                source_id=f"test-skill-{i}",
                properties={},
            )
            for i in range(1, 13)
        ]
        for skill in skills:
            self._nodes_by_id[skill.id] = skill
        return FakeToolResult(
            nodes=[center, *skills],
            edges=[
                FakeEdge(
                    type="HAS_SKILL",
                    from_id=node_id,
                    to_id=skill.id,
                    properties={"relation_type": "essential"},
                )
                for skill in skills
            ],
            evidence=[f"test:neighbors:{node_id}"],
        )


def test_tell_me_about_skill_11_uses_last_connect_without_search() -> None:
    from talent_angels.session.kernel import handle_line

    suite = TwelveSkillSuite(unique_nodes=[_occ(1, "web developer")])
    state = new_session()
    runner = _runner_for(suite)
    handle_line(state, "web developer", runner=runner)
    handle_line(state, "essential skills", runner=runner)
    searches_before = list(suite.searches)
    occupation_id = state.binding.node.id if state.binding is not None else None
    expected = state.last_result.nodes[11].pref_label if state.last_result else None

    reply = handle_line(state, "tell me about skill 11", runner=runner)
    assert suite.searches == searches_before
    assert expected == "skill 11"
    assert "skill 11" in reply.text
    assert state.binding is not None
    assert state.binding.node.id == occupation_id
    assert state.binding.node.pref_label == "web developer"
    assert reply.source_note == "TEST"


def test_bare_11_after_connect_focuses_skill_not_search() -> None:
    from talent_angels.session.kernel import handle_line

    suite = TwelveSkillSuite(unique_nodes=[_occ(1, "web developer")])
    state = new_session()
    runner = _runner_for(suite)
    handle_line(state, "web developer", runner=runner)
    handle_line(state, "essential skills", runner=runner)
    assert state.pending == []
    searches_before = list(suite.searches)

    reply = handle_line(state, "11", runner=runner)
    assert suite.searches == searches_before
    assert "11" not in suite.searches
    assert "skill 11" in reply.text
    assert state.binding is not None
    assert state.binding.node.pref_label == "web developer"
    assert reply.source_note == "TEST"


def test_how_to_become_one_uses_bound_occupation_without_new_search() -> None:
    from talent_angels.session.kernel import handle_line

    suite = RecordingFakeSuite(unique_nodes=[_occ(1, "accountant")])
    state = new_session()
    runner = _runner_for(suite)
    handle_line(state, "what is an Accountant ?", runner=runner)
    assert state.binding is not None
    assert state.binding.node.pref_label == "accountant"
    searches_before = list(suite.searches)
    reply = handle_line(state, "how to become one ?", runner=runner)
    assert suite.searches == searches_before
    assert "how to become one" not in suite.searches
    assert state.binding.node.pref_label == "accountant"
    assert "accountant" in reply.text.lower() or "computer programming" in reply.text.lower()
    assert reply.source_note == "TEST"


def test_show_full_list_widens_occupation_picker() -> None:
    from talent_angels.session.kernel import handle_line

    nodes = [_occ(i, f"developer {i}") for i in range(1, 13)]
    suite = RecordingFakeSuite(ambiguous_nodes=nodes)
    state = new_session()
    runner = _runner_for(suite)
    listed = handle_line(state, "developer", runner=runner)
    assert listed.pending_count == 10
    assert "Showing 10 of 12" in listed.text
    widened = handle_line(state, "show full list", runner=_boom)
    assert "developer 12" in widened.text
    assert widened.pending_count == 12
    assert "skill list stored" not in widened.text.casefold()
    picked = handle_line(state, "12", runner=_boom)
    assert picked.bound_label == "developer 12"


def test_list_all_jobs_refuses_instead_of_dumping_skills() -> None:
    from talent_angels.session.kernel import handle_line

    suite = RecordingFakeSuite(unique_nodes=[_occ(1, "software developer")])
    state = new_session()
    runner = _runner_for(suite)
    handle_line(state, "software developer", runner=runner)
    handle_line(state, "essential skills", runner=runner)
    searches_before = list(suite.searches)
    reply = handle_line(state, "list all jobs", runner=_boom)
    assert suite.searches == searches_before
    lowered = reply.text.casefold()
    assert "skill 1" not in lowered
    assert "name a job" in lowered or "every occupation" in lowered


def test_expand_and_skill_n_work_while_occupation_pending_remains() -> None:
    from talent_angels.session.kernel import handle_line

    suite = TwelveSkillSuite(
        unique_nodes=[_occ(1, "web developer")],
        ambiguous_nodes=[_occ(1, "software developer"), _occ(2, "web developer")],
    )
    state = new_session()
    runner = _runner_for(suite)
    handle_line(state, "developer", runner=runner)
    assert state.pending
    web = next(c for c in state.pending if c.node.pref_label == "web developer")
    handle_line(state, str(web.number), runner=runner)
    handle_line(state, "essential skills", runner=runner)
    assert state.pending  # occupation list still on screen
    searches_before = list(suite.searches)
    listed = handle_line(state, "yes list them all", runner=runner)
    assert suite.searches == searches_before
    assert "skill 1" in listed.text
    focused = handle_line(state, "tell me about skill 11", runner=runner)
    assert suite.searches == searches_before
    assert "skill 11" in focused.text


def test_tell_me_more_expands_last_connect_without_search() -> None:
    from talent_angels.session.kernel import handle_line

    suite = TwelveSkillSuite(unique_nodes=[_occ(1, "web developer")])
    state = new_session()
    runner = _runner_for(suite)
    handle_line(state, "web developer", runner=runner)
    handle_line(state, "essential skills", runner=runner)
    searches_before = list(suite.searches)
    listed = handle_line(state, "tell me more", runner=runner)
    assert suite.searches == searches_before
    assert "skill 1" in listed.text and "skill 12" in listed.text


def test_those_skills_is_bound_connect_without_search() -> None:
    from talent_angels.session.kernel import handle_line

    suite = TwelveSkillSuite(unique_nodes=[_occ(1, "web developer")])
    state = new_session()
    runner = _runner_for(suite)
    handle_line(state, "web developer", runner=runner)
    searches_before = list(suite.searches)
    reply = handle_line(state, "those skills", runner=runner)
    assert suite.searches == searches_before
    assert "skill 1" in reply.text or "web developer" in reply.text.lower()


def test_what_is_listed_skill_label_uses_last_result() -> None:
    from talent_angels.session.kernel import handle_line

    suite = TwelveSkillSuite(unique_nodes=[_occ(1, "web developer")])
    state = new_session()
    runner = _runner_for(suite)
    handle_line(state, "web developer", runner=runner)
    handle_line(state, "essential skills", runner=runner)
    searches_before = list(suite.searches)
    reply = handle_line(state, "what is skill 11", runner=runner)
    assert suite.searches == searches_before
    assert "skill 11" in reply.text


def test_skill_99_out_of_range_is_honest_and_does_not_search() -> None:
    from talent_angels.session.kernel import handle_line

    suite = TwelveSkillSuite(unique_nodes=[_occ(1, "web developer")])
    state = new_session()
    runner = _runner_for(suite)
    handle_line(state, "web developer", runner=runner)
    handle_line(state, "essential skills", runner=runner)
    searches_before = list(suite.searches)
    occupation_id = state.binding.node.id if state.binding is not None else None

    reply = handle_line(state, "skill 99", runner=runner)
    assert suite.searches == searches_before
    assert "99" in reply.text
    assert "skill 99" not in (state.last_result.nodes[0].pref_label if state.last_result else "")
    lowered = reply.text.lower()
    assert "list" in lowered or "not" in lowered
    assert state.binding is not None
    assert state.binding.node.id == occupation_id


def test_occupation_pending_wins_for_1_when_ambiguous_locate() -> None:
    from talent_angels.session.kernel import handle_line

    suite = RecordingFakeSuite(
        ambiguous_nodes=[
            _occ(1, "software developer"),
            _occ(2, "web developer"),
        ]
    )
    runner = _runner_for(suite)
    state = new_session()
    handle_line(state, "developer", runner=runner)
    assert state.pending
    first = state.pending[0]
    searches_before = list(suite.searches)
    picked = handle_line(state, "1", runner=_boom)
    assert suite.searches == searches_before
    assert state.binding is not None
    assert state.binding.node.id == first.node.id
    assert f"Bound {first.node.pref_label}" in picked.text


def test_occupation_pending_wins_over_connect_skill_index() -> None:
    from talent_angels.session.kernel import handle_line

    suite = TwelveSkillSuite(unique_nodes=[_occ(1, "web developer")])
    state = new_session()
    runner = _runner_for(suite)
    handle_line(state, "web developer", runner=runner)
    handle_line(state, "essential skills", runner=runner)
    state.pending = [
        PendingChoice(number=1, node=_node_ref_occ("software developer")),
        PendingChoice(number=2, node=_node_ref_occ("web developer", n=2)),
    ]
    searches_before = list(suite.searches)
    picked = handle_line(state, "1", runner=_boom)
    assert suite.searches == searches_before
    assert state.binding is not None
    assert state.binding.node.pref_label == "software developer"
    assert "Bound software developer" in picked.text


def test_greeting_after_bind_passes_bound_title_into_phrasing() -> None:
    from talent_angels.llm.protocol import LLMResult, LLMUsage, Message
    from talent_angels.session.kernel import handle_line

    class LiveClient:
        provider = "litellm"
        model = "test"

        def __init__(self) -> None:
            self.calls: list[list[Message]] = []

        def complete(self, messages: list[Message], **_kwargs: object) -> LLMResult:
            self.calls.append(messages)
            return LLMResult(
                text="Still bound to nurse — ask for skills or another title.",
                provider=self.provider,
                model=self.model,
                usage=LLMUsage(),
            )

    suite = RecordingFakeSuite(unique_nodes=[_occ(9, "nurse")])
    state = new_session()
    handle_line(state, "nurse", runner=_runner_for(suite))
    client = LiveClient()
    reply = handle_line(state, "hi", runner=_boom, llm_client=client)
    blob = "\n".join(message.content for call in client.calls for message in call)
    assert "Bound title this session: nurse. Do not pretend you forgot." in blob
    assert "Bound occupation: nurse" in blob
    assert "Still bound to nurse" in reply.text


def test_help_plain_after_bind_passes_bound_title_into_phrasing() -> None:
    from talent_angels.llm.protocol import LLMResult, LLMUsage, Message
    from talent_angels.session.kernel import handle_line

    class LiveClient:
        provider = "litellm"
        model = "test"

        def __init__(self) -> None:
            self.calls: list[list[Message]] = []

        def complete(self, messages: list[Message], **_kwargs: object) -> LLMResult:
            self.calls.append(messages)
            return LLMResult(
                text="I look up titles. You already bound nurse.",
                provider=self.provider,
                model=self.model,
                usage=LLMUsage(),
            )

    suite = RecordingFakeSuite(unique_nodes=[_occ(9, "nurse")])
    state = new_session()
    handle_line(state, "nurse", runner=_runner_for(suite))
    client = LiveClient()
    handle_line(state, "what can you do", runner=_boom, llm_client=client)
    blob = "\n".join(message.content for call in client.calls for message in call)
    assert "Bound title this session: nurse. Do not pretend you forgot." in blob
    assert "Bound occupation: nurse" in blob


def test_greeting_uses_llm_when_phrasing_client_is_live() -> None:
    from talent_angels.llm.protocol import LLMResult, LLMUsage, Message
    from talent_angels.session.kernel import handle_line

    class LiveClient:
        provider = "litellm"
        model = "test"

        def complete(self, messages: list[Message], **_kwargs: object) -> LLMResult:
            return LLMResult(
                text="Hi — name a job title and I will look it up.",
                provider=self.provider,
                model=self.model,
                usage=LLMUsage(),
            )

    reply = handle_line(new_session(), "hi", runner=_boom, llm_client=LiveClient())
    assert "name a job title" in reply.text
    assert "Hey. I'm here." not in reply.text


def test_tui_renders_a_card_per_suite() -> None:
    from talent_angels.assistant.planning import build_plan_for_capability
    from talent_angels.assistant.turn import TurnOutcome
    from talent_angels.contracts import AgentResult, NodeRef
    from talent_angels.runlog import RunLogRecord
    from talent_angels.session.kernel import handle_line

    esco = AgentResult(
        capability="locate",
        suite="esco",
        nodes=[
            NodeRef(
                id="esco:occupation:dev",
                suite="esco",
                source="esco",
                source_id="esco-dev",
                kind="Occupation",
                pref_label="software developer",
            )
        ],
        confidence=0.95,
    )
    onet = AgentResult(
        capability="locate",
        suite="onet",
        nodes=[
            NodeRef(
                id="onet:occupation:15-1252.00",
                suite="onet",
                source="onet",
                source_id="15-1252.00",
                kind="Occupation",
                pref_label="Software Developers",
            )
        ],
        confidence=0.95,
    )

    def runner(question: str, **_kwargs: object) -> TurnOutcome:
        return TurnOutcome(
            capability="locate",
            plan=build_plan_for_capability("locate", suites=("esco", "onet")),
            result=esco,
            results=(esco, onet),
            answer="ignored",
            record=RunLogRecord(suite="esco,onet", plan=["locate"], question=question),
        )

    reply = handle_line(new_session(), "software developer", runner=runner)
    assert "software developer" in reply.text
    assert "Software Developers" in reply.text
    assert "Sources used: ESCO · O*NET" in reply.text
    assert reply.source_note == "ESCO · O*NET"


def test_tui_shows_onet_picker_next_to_esco_card() -> None:
    from talent_angels.assistant.planning import build_plan_for_capability
    from talent_angels.assistant.turn import TurnOutcome
    from talent_angels.contracts import AgentResult, NodeRef
    from talent_angels.runlog import RunLogRecord
    from talent_angels.session.kernel import handle_line

    esco = AgentResult(
        capability="connect",
        suite="esco",
        nodes=[
            NodeRef(
                id="esco:occupation:dev",
                suite="esco",
                source="esco",
                source_id="esco-dev",
                kind="Occupation",
                pref_label="software developer",
            )
        ],
        confidence=0.95,
    )
    onet = AgentResult(
        capability="connect",
        suite="onet",
        nodes=[
            NodeRef(
                id="onet:occupation:15-1252.00",
                suite="onet",
                source="onet",
                source_id="15-1252.00",
                kind="Occupation",
                pref_label="Software Developers",
            ),
            NodeRef(
                id="onet:occupation:15-1299.07",
                suite="onet",
                source="onet",
                source_id="15-1299.07",
                kind="Occupation",
                pref_label="Blockchain Engineers",
            ),
        ],
        warnings=["ambiguous"],
        confidence=0.7,
    )

    def runner(question: str, **_kwargs: object) -> TurnOutcome:
        return TurnOutcome(
            capability="connect",
            plan=build_plan_for_capability("connect", suites=("esco", "onet")),
            result=esco,
            results=(esco, onet),
            answer="ignored",
            record=RunLogRecord(suite="esco,onet", plan=["locate", "connect"], question=question),
        )

    state = new_session()
    reply = handle_line(state, "I want to be a software engineer", runner=runner)
    assert "software developer" in reply.text.casefold() or "ESCO" in reply.text
    assert "Blockchain Engineers" in reply.text
    assert "Sources used:" in reply.text
    assert reply.source_note == "ESCO · O*NET"
    assert state.binding is not None
    assert state.binding.node.suite == "esco"
    assert state.bindings["esco"].id == "esco:occupation:dev"
    assert "onet" not in state.bindings
    assert reply.pending_count == 2


def test_tui_pick_keeps_the_other_suite_binding() -> None:
    from talent_angels.assistant.planning import build_plan_for_capability
    from talent_angels.assistant.turn import TurnOutcome
    from talent_angels.contracts import AgentResult, NodeRef
    from talent_angels.runlog import RunLogRecord
    from talent_angels.session.kernel import handle_line

    esco_node = NodeRef(
        id="esco:occupation:dev",
        suite="esco",
        source="esco",
        source_id="esco-dev",
        kind="Occupation",
        pref_label="software developer",
    )
    onet_a = NodeRef(
        id="onet:occupation:15-1252.00",
        suite="onet",
        source="onet",
        source_id="15-1252.00",
        kind="Occupation",
        pref_label="Software Developers",
    )
    onet_b = NodeRef(
        id="onet:occupation:15-1254.00",
        suite="onet",
        source="onet",
        source_id="15-1254.00",
        kind="Occupation",
        pref_label="Web Developers",
    )
    esco = AgentResult(capability="locate", suite="esco", nodes=[esco_node], confidence=0.95)
    onet = AgentResult(
        capability="locate",
        suite="onet",
        nodes=[onet_a, onet_b],
        warnings=["ambiguous"],
        confidence=0.7,
    )

    def runner(question: str, **_kwargs: object) -> TurnOutcome:
        return TurnOutcome(
            capability="locate",
            plan=build_plan_for_capability("locate", suites=("esco", "onet")),
            result=esco,
            results=(esco, onet),
            answer="ignored",
            record=RunLogRecord(suite="esco,onet", plan=["locate"], question=question),
        )

    state = new_session()
    handle_line(state, "software engineer", runner=runner)
    picked = handle_line(state, "1", runner=_boom)
    assert state.bindings["esco"].id == esco_node.id
    assert state.bindings["onet"].id == onet_a.id
    assert "ESCO" in (picked.bound_label or "")
    assert "O*NET" in (picked.bound_label or "")


def test_show_suite_uses_stored_card_without_search() -> None:
    from talent_angels.assistant.planning import build_plan_for_capability
    from talent_angels.assistant.turn import TurnOutcome
    from talent_angels.contracts import AgentResult, NodeRef
    from talent_angels.runlog import RunLogRecord
    from talent_angels.session.kernel import handle_line

    esco = AgentResult(
        capability="locate",
        suite="esco",
        nodes=[
            NodeRef(
                id="esco:occupation:dev",
                suite="esco",
                source="esco",
                source_id="esco-dev",
                kind="Occupation",
                pref_label="software developer",
            )
        ],
        confidence=0.95,
    )
    onet = AgentResult(
        capability="locate",
        suite="onet",
        nodes=[
            NodeRef(
                id="onet:occupation:15-1252.00",
                suite="onet",
                source="onet",
                source_id="15-1252.00",
                kind="Occupation",
                pref_label="Software Developers",
            )
        ],
        confidence=0.95,
    )

    def runner(question: str, **_kwargs: object) -> TurnOutcome:
        return TurnOutcome(
            capability="locate",
            plan=build_plan_for_capability("locate", suites=("esco", "onet")),
            result=esco,
            results=(esco, onet),
            answer="ignored",
            record=RunLogRecord(suite="esco,onet", plan=["locate"], question=question),
        )

    state = new_session()
    handle_line(state, "software developer", runner=runner)
    shown = handle_line(state, "show O*NET", runner=_boom)
    assert "Software Developers" in shown.text
    assert shown.source_note == "O*NET"


def test_what_does_bound_job_do_does_not_search() -> None:
    from talent_angels.contracts import NodeRef
    from talent_angels.session.kernel import handle_line
    from talent_angels.session.models import LastBinding

    node = NodeRef(
        id="onet:occupation:29-1229.03",
        suite="onet",
        source="onet",
        source_id="29-1229.03",
        kind="Occupation",
        pref_label="Urologists",
        description="Diagnose and treat diseases of the urinary tract.",
    )
    state = new_session()
    state.binding = LastBinding(node=node)
    state.bindings = {"onet": node}
    reply = handle_line(state, "what does Urologists do ?", runner=_boom)
    assert "Diagnose and treat" in reply.text
    assert "Urologists" in reply.text
    assert "O*NET" in (reply.source_note or "")
    typo = handle_line(state, "what does Urologitst do ?", runner=_boom)
    assert "Diagnose and treat" in typo.text


def test_empty_label_connect_does_not_clobber_good_binding() -> None:
    from talent_angels.assistant.planning import build_plan_for_capability
    from talent_angels.assistant.turn import TurnOutcome
    from talent_angels.contracts import AgentResult, NodeRef
    from talent_angels.runlog import RunLogRecord
    from talent_angels.session.kernel import _set_bind, handle_line

    good = NodeRef(
        id="esco:occupation:f2b15a0e",
        suite="esco",
        source="esco",
        source_id="http://data.europa.eu/esco/occupation/f2b15a0e",
        kind="Occupation",
        pref_label="software developer",
    )
    blank = NodeRef(
        id="esco:occupation:f2b15a0e",
        suite="esco",
        source="esco",
        source_id="esco:occupation:some-graph-id",
        kind="Occupation",
        pref_label="",
    )
    state = new_session()
    _set_bind(state, good)
    assert state.binding is not None
    assert state.binding.node.pref_label == "software developer"

    connect = AgentResult(
        capability="connect",
        suite="esco",
        nodes=[blank],
        confidence=0.9,
    )

    def runner(question: str, **_kwargs: object) -> TurnOutcome:
        return TurnOutcome(
            capability="connect",
            plan=build_plan_for_capability("connect", suites=("esco",)),
            result=connect,
            results=(connect,),
            answer="On the attached maps this lines up with.",
            record=RunLogRecord(suite="esco", plan=["connect"], question=question),
        )

    reply = handle_line(state, "what skills ?", runner=runner)
    assert state.binding.node.pref_label == "software developer"
    assert "software developer" in (reply.bound_label or "")


def test_ambiguous_multi_suite_connect_never_calls_the_llm_narrative() -> None:
    """Regression, found live against a real model: when locate is ambiguous
    (Connect never ran — no unique node to hop from), the fact card handed to
    synthesize() has occupation titles only, zero skill data. In 1 of 5
    identical live runs the model invented a specific skill list for one
    candidate it picked unprompted, presented as graph fact, while the very
    next paragraph of the same reply still asked the user to pick among all
    candidates. Fixed by never calling the free-form synthesize() LLM
    narrative for this branch — synthesize_structured() produces the same
    safe framing with no model call and nothing left to hallucinate.
    """
    from talent_angels.assistant.planning import build_plan_for_capability
    from talent_angels.assistant.turn import TurnOutcome
    from talent_angels.contracts import AgentResult, NodeRef
    from talent_angels.runlog import RunLogRecord
    from talent_angels.session.kernel import handle_line

    class _RecordingClient:
        """Answers the picker's own (already-safe) one-sentence intro call, but
        records every system prompt so the test can assert that synthesize()'s
        free-form narrative system prompt specifically was never sent."""

        provider = "litellm"
        model = "test"

        def __init__(self) -> None:
            self.system_prompts: list[str] = []

        def complete(self, messages, **_kwargs):  # noqa: ANN001, ANN003
            self.system_prompts.append(messages[0].content)
            from talent_angels.llm.protocol import LLMResult, LLMUsage

            return LLMResult(
                text="Several matches came up. Which one did you mean?",
                provider=self.provider,
                model=self.model,
                usage=LLMUsage(),
            )

    esco_miss = AgentResult(capability="connect", suite="esco", warnings=["not_found"])
    onet_candidates = [
        NodeRef(
            id=f"onet:occupation:{i}",
            suite="onet",
            source="onet",
            source_id=f"29-114{i}.00",
            kind="Occupation",
            pref_label=f"Nurse Kind {i}",
        )
        for i in range(25)
    ]
    onet_ambiguous = AgentResult(
        capability="connect",
        suite="onet",
        nodes=onet_candidates,
        warnings=["ambiguous", "truncated"],
    )

    def runner(question: str, **_kwargs: object) -> TurnOutcome:
        return TurnOutcome(
            capability="connect",
            plan=build_plan_for_capability("connect", suites=("esco", "onet")),
            result=esco_miss,
            results=(esco_miss, onet_ambiguous),
            answer="ignored",
            record=RunLogRecord(suite="esco,onet", plan=["locate", "connect"], question=question),
        )

    state = new_session()
    client = _RecordingClient()
    reply = handle_line(state, "what skills does a nurse need", runner=runner, llm_client=client)

    # synthesize()'s free-form narrative system prompt was never sent — only
    # the picker's own tightly-scoped, already-safe intro call fired.
    assert not any("phrase taxonomy map facts" in prompt for prompt in client.system_prompts)
    # The safe, deterministic sentence is exactly synthesize_structured()'s output.
    assert "O*NET: several matches" in reply.text
    # The picker itself still renders every candidate.
    assert "Nurse Kind 0" in reply.text
    assert "I won't pick for you" in reply.text
    # None of the fabricated terms from the live repro can appear — there is
    # no code path left that could produce them.
    for invented in ("Active Listening", "Critical Thinking", "Monitoring"):
        assert invented not in reply.text


def test_multi_suite_pathfind_shows_redirect_not_a_generic_miss() -> None:
    """Regression, found via a live run against real Neo4j: when every attached
    suite classifies a question as pathfind (the same capability, uniformly —
    that classification never differs per suite for one question), each
    suite's AgentResult carries zero nodes and a `capability_not_implemented`
    warning. The merge path treated that identically to a genuine search miss
    (`_hit_phrase`/`synthesize_structured` cannot tell "not implemented" apart
    from "not found" — both are just zero nodes) and showed "No node for that
    phrase with today's search", which actively misleads a user into thinking
    their search failed rather than that the capability doesn't exist yet.
    """
    from talent_angels.assistant.planning import build_plan_for_capability
    from talent_angels.assistant.turn import TurnOutcome
    from talent_angels.contracts import AgentResult
    from talent_angels.runlog import RunLogRecord
    from talent_angels.session.copy import PATHFIND_REDIRECT
    from talent_angels.session.kernel import handle_line

    esco = AgentResult(
        capability="pathfind",
        suite="esco",
        warnings=["capability_not_implemented:pathfind"],
    )
    onet = AgentResult(
        capability="pathfind",
        suite="onet",
        warnings=["capability_not_implemented:pathfind"],
    )

    def runner(question: str, **_kwargs: object) -> TurnOutcome:
        return TurnOutcome(
            capability="pathfind",
            plan=build_plan_for_capability("pathfind", suites=("esco", "onet")),
            result=esco,
            results=(esco, onet),
            answer="ignored",
            record=RunLogRecord(suite="esco,onet", plan=["pathfind"], question=question),
        )

    state = new_session()
    reply = handle_line(state, "how do I get from nurse to software developer", runner=runner)

    assert reply.text == PATHFIND_REDIRECT
    assert "No node for that phrase" not in reply.text


def test_connect_bridge_intro_needs_at_least_two_suites_with_hits() -> None:
    from talent_angels.contracts import AgentResult
    from talent_angels.session.kernel import _connect_bridge_intro

    hit = AgentResult(capability="connect", suite="esco", nodes=[_node_ref_occ("nurse")], edges=[])
    miss = AgentResult(
        capability="connect", suite="onet", nodes=[], warnings=["not_found"], edges=[]
    )
    assert _connect_bridge_intro([hit]) == ""
    assert _connect_bridge_intro([hit, miss]) == ""


def test_connect_bridge_intro_names_two_suites() -> None:
    from talent_angels.contracts import AgentResult
    from talent_angels.session.kernel import _connect_bridge_intro

    esco = AgentResult(
        capability="connect", suite="esco", nodes=[_node_ref_occ("nurse", 1)], edges=[]
    )
    onet = AgentResult(
        capability="connect", suite="onet", nodes=[_node_ref_occ("nurse", 2)], edges=[]
    )
    intro = _connect_bridge_intro([esco, onet])
    assert intro.startswith("ESCO and O*NET both answer this")
    assert "not one shared id" in intro


def test_multi_suite_connect_reads_as_one_answer_not_two_blocks() -> None:
    """GAP B: a bridging sentence frames both suites' cards as one turn.

    Skills must NOT merge across suites (no cross-suite identity without a
    crosswalk) — both full per-suite cards must still be present, in their
    own `---`-separated blocks (so the TUI still renders two tables), just
    introduced by one sentence instead of appearing as two disconnected
    replies.
    """
    from talent_angels.assistant.planning import build_plan_for_capability
    from talent_angels.assistant.turn import TurnOutcome
    from talent_angels.contracts import AgentResult, EdgeRef, NodeRef
    from talent_angels.runlog import RunLogRecord
    from talent_angels.session.kernel import handle_line

    esco_center = NodeRef(
        id="esco:occupation:nurse",
        suite="esco",
        source="esco",
        source_id="n1",
        kind="Occupation",
        pref_label="nurse",
    )
    esco_skill = NodeRef(
        id="esco:skill:care",
        suite="esco",
        source="esco",
        source_id="s1",
        kind="Skill",
        pref_label="patient care",
    )
    onet_center = NodeRef(
        id="onet:occupation:29-1141.00",
        suite="onet",
        source="onet",
        source_id="29-1141.00",
        kind="Occupation",
        pref_label="Registered Nurses",
    )
    onet_skill = NodeRef(
        id="onet:skill:med",
        suite="onet",
        source="onet",
        source_id="sk1",
        kind="Skill",
        pref_label="medicine and dentistry",
    )
    esco = AgentResult(
        capability="connect",
        suite="esco",
        nodes=[esco_center, esco_skill],
        edges=[
            EdgeRef(
                type="HAS_SKILL",
                suite="esco",
                source_node_id=esco_center.id,
                target_node_id=esco_skill.id,
                properties={"relation_type": "essential"},
            )
        ],
        confidence=0.9,
    )
    onet = AgentResult(
        capability="connect",
        suite="onet",
        nodes=[onet_center, onet_skill],
        edges=[
            EdgeRef(
                type="HAS_SKILL",
                suite="onet",
                source_node_id=onet_center.id,
                target_node_id=onet_skill.id,
                properties={"relation_type": "essential"},
            )
        ],
        confidence=0.85,
    )

    def runner(question: str, **_kwargs: object) -> TurnOutcome:
        return TurnOutcome(
            capability="connect",
            plan=build_plan_for_capability("connect", suites=("esco", "onet")),
            result=esco,
            results=(esco, onet),
            answer="ignored",
            record=RunLogRecord(suite="esco,onet", plan=["connect"], question=question),
        )

    state = new_session()
    reply = handle_line(state, "what skills does a nurse need", runner=runner)

    # One bridging sentence, ahead of both blocks.
    assert reply.text.index("ESCO and O*NET both answer this") < reply.text.index("## ESCO")
    assert "not one shared id" in reply.text
    # Both full per-suite cards survive, still `---`-separated (two TUI tables).
    assert "## ESCO" in reply.text
    assert "## O*NET" in reply.text
    assert "patient care" in reply.text
    assert "medicine and dentistry" in reply.text
    assert "\n\n---\n\n" in reply.text
    # No cross-suite merging: each suite's skill only appears under its own heading.
    esco_block, _, onet_block = reply.text.partition("## O*NET")
    assert "medicine and dentistry" not in esco_block
    assert "patient care" not in onet_block


def test_single_suite_connect_has_no_bridge_sentence() -> None:
    """The bridge is a multi-suite-only concept; one suite must render exactly as before."""
    from talent_angels.assistant.planning import build_plan_for_capability
    from talent_angels.assistant.turn import TurnOutcome
    from talent_angels.contracts import AgentResult, EdgeRef, NodeRef
    from talent_angels.runlog import RunLogRecord
    from talent_angels.session.kernel import handle_line

    center = NodeRef(
        id="esco:occupation:nurse",
        suite="esco",
        source="esco",
        source_id="n1",
        kind="Occupation",
        pref_label="nurse",
    )
    skill = NodeRef(
        id="esco:skill:care",
        suite="esco",
        source="esco",
        source_id="s1",
        kind="Skill",
        pref_label="patient care",
    )
    esco = AgentResult(
        capability="connect",
        suite="esco",
        nodes=[center, skill],
        edges=[
            EdgeRef(
                type="HAS_SKILL",
                suite="esco",
                source_node_id=center.id,
                target_node_id=skill.id,
                properties={"relation_type": "essential"},
            )
        ],
        confidence=0.9,
    )

    def runner(question: str, **_kwargs: object) -> TurnOutcome:
        return TurnOutcome(
            capability="connect",
            plan=build_plan_for_capability("connect", suites=("esco",)),
            result=esco,
            results=(esco,),
            answer="ignored",
            record=RunLogRecord(suite="esco", plan=["connect"], question=question),
        )

    state = new_session()
    reply = handle_line(state, "what skills does a nurse need", runner=runner)
    assert "both answer this" not in reply.text


def test_multi_suite_goal_writes_once_not_last_wins() -> None:
    from unittest.mock import patch

    from talent_angels.assistant.llm_plan import PlanDraft
    from talent_angels.assistant.planning import build_plan_for_capability
    from talent_angels.assistant.turn import TurnOutcome
    from talent_angels.contracts import AgentResult, NodeRef
    from talent_angels.runlog import RunLogRecord
    from talent_angels.session.kernel import handle_line

    esco_node = NodeRef(
        id="esco:occupation:sw",
        suite="esco",
        source="esco",
        source_id="http://data.europa.eu/esco/occupation/sw",
        kind="Occupation",
        pref_label="software developer",
    )
    onet_node = NodeRef(
        id="onet:occupation:15-1252.00",
        suite="onet",
        source="onet",
        source_id="15-1252.00",
        kind="Occupation",
        pref_label="Software Developers",
    )
    esco = AgentResult(capability="locate", suite="esco", nodes=[esco_node], confidence=0.95)
    onet = AgentResult(
        capability="locate",
        suite="onet",
        nodes=[onet_node],
        confidence=0.9,
    )
    draft = PlanDraft(
        target="locate",
        subject="software developer",
        kind="occupation",
        profile_intent="goal",
    )

    def runner(question: str, **_kwargs: object) -> TurnOutcome:
        return TurnOutcome(
            capability="locate",
            plan=build_plan_for_capability("locate", suites=("esco", "onet")),
            result=esco,
            results=(esco, onet),
            answer="ignored",
            plan_draft=draft,
            record=RunLogRecord(suite="esco,onet", plan=["locate"], question=question),
        )

    with patch("talent_angels.session.kernel.write_goal") as mock_goal:
        state = new_session()
        handle_line(state, "set my goal to software developer", runner=runner)
        assert mock_goal.call_count == 1
        assert mock_goal.call_args.args[0].id == esco_node.id
