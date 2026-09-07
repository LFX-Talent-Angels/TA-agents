"""Memory layer tests: stub behaviour, factory selection, and answer injection.

No Neo4j and no mem0 needed — these run everywhere, unlike the fixture-backed
suites. The mem0 client itself is exercised only for its response-normalizing
helper, which is pure and needs no SDK.
"""

from __future__ import annotations

import pytest

from talent_angels.assistant.answer import build_answer
from talent_angels.contracts import AgentResult, NodeRef
from talent_angels.llm.stub_client import StubLLMClient
from talent_angels.memory import MemoryClient, StubMemoryClient, UserMemory, get_memory_client
from talent_angels.memory.mem0_client import _to_memories


# --- stub client ---


def test_stub_stores_user_utterances_and_recalls_them() -> None:
    client = StubMemoryClient()
    client.add([{"role": "user", "content": "I am a Python backend developer"}], user_id="alice")

    hits = client.search("which Python roles suit me", user_id="alice")

    assert [m.text for m in hits] == ["I am a Python backend developer"]
    assert hits[0].score > 0


def test_stub_ignores_assistant_messages() -> None:
    """Only the user's own words are user facts; the assistant's are not."""
    client = StubMemoryClient()
    client.add(
        [
            {"role": "user", "content": "I work in Berlin"},
            {"role": "assistant", "content": "Understood, noting Berlin."},
        ],
        user_id="alice",
    )

    assert [m.text for m in client.get_all(user_id="alice")] == ["I work in Berlin"]


def test_stub_dedupes_repeated_facts_case_insensitively() -> None:
    client = StubMemoryClient()
    client.add([{"role": "user", "content": "I like Rust"}], user_id="alice")
    client.add([{"role": "user", "content": "i like rust"}], user_id="alice")

    assert len(client.get_all(user_id="alice")) == 1


def test_stub_keeps_users_isolated() -> None:
    client = StubMemoryClient()
    client.add([{"role": "user", "content": "alice likes Go"}], user_id="alice")
    client.add([{"role": "user", "content": "bob likes Java"}], user_id="bob")

    assert client.get_all(user_id="bob") == [
        UserMemory(id=client.get_all(user_id="bob")[0].id, text="bob likes Java", score=1.0)
    ]
    assert all("Java" not in m.text for m in client.get_all(user_id="alice"))


def test_stub_search_unknown_user_is_empty_not_error() -> None:
    assert StubMemoryClient().search("anything", user_id="nobody") == []


def test_stub_search_ranks_higher_overlap_first() -> None:
    client = StubMemoryClient()
    client.add([{"role": "user", "content": "I use Python and Spark daily"}], user_id="alice")
    client.add([{"role": "user", "content": "I live in Berlin"}], user_id="alice")

    hits = client.search("Python Spark", user_id="alice")

    assert hits[0].text == "I use Python and Spark daily"
    assert all("Berlin" not in h.text for h in hits)


def test_stub_search_respects_top_k() -> None:
    client = StubMemoryClient()
    for i in range(5):
        client.add([{"role": "user", "content": f"fact number {i}"}], user_id="alice")

    assert len(client.search("fact", user_id="alice", top_k=2)) == 2


def test_stub_delete_all_erases_only_that_user() -> None:
    client = StubMemoryClient()
    client.add([{"role": "user", "content": "alice fact"}], user_id="alice")
    client.add([{"role": "user", "content": "bob fact"}], user_id="bob")

    client.delete_all(user_id="alice")

    assert client.get_all(user_id="alice") == []
    assert len(client.get_all(user_id="bob")) == 1


def test_stub_satisfies_the_protocol() -> None:
    assert isinstance(StubMemoryClient(), MemoryClient)


# --- factory ---


def test_factory_defaults_to_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MEMORY_PROVIDER", raising=False)
    client = get_memory_client()
    assert isinstance(client, StubMemoryClient)
    assert client.provider == "none"


def test_factory_treats_blank_as_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MEMORY_PROVIDER", "   ")
    assert isinstance(get_memory_client(), StubMemoryClient)


def test_factory_rejects_unknown_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MEMORY_PROVIDER", "not-a-provider")
    with pytest.raises(ValueError, match="Unsupported MEMORY_PROVIDER"):
        get_memory_client()


# --- mem0 response normalization (pure; no SDK required) ---


@pytest.mark.parametrize(
    "response",
    [
        {"results": [{"id": "m1", "memory": "likes Python", "score": 0.9}]},
        [{"id": "m1", "memory": "likes Python", "score": 0.9}],
    ],
)
def test_to_memories_handles_both_mem0_shapes(response: object) -> None:
    assert _to_memories(response) == [UserMemory(id="m1", text="likes Python", score=0.9)]


def test_to_memories_skips_entries_without_text_and_tolerates_empty() -> None:
    assert _to_memories({"results": [{"id": "m1"}]}) == []
    assert _to_memories(None) == []


def test_to_memories_defaults_missing_score() -> None:
    assert _to_memories([{"id": "m1", "memory": "fact"}])[0].score == 0.0


# --- answer injection ---


def _result() -> AgentResult:
    return AgentResult(
        capability="locate",
        suite="esco",
        nodes=[
            NodeRef(
                id="esco:occupation:1",
                suite="esco",
                source="esco",
                source_id="http://example/1",
                kind="Occupation",
                pref_label="data engineer",
            )
        ],
        confidence=0.9,
    )


class _CapturingLLM(StubLLMClient):
    def __init__(self) -> None:
        super().__init__()
        self.system_prompt = ""

    def complete(self, messages):  # type: ignore[no-untyped-def]
        self.system_prompt = next(m.content for m in messages if m.role == "system")
        return super().complete(messages)


def test_structured_mode_ignores_memories_entirely() -> None:
    """Structured answers must stay byte-identical and zero-token."""
    memories = [UserMemory(id="m1", text="prefers remote work")]

    with_memory, usage = build_answer(
        _result(), llm_client=StubLLMClient(), user_memories=memories
    )
    without, _ = build_answer(_result(), llm_client=StubLLMClient())

    assert with_memory == without
    assert usage is None
    assert "remote" not in with_memory


def test_natural_mode_injects_memories_as_non_evidence() -> None:
    llm = _CapturingLLM()

    build_answer(
        _result(),
        llm_client=llm,
        mode="natural",
        user_memories=[UserMemory(id="m1", text="is a Python backend developer")],
    )

    assert "is a Python backend developer" in llm.system_prompt
    assert "not evidence" in llm.system_prompt


def test_natural_mode_without_memories_adds_no_context_block() -> None:
    llm = _CapturingLLM()

    build_answer(_result(), llm_client=llm, mode="natural", user_memories=[])

    assert "Background on the person asking" not in llm.system_prompt
