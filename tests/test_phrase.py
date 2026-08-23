from talent_angels.contracts import AgentResult, NodeRef
from talent_angels.llm.protocol import LLMResult, LLMUsage, Message
from talent_angels.llm.stub_client import StubLLMClient
from talent_angels.session.phrase import (
    connect_card,
    locate_card,
    phrase_chat,
    phrase_map,
    uses_chat_phrasing,
)


class _Scripted:
    provider = "litellm"
    model = "test-phrase"

    def __init__(self, text: str) -> None:
        self.text = text
        self.calls: list[list[Message]] = []

    def complete(self, messages: list[Message], **_kwargs: object) -> LLMResult:
        self.calls.append(messages)
        return LLMResult(text=self.text, provider=self.provider, model=self.model, usage=LLMUsage())


def _occ() -> NodeRef:
    return NodeRef(
        id="esco:occupation:1",
        suite="esco",
        source="esco",
        source_id="s1",
        kind="Occupation",
        pref_label="software developer",
    )


def test_stub_does_not_use_chat_phrasing() -> None:
    assert uses_chat_phrasing(None) is False
    assert uses_chat_phrasing(StubLLMClient()) is False
    assert uses_chat_phrasing(_Scripted("x")) is True


def test_phrase_chat_falls_back_for_stub() -> None:
    assert phrase_chat(StubLLMClient(), user_text="hi", fallback="Hey.", hint="") == "Hey."


def test_phrase_chat_uses_live_client() -> None:
    client = _Scripted("Hi — name a job title and I'll look it up.")
    text = phrase_chat(client, user_text="hi", fallback="Hey.", hint="User greeted you.")
    assert "name a job title" in text
    assert client.calls


def test_phrase_map_rejects_invented_numbered_list() -> None:
    result = AgentResult(capability="locate", suite="esco", nodes=[_occ()], confidence=0.95)
    client = _Scripted("1. software developer\n2. web developer")
    text = phrase_map(
        client,
        question="software developer",
        result=result,
        fallback="software developer on the map.",
        card=locate_card(result),
    )
    assert text == "software developer on the map."


def test_phrase_chat_miss_rejects_invented_skill_page() -> None:
    client = _Scripted(
        "Python Programming is a skill linked to occupations such as:\n- Software Developer"
    )
    text = phrase_chat(
        client,
        user_text="python programming",
        fallback="That's a miss, not a maybe.",
        hint="miss",
        mode="miss",
    )
    assert text == "That's a miss, not a maybe."


def test_phrase_map_unique_locate_rejects_related_titles_not_on_card() -> None:
    result = AgentResult(capability="locate", suite="esco", nodes=[_occ()], confidence=0.95)
    client = _Scripted("Located **software developer**. Also try **Software Engineer**.")
    text = phrase_map(
        client,
        question="software developer",
        result=result,
        fallback="software developer on the map.",
        card=locate_card(result),
    )
    assert text == "software developer on the map."


def test_phrase_map_keeps_allowed_bold_title() -> None:
    result = AgentResult(capability="locate", suite="esco", nodes=[_occ()], confidence=0.95)
    client = _Scripted("Located **software developer**. Ask for essential skills next.")
    text = phrase_map(
        client,
        question="software developer",
        result=result,
        fallback="fallback",
        card=locate_card(result),
    )
    assert "software developer" in text
    assert "the person is" not in text.lower()


def test_connect_card_counts_omitted_skills() -> None:
    skills = [
        NodeRef(
            id=f"esco:skill:{i}",
            suite="esco",
            source="esco",
            source_id=f"s{i}",
            kind="Skill",
            pref_label=f"skill {i}",
        )
        for i in range(8)
    ]
    result = AgentResult(
        capability="connect",
        suite="esco",
        nodes=[_occ(), *skills],
        edges=[],
    )
    card = connect_card(result, shown=5)
    assert "software developer" in card
    assert "3 more skills" in card
