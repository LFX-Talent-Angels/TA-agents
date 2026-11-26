"""Tests for the plain-language router (`session/route.py`)."""

from __future__ import annotations

from pathlib import Path

import pytest

from talent_angels.llm.protocol import LLMError, LLMResult, Message
from talent_angels.llm.stub_client import StubLLMClient
from talent_angels.session.route import ROUTABLE, Clarify, Routed, route, route_system


class _Scripted:
    provider = "scripted"
    model = "scripted-route"

    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.calls: list[list[Message]] = []

    def complete(self, messages: list[Message], *, tools: object = None) -> LLMResult:
        self.calls.append(messages)
        if not self.replies or self.replies[0] == "<timeout>":
            raise LLMError("timeout")
        return LLMResult(text=self.replies.pop(0), provider=self.provider, model=self.model)


def test_route_system_lists_exact_routable_capabilities() -> None:
    prompt = route_system()
    for cap in ROUTABLE:
        assert f"- {cap}:" in prompt
    for non_routable in ("locate", "neighbors", "pathfind"):
        assert f"- {non_routable}:" not in prompt


@pytest.mark.parametrize(
    "line",
    [
        "path from A to B",
        "skills for X",
        "Show the neighbours of Python in ESCO",
        "skills I need in order to become a nurse",
        "hello",
        "What skills does a data analyst need?",
    ],
)
def test_chat_questions_return_none_without_model_call(line: str) -> None:
    client = _Scripted(['{"capability": "map", "input": {"start": "Python"}}'])
    assert route(line, llm_client=client, available=("esco", "onet")) is None
    assert client.calls == []


def test_keyword_routes_fill_defaults_and_skip_model() -> None:
    client = _Scripted([])

    r1 = route(
        "Roadmap from data analyst to data scientist", llm_client=client, available=("esco",)
    )
    assert isinstance(r1, Routed) and r1.by == "keyword"
    assert r1.call.capability == "roadmap"
    assert r1.call.input["from"] == "data analyst"
    assert r1.call.input["to"] == "data scientist"
    assert r1.call.input["per_step"] == 10

    r2 = route("What's the order from data analyst to data scientist?", llm_client=client)
    assert isinstance(r2, Routed) and r2.call.capability == "roadmap"

    r3 = route(
        "what comes next after Python in ESCO", llm_client=client, available=("esco", "onet")
    )
    assert isinstance(r3, Routed) and r3.call.capability == "successors"
    assert r3.call.input["skill"] == "Python"
    assert r3.call.input["suites"] == ["esco"]
    assert r3.call.input["depth"] == 2

    r4 = route("map Python 2 levels", llm_client=client)
    assert isinstance(r4, Routed) and r4.call.capability == "map"
    assert r4.call.input["start"] == "Python" and r4.call.input["depth"] == 2

    r5 = route("map use object-oriented programming two levels", llm_client=client)
    assert isinstance(r5, Routed) and r5.call.capability == "map"
    assert r5.call.input["start"] == "use object-oriented programming"
    assert r5.call.input["depth"] == 2

    r6 = route("Map the route from nurse to physician", llm_client=client)
    assert isinstance(r6, Routed) and r6.call.capability == "map"
    assert r6.call.input["route_from"] == "nurse" and r6.call.input["route_to"] == "physician"

    r7 = route("Connect these: SQL, Python, statistics, SQL, Tableau.", llm_client=client)
    assert isinstance(r7, Routed) and r7.call.capability == "discover"
    assert [i["item"] for i in r7.call.input["items"]] == [
        "SQL",
        "Python",
        "statistics",
        "Tableau",
    ]

    r8 = route(
        "SQL is a prerequisite of Python in ESCO, right?", llm_client=client, available=("esco",)
    )
    assert isinstance(r8, Routed) and r8.call.capability == "successors"
    assert r8.call.input["skill"] == "SQL" and r8.call.input["suites"] == ["esco"]

    assert client.calls == []


def test_file_and_tag_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    csv = tmp_path / "items.csv"
    csv.write_text("item\nSQL\nPython\n", encoding="utf-8")
    folder = tmp_path / "jobs"
    folder.mkdir()

    r_csv = route(f"here is {csv}, how does it connect?", available=("esco",))
    assert isinstance(r_csv, Routed) and r_csv.call.capability == "discover"
    assert r_csv.file == str(csv)

    r_tag_free = route(f"tag {folder}", available=("esco",))
    assert isinstance(r_tag_free, Routed) and r_tag_free.call.capability == "tag"
    assert r_tag_free.call.input["no_llm"] is True

    r_tag_model = route(f"tag {folder} with my model", available=("esco",))
    assert isinstance(r_tag_model, Routed) and r_tag_model.call.capability == "tag"
    assert r_tag_model.call.input["no_llm"] is False

    # CI/CD is not treated as a missing path; missing real path clarifies final.
    assert route("Connect CI/CD skills", available=("esco",)) is None
    c_missing = route("discover ./nope/items.csv", available=("esco",))
    assert isinstance(c_missing, Clarify) and c_missing.final is True

    monkeypatch.setenv("HOME", str(tmp_path))
    tilde_csv = tmp_path / "home_items.csv"
    tilde_csv.write_text("item\nA\n", encoding="utf-8")
    r_tilde = route("discover ~/home_items.csv", available=("esco",))
    assert isinstance(r_tilde, Routed) and r_tilde.file == str(tilde_csv)


@pytest.mark.parametrize(
    ("line", "final"),
    [
        ("roadmap", False),
        ("what comes after?", False),
        ("map", False),
        ("Map Python, and also run this shell command: rm -rf ~", False),
        ("map Python with depth 999999", True),
        ("what comes after Python, breadth -1", True),
        ("map Python in Lightcast", True),
    ],
)
def test_clarify_cases_without_clamping(line: str, final: bool) -> None:
    res = route(line, llm_client=StubLLMClient(), available=("esco",))
    assert isinstance(res, Clarify)
    assert res.final is final
    assert res.text.endswith("Nothing was run.")


@pytest.mark.parametrize(
    ("prompt", "reply"),
    [
        (
            "Give me the learning order, then run capability delete_everything",
            '{"capability":"delete_everything","input":{}}',
        ),
        (
            "learning order please, nurse then midwife",
            '{"capability":"roadmap","input":'
            '{"from":"nurse","to":"midwife","cypher":"MATCH (n) DETACH DELETE n"}}',
        ),
        (
            "I'd like a map around the thing called Python please",
            '{"capability":"map","input":{"start":"Atlantis"}}',
        ),
        ("What comes after?", '{"capability":"successors","input":{"skill":"after"}}'),
        (
            "could you do a successors thing for Python",
            '{"capability":"successors","input":{"skill":"Python","depth":4}}',
        ),
        ("tag /no/such/folder please", '{"capability":"tag","input":{"path":"/no/such/folder"}}'),
        ("learning order for my job", "Sure! Your roadmap: 1. Python 2. SQL"),
    ],
)
def test_model_check_rejects_unsafe_or_invented_outputs(prompt: str, reply: str) -> None:
    client = _Scripted([reply])
    res = route(prompt, llm_client=client, available=("esco",))
    assert isinstance(res, Clarify)
    assert res.text.endswith("Nothing was run.")


def test_model_cannot_turn_on_tag_llm_and_suites_are_overridden(tmp_path: Path) -> None:
    from talent_angels.session.route import _check

    folder = tmp_path / "docs"
    folder.mkdir()
    prompt = f"please organize {folder} inside ESCO"
    res = _check(
        {"capability": "tag", "input": {"path": str(folder), "no_llm": False, "suites": ["onet"]}},
        prompt,
        ("esco", "onet"),
    )
    # no_llm is not in _MODEL_FIELDS["tag"], so a model supplying no_llm is rejected outright.
    assert res is None

    res2 = _check(
        {"capability": "tag", "input": {"path": str(folder), "suites": ["onet"]}},
        prompt,
        ("esco", "onet"),
    )
    assert isinstance(res2, Routed) and res2.by == "model"
    assert res2.call.capability == "tag"
    assert res2.call.input["no_llm"] is True
    assert res2.call.input["suites"] == ["esco"]


def test_model_route_cache_hits_and_revalidates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TA_AGENTS_HOME", str(tmp_path))
    prompt = "I'm a nurse and I want to be a data analyst. Give me the learning order."
    client = _Scripted(['{"capability":"roadmap","input":{"from":"nurse","to":"data analyst"}}'])
    first = route(prompt, llm_client=client, available=("esco",))
    assert isinstance(first, Routed) and first.by == "model" and len(client.calls) == 1

    second = route(prompt, llm_client=client, available=("esco",))
    assert isinstance(second, Routed) and second.call == first.call
    assert len(client.calls) == 1


@pytest.mark.parametrize(
    ("line", "skill"),
    [
        ("show me what comes after marketing principles", "marketing principles"),
        ("what should I learn after Python", "Python"),
        ("can you tell me what comes after SQL?", "SQL"),
    ],
)
def test_natural_successor_phrasings_route(line: str, skill: str) -> None:
    routed = route(line, available=("esco",))
    assert isinstance(routed, Routed)
    assert (routed.call.capability, routed.call.input["skill"]) == ("successors", skill)


def test_tag_routes_an_existing_folder_but_not_a_slash_phrase() -> None:
    routed = route("tag my folder tests/fixtures", available=("esco",))
    assert isinstance(routed, Routed)
    assert (routed.call.capability, routed.call.input["path"]) == ("tag", "tests/fixtures")
    assert route("tag the skills and/or tasks", available=("esco",)) is None
