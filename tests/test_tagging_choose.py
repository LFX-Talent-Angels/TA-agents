"""`choose.choose`/`choose.validate`: the model call and its code-side guardrails."""

from __future__ import annotations

from talent_angels.llm.protocol import LLMError, LLMResult, LLMUsage, Message
from talent_angels.tagging.choose import Choice, Pick, choose, validate
from talent_angels.tagging.models import CandidateRef, SearchOutcome, TagUnit


class ScriptedLLMClient:
    provider = "litellm"
    model = "test-model"

    def __init__(self, replies: list[object]) -> None:
        self.replies = list(replies)
        self.calls: list[list[Message]] = []

    def complete(
        self, messages: list[Message], *, tools: list[dict[str, object]] | None = None
    ) -> LLMResult:
        self.calls.append(messages)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return LLMResult(
            text=str(reply),
            provider=self.provider,
            model=self.model,
            usage=LLMUsage(input_tokens=5, output_tokens=5),
        )


def _unit(**overrides: object) -> TagUnit:
    base: dict[str, object] = {
        "unit_key": "k1",
        "doc_key": "doc",
        "row_id": "1",
        "kind": "performance",
        "role": "leaf",
        "text": "raw text",
        "clean_text": "define acceptance criteria",
        "context_path": [],
        "group_key": "g",
        "seed_phrases": [],
    }
    base.update(overrides)
    return TagUnit(**base)  # type: ignore[arg-type]


def _candidate(id_: str, kind: str = "Skill") -> CandidateRef:
    return CandidateRef(
        suite="esco",
        id=id_,
        source_id=f"urn:{id_}",
        kind=kind,
        label=id_,
        method="exact_pref",
        best_rank=1,
        hits=1,
        queries=["q"],
    )


def _outcome(candidates: list[CandidateRef]) -> SearchOutcome:
    return SearchOutcome(suite="esco", status="ok", routed_kind="skill", candidates=candidates)


# --- choose() -----------------------------------------------------------


def test_choose_returns_parsed_choice_on_first_try() -> None:
    client = ScriptedLLMClient(['{"fit": "exact", "picks": [{"n": 1}], "reason": "matches"}'])
    choice, stages = choose(
        client, _unit(), _outcome([_candidate("esco:skill:1")]), {}, suite_display="ESCO"
    )
    assert choice is not None
    assert choice.fit == "exact"
    assert len(stages) == 1


def test_choose_retries_once_on_bad_json_then_succeeds() -> None:
    client = ScriptedLLMClient(["not json", '{"fit": "none", "picks": [], "reason": "no match"}'])
    choice, stages = choose(
        client, _unit(), _outcome([_candidate("esco:skill:1")]), {}, suite_display="ESCO"
    )
    assert choice is not None
    assert choice.fit == "none"
    assert len(client.calls) == 2
    # Both calls were billed, the unusable one included.
    assert len(stages) == 2


def test_choose_gives_up_after_two_bad_replies() -> None:
    client = ScriptedLLMClient(["garbage", "still garbage"])
    choice, stages = choose(
        client, _unit(), _outcome([_candidate("esco:skill:1")]), {}, suite_display="ESCO"
    )
    assert choice is None
    assert len(stages) == 2  # no answer, but the cost still counts
    assert len(client.calls) == 2


def test_choose_treats_llm_error_the_same_as_bad_json() -> None:
    client = ScriptedLLMClient([LLMError("boom"), LLMError("boom again")])
    choice, _stages = choose(
        client, _unit(), _outcome([_candidate("esco:skill:1")]), {}, suite_display="ESCO"
    )
    assert choice is None


# --- validate() -----------------------------------------------------------


def test_validate_drops_out_of_range_picks() -> None:
    candidates = [_candidate("esco:skill:1")]
    choice = Choice(fit="exact", picks=[Pick(n=5)], reason="r")
    validated = validate(candidates, "performance", choice)
    assert validated.picks == []


def test_validate_drops_duplicate_picks() -> None:
    candidates = [_candidate("esco:skill:1"), _candidate("esco:skill:2")]
    choice = Choice(fit="exact", picks=[Pick(n=1), Pick(n=1), Pick(n=2)], reason="r")
    validated = validate(candidates, "performance", choice)
    assert [c.id for c, _ in validated.picks] == ["esco:skill:1", "esco:skill:2"]


def test_validate_drops_occupation_pick_for_a_non_occupation_unit() -> None:
    candidates = [_candidate("esco:occ:1", kind="Occupation"), _candidate("esco:skill:2")]
    choice = Choice(fit="exact", picks=[Pick(n=1), Pick(n=2)], reason="r")
    validated = validate(candidates, "performance", choice)
    assert [c.id for c, _ in validated.picks] == ["esco:skill:2"]


def test_validate_drops_non_occupation_pick_for_an_occupation_unit() -> None:
    candidates = [_candidate("esco:occ:1", kind="Occupation"), _candidate("esco:skill:2")]
    choice = Choice(fit="exact", picks=[Pick(n=1), Pick(n=2)], reason="r")
    validated = validate(candidates, "occupation", choice)
    assert [c.id for c, _ in validated.picks] == ["esco:occ:1"]


def test_validate_caps_at_three_picks() -> None:
    candidates = [_candidate(f"esco:skill:{i}") for i in range(1, 6)]
    choice = Choice(
        fit="partial",
        picks=[Pick(n=i, covers=f"part {i}") for i in range(1, 6)],
        reason="r",
    )
    validated = validate(candidates, "performance", choice)
    assert len(validated.picks) == 3
    assert [c.id for c, _ in validated.picks] == [
        "esco:skill:1",
        "esco:skill:2",
        "esco:skill:3",
    ]


def test_validate_keeps_covers_text_per_pick() -> None:
    candidates = [_candidate("esco:skill:1")]
    choice = Choice(fit="partial", picks=[Pick(n=1, covers="part of it")], reason="r")
    validated = validate(candidates, "performance", choice)
    assert validated.picks[0][1] == "part of it"


def test_an_exact_pick_with_null_covers_is_accepted() -> None:
    """Models write `"covers": null` on exact picks; that must not void the reply."""
    from talent_angels.tagging.choose import Choice

    choice = Choice.model_validate(
        {"fit": "exact", "picks": [{"n": 1, "covers": None}], "reason": "r", "missing_skill": None}
    )
    assert choice.picks[0].covers == ""
